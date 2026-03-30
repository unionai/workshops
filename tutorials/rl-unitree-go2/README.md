# Unitree Go2 Quadruped — RL Locomotion (CPU)

Train a [Unitree Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2) robot dog to walk using Proximal Policy Optimization (PPO) on CPU MuJoCo, orchestrated by Flyte.

The Go2 model comes from Google DeepMind's [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) and is loaded into Gymnasium's `Ant-v5` environment with a custom wrapper (`Go2ActionScaler`) that handles PD position-control action scaling, action repeat for exploration, a 42-dim observation matching the MJX version, and a custom reward function with feet air time tracking.

> **See also**: [`rl-unitree-go2-mjx`](../rl-unitree-go2-mjx/) — the GPU-accelerated version using MJX + Brax PPO. 100-1000x faster, same robot.

## What it does

```
train_agent (orchestrator)
  ├── baseline eval (in-memory)        → random policy reward
  ├── for each iteration:
  │     ├── collect_rollouts(worker 1)  ┐
  │     ├── collect_rollouts(worker 2)  ├── parallel Flyte tasks (each runs 32 envs)
  │     ├── ...                         ├── step-based collection (fixed data volume)
  │     ├── collect_rollouts(worker N)  ┘
  │     ├── merge rollouts + obs normalizer stats
  │     ├── ppo_update (GPU)           → update policy + value networks
  │     ├── eval (in-memory)           → measure reward
  │     └── every N iters: evaluate    → save checkpoint + render replay
  ├── evaluate(best checkpoint)        → render best model replay
  ├── evaluate(random policy)          → render baseline replay
  └── final report                     → charts + replay tabs
```

## Setup

```bash
cd tutorials/rl-unitree-go2

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

> **Note**: The Go2 MuJoCo model is not a pip package. For remote runs, the container image clones the repo automatically. For local runs:
> ```bash
> git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie
> ```

## Run it

```bash
# Quick local test
flyte run --local workflow.py train_agent --num_iterations 3 --steps_per_iter 1000

# Remote — quick test (should show walking by iteration ~13)
flyte run workflow.py train_agent --num_iterations 15 --num_workers 10

# Remote — full training run
flyte run workflow.py train_agent --num_iterations 30 --num_workers 10
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_iterations` | 60 | Training rounds — each iteration collects rollouts, updates the policy, and evaluates |
| `--steps_per_iter` | 80000 | Total environment steps per iteration, split evenly across workers |
| `--max_steps` | 1000 | Max steps per episode during evaluation |
| `--num_workers` | 10 | Parallel Flyte tasks for rollout collection — each worker runs 32 parallel envs |
| `--eval_every` | 10 | How often to save a checkpoint and generate a replay video |
| `--ppo_epochs` | 10 | Gradient passes over the collected rollouts per iteration |
| `--lr` | 3e-4 | Initial learning rate (decays linearly to 10% by the final iteration) |

## Replay any checkpoint

Grab a `ppo_update` output from the Flyte UI and render it:

```bash
flyte run workflow.py evaluate \
  --checkpoint s3://your-bucket/path/to/ppo_checkpoint.pt \
  --label "My replay" \
  --capture_frames
```

## Resume training

Continue from any checkpoint:

```bash
flyte run workflow.py train_agent \
  --num_iterations 30 \
  --resume_checkpoint s3://your-bucket/path/to/ppo_checkpoint.pt
```

On retries, `train_agent` automatically resumes from the last completed iteration.

## How it works

### Motor control — the most important detail

The Go2 uses **position-controlled PD servos**, not torque motors. This is the single most important thing to understand about this model — getting it wrong means the robot collapses instantly and no amount of reward tuning helps.

**How it works**: The policy outputs actions in [-1, 1], which get scaled to small joint angle offsets from the default standing pose. The PD controller then drives the joints to those targets:

```
motor_targets = default_pose + action * 0.3
force = Kp * (target - joint_pos) - Kd * joint_vel     (Kp=35, Kd=0.5)
```

**The `go2.xml` vs `go2_mjx.xml` trap**: The CPU model (`go2.xml`) defines actuators as `<motor>` elements, which in MuJoCo means `biastype="none"` (pure torque, no position feedback). The MJX model (`go2_mjx.xml`) uses `<general biastype="affine">` (PD position control). This means:

- Setting `biasprm[:, 1] = -35.0` on the CPU model does **nothing** — MuJoCo silently ignores bias terms when `biastype="none"`
- The code must explicitly set `actuator_biastype` to `mjBIAS_AFFINE` for PD control to work
- Without this, `force = 35 * ctrl` (constant torque, no feedback) instead of `force = 35 * (target - pos) - 0.5 * vel` (PD control)
- The robot applies constant torque with no position correction and falls over in ~12 steps

The fix in code:
```python
import mujoco
mj_model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE  # Enable PD control
mj_model.actuator_gainprm[:, 0] = 35.0       # P-gain (position)
mj_model.actuator_biasprm[:, 0] = 0.0        # no constant bias
mj_model.actuator_biasprm[:, 1] = -35.0      # position feedback
mj_model.actuator_biasprm[:, 2] = -0.5       # velocity feedback (D-gain)
```

### Action repeat — why random exploration needs it

With PD position control, the controller smoothly drives joints to targets. If you command a new random target every 0.01s (frame_skip=5), the joints barely move before getting a new command — the robot vibrates in place with zero net forward velocity. No reward signal, no learning.

**Action repeat = 4** holds each policy action for 4 env steps (0.04s total = 20 physics substeps). This gives the PD controller time to actually move the legs, producing sustained movements that generate forward velocity during exploration. Without action repeat, the policy converges to "stand still" because random per-step noise produces zero net displacement.

### Initial velocity push

Each episode starts with a random forward velocity (0.2-1.0 m/s). This breaks the exploration deadlock — the robot experiences forward movement from the start, giving the policy a gradient signal to learn from. Without this, the policy has to discover coordinated walking from scratch through random noise, which is extremely unlikely with PD position control.

### Reward normalization

Rewards are divided by their batch standard deviation before computing GAE (Generalized Advantage Estimation). This is critical for CPU PPO — without it, constant per-step terms (like alive bonus) dominate the reward signal over variable terms (like forward velocity), and the policy converges to standing still. Brax's PPO does this internally, which is why the MJX version works without explicit normalization.

### Custom observation (42 dims)

Matches the MJX version exactly — no raw quaternion or contact forces from Ant-v5:

| Component | Dims | Description |
|-----------|------|-------------|
| Projected gravity | 3 | [0,0,-1] rotated by inverse body quaternion — tells policy which way is "up" |
| Angular velocity | 3 | Body angular velocity, scaled by 0.25 |
| Joint deltas | 12 | Current joint angles minus default standing pose |
| Joint velocities | 12 | Joint angular velocities |
| Last action | 12 | Previous policy output (action history) |

### Reward function

| Term | Weight | Purpose |
|------|--------|---------|
| Forward velocity | +3.0 | `3.0 * x_vel` — reward moving forward |
| Alive bonus | +0.1 | Tiny — stabilizes training without encouraging standing still |
| Orientation penalty | -2.0 | `-2.0 * (gx² + gy²)` — penalize tilting |
| Control cost | -0.02 | `-0.02 * sum(action²)` — penalize large actions |
| Feet air time | +1.0 | Rewards feet landing after ~0.15s airborne — encourages trotting gait |

### Step-based collection

Rollout collection is **step-based** (not episode-based). Each worker collects a fixed number of environment steps (default 8000). When an episode ends, the environment auto-resets and collection continues. This guarantees consistent data volume regardless of episode length.

### Task environments

| Environment | Tasks | Resources | Why |
|-------------|-------|-----------|-----|
| `worker_env` | `collect_rollouts`, `evaluate` | 2 CPU, 8 GiB | Lightweight — many run in parallel |
| `gpu_env` | `ppo_update` | 4 CPU, 16 GiB, 1 GPU | GPU for gradient updates, allocated briefly per iteration |
| `train_env` | `train_agent` | 2 CPU, 8 GiB | Orchestrator — coordinates workers and GPU task |

### PPO hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | 3e-4 (linear decay to 10%) | Standard PPO learning rate |
| `gamma` | 0.97 | Slightly shorter horizon |
| `clip_eps` | 0.2 | Standard PPO clipping |
| `ppo_epochs` | 10 | More passes for better sample efficiency |
| `gae_lambda` | 0.95 | Standard GAE |
| `entropy_coef` | 0.01 | Moderate — maintain exploration without dominating the loss |
| `log_std_init` | -0.5 | std=0.6 — enough exploration to stumble into forward motion |
| `hidden_dim` | 256 | Two hidden layers with SiLU activation |
| `batch_size` | 256 | Mini-batch size for PPO updates |
| `grad_clip` | 2.0 | Gradient clipping norm |

### Environment config

```python
GO2_ENV_KWARGS = {
    "healthy_z_range": (0.18, 0.6),   # Match MJX termination bounds
    "frame_skip": 5,                   # Match MJX n_frames=5
    "reset_noise_scale": 0.01,         # Small noise near standing pose
    "terminate_when_unhealthy": True,
    # All built-in rewards zeroed — Go2ActionScaler handles everything
    "healthy_reward": 0.0,
    "forward_reward_weight": 0.0,
    "ctrl_cost_weight": 0.0,
    "contact_cost_weight": 0.0,
}
```

## Lessons learned

Getting the Go2 to walk on CPU MuJoCo required solving several problems that don't exist in the MJX/Brax version:

1. **Actuator biastype** — `go2.xml` uses `<motor>` (torque), `go2_mjx.xml` uses `<general biastype="affine">` (PD). Must manually convert to affine on CPU.
2. **Init qpos** — Gymnasium resets to `model.qpos0` (joints at 0, legs extended), not the keyframe standing pose. Must override `init_qpos`.
3. **Action repeat** — PD position control + per-step Gaussian noise = vibrating in place. Need action repeat to produce sustained movements.
4. **Initial velocity push** — Random exploration with PD control doesn't produce forward movement. Starting with momentum gives the policy something to learn from.
5. **Reward normalization** — Without it, alive bonus dominates over velocity signal across long episodes. Brax handles this internally.
6. **Alive bonus scaling** — 0.5/step × 1000 steps = 500 free reward for standing still. Reduced to 0.1.

## Project structure

```
tutorials/rl-unitree-go2/
├── workflow.py          # Everything — wrapper, networks, PPO, training loop, reports
├── requirements.txt     # Python dependencies
└── README.md
```

## CPU vs MJX

This is the CPU version. For comparison:

| | CPU (this tutorial) | MJX ([`rl-unitree-go2-mjx`](../rl-unitree-go2-mjx/)) |
|---|---|---|
| Physics engine | MuJoCo (CPU) | MJX (GPU via JAX) |
| Parallel envs | 32 per worker × N workers | 4096 on one GPU |
| PPO implementation | Custom (PyTorch) | Brax built-in (JAX) |
| Training speed | ~30 min for 30 iterations | ~5 min for 50M timesteps |
| XML model | `scene.xml` → `go2.xml` | `scene_mjx.xml` → `go2_mjx.xml` |
| Actuator type | `<motor>` — must convert to PD | `<general biastype="affine">` — PD built-in |
| Exploration tricks | Action repeat + velocity push | Not needed (4096 env diversity) |
| Reward normalization | Manual (divide by batch std) | Built into Brax PPO |
| Use case | Learning PPO internals, debugging | Production training speed |
