# Unitree G1 Humanoid — RL Locomotion

Train a [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) humanoid robot to walk using Proximal Policy Optimization (PPO), running on Flyte.

The G1 model comes from Google DeepMind's [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) and is loaded into Gymnasium's `Humanoid-v5` environment. Training uses **vectorized parallel environments** (32 simultaneous simulations), **observation normalization**, and a **custom reward function** inspired by [MuJoCo Playground](https://playground.mujoco.org/) and [legged_gym](https://github.com/leggedrobotics/legged_gym). Live progress reports show reward curves, loss charts, and a before/after replay comparison.

## What it does

```
train_agent (orchestrator)
  ├── baseline eval (in-memory)        → random policy reward
  ├── for each iteration:
  │     ├── collect_rollouts(worker 1)  ┐
  │     ├── collect_rollouts(worker 2)  ├── parallel Flyte tasks (each runs 32 envs)
  │     ├── collect_rollouts(worker 3)  ├──
  │     ├── collect_rollouts(worker 4)  ├──
  │     ├── collect_rollouts(worker 5)  ┘
  │     ├── merge rollouts + obs normalizer stats
  │     ├── ppo_update (in-memory)      → update policy + value networks
  │     ├── eval (in-memory)            → measure reward
  │     └── every N iters: evaluate     → save checkpoint + render replay
  ├── evaluate(best checkpoint)         → render best model replay
  ├── evaluate(random policy)           → render baseline replay
  └── final report                      → charts + replay tabs
```

Rollout collection fans out across **parallel Flyte tasks** — each worker runs 32 MuJoCo simulations simultaneously, so 5 workers × 32 envs = 160 simulations in parallel. PPO update and evaluation run in-memory (no serialization overhead). Frame rendering only happens periodically and at the end.

## Setup

```bash
cd tutorials/rl-unitree-g1

# Create virtual environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

> **Note**: The G1 MuJoCo model is not a pip package. For remote runs, the container image clones the repo automatically via `Image.with_commands()`. For local runs, you'll need to clone it yourself:
> ```bash
> git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git /opt/mujoco_menagerie
> ```

## Run it

```bash
# Quick local test
flyte run --local workflow.py train_agent \
  --num_iterations 3 \
  --episodes_per_iter 5 \
  --max_steps 100

# Remote — moderate training run
flyte run workflow.py train_agent \
  --num_iterations 30 \
  --episodes_per_iter 200 \
  --max_steps 1000 \
  --num_workers 5

# Remote — full training run
flyte run workflow.py train_agent \
  --num_iterations 150 \
  --episodes_per_iter 500 \
  --max_steps 1000 \
  --num_workers 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_iterations` | 15 | Training rounds — each iteration collects rollouts, updates the policy, and evaluates |
| `--episodes_per_iter` | 20 | Rollouts per iteration — each rollout is one full simulation (robot starts standing, acts until it falls or hits the step limit). Split evenly across workers |
| `--max_steps` | 1000 | How long each rollout lasts — more steps means the robot gets to walk further before the episode ends. With `frame_skip=4`, 1000 steps = 4000 physics timesteps |
| `--num_workers` | 5 | Parallel Flyte tasks for rollout collection — instead of one machine running all rollouts sequentially, N machines each collect a share simultaneously |
| `--eval_every` | 20 | How often to save a checkpoint and generate a replay video (e.g., every 20 iterations). Replays show up as tabs in the Flyte report |
| `--ppo_epochs` | 5 | Gradient passes over the collected rollouts per iteration |
| `--lr` | 3e-5 | Initial learning rate (decays linearly to 10% by the final iteration) |

## What you'll see

**During training** — the Flyte report updates after each iteration with:
- A progress table showing iteration, reward, and loss
- Live reward and loss charts

**After training** — the report adds:
- A summary table (baseline vs best reward, improvement)
- Two replay tabs: **Random Policy** (before) and **Best Model** (after) with a frame-by-frame scrubber

## Replay any checkpoint

The `evaluate` task can be run standalone on any checkpoint from a previous run:

```bash
flyte run --local workflow.py evaluate \
  --checkpoint <path-to-checkpoint.pt> \
  --label "My replay" \
  --capture_frames true
```

## How it works

### Custom reward function

The default Gymnasium Humanoid-v5 reward (`alive_bonus + forward_velocity - ctrl_cost`) is too simple for real walking — it rewards falling forward as much as actual locomotion. The `G1RewardWrapper` replaces it with reward terms used by real humanoid locomotion projects:

| Term | Weight | Purpose |
|------|--------|---------|
| Velocity tracking | +1.5 | `exp(-(x_vel - 0.8)^2 / 0.25)` — peaks at 0.8 m/s target speed, penalizes both standing still and going too fast |
| Alive bonus | +0.5 | Small incentive to stay alive (not dominant) |
| Termination penalty | -50 | Large cost for falling — makes dying very expensive |
| Orientation penalty | -2.0 | Penalize torso tilt from upright (based on quaternion) |
| Vertical velocity | -2.0 | Penalize bouncing or falling (`z_vel^2`) |
| Angular velocity | -0.15 | Penalize roll/pitch rate — don't tumble |
| Action rate | -0.01 | Penalize jerky actions (`(a_t - a_{t-1})^2`) — smooth walking |
| Control cost | -0.01 | Moderate penalty for large joint torques |

The key insight: **velocity tracking with an exponential kernel** creates a clear optimum at the target speed. Unlike raw forward velocity reward, there's no incentive to fall forward faster — the reward peaks at 0.8 m/s and decays for anything faster or slower.

This reward structure follows the same pattern used by real humanoid locomotion projects — [legged_gym](https://github.com/leggedrobotics/legged_gym) (ETH Zurich / NVIDIA) and [MuJoCo Playground](https://playground.mujoco.org/) (Google DeepMind) both use velocity tracking with exponential kernels, orientation penalties, and stability rewards as the foundation for sim-to-real humanoid walking. Production systems add domain randomization, curriculum learning, and foot contact rewards on top, but the core reward structure is the same.

### PPO (Proximal Policy Optimization)

Trains two neural networks:
- **Policy network** — maps observations (joint angles, velocities, contacts) to actions (joint torques)
- **Value network** — estimates expected future reward from a given state

Each iteration: collect episodes with the current policy, compute advantages using GAE (Generalized Advantage Estimation), then update both networks with clipped surrogate loss.

### Distributed rollout collection

Rollout collection is the training bottleneck — the robot needs to run hundreds of simulations each iteration. We parallelize this at two levels:

1. **Across workers** — `train_agent` fans out N parallel Flyte tasks (default 5), each collecting a share of the episodes. With 500 episodes and 5 workers, each worker collects 100.
2. **Within each worker** — each worker uses `gymnasium.vector.SyncVectorEnv` to run **32 MuJoCo simulations simultaneously**. Each `env.step()` advances all 32 environments at once.

Combined: 5 workers × 32 envs = **160 parallel simulations per iteration**. After all workers finish, their episodes and observation normalizer statistics are merged, and PPO updates the policy in-memory.

### Observation normalization

Raw observations from the G1 environment have wildly different scales — joint angles might be -3 to 3, velocities -10 to 10, contact forces 0 to 1000. Feeding these directly to the neural network makes training unstable.

The `ObsNormalizer` tracks a running mean and standard deviation using Welford's online algorithm, then normalizes inputs to roughly zero-mean, unit-variance. The normalizer state is saved in checkpoints so `evaluate` uses the same statistics learned during training.

### PPO hyperparameters

Tuned based on [RL Zoo3](https://github.com/DLR-RM/rl-baselines3-zoo) Humanoid configs and MuJoCo locomotion best practices:

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | 3e-5 (linear decay) | Conservative — humanoids need small, stable updates |
| `gamma` | 0.95 | Shorter horizon than default 0.99 — humanoids need to react to immediate balance |
| `clip_eps` | 0.2 | Standard PPO clipping range |
| `ppo_epochs` | 5 | Fewer passes to avoid overfitting on each batch |
| `gae_lambda` | 0.9 | Slightly lower for less variance in advantage estimates |
| `entropy_coef` | 0.002 | Very small — enough exploration without overwhelming reward |
| `log_std_init` | -2.0 | Tight initial actions — start conservative, let the policy learn to be bold |
| `grad_norm` | 2.0 | Looser than default for more expressive updates |
| `hidden_dim` | 256 | Two-layer Tanh network, sized for ~70 obs dims and ~20 action dims |
| `parallel_envs` | 32 | Vectorized envs per worker — balances throughput with memory on 8 CPU cores |
| `num_workers` | 5 | Parallel Flyte tasks for rollout collection — 5 workers × 32 envs = 160 parallel sims |

### Environment config

```python
G1_ENV_KWARGS = {
    "healthy_z_range": (0.6, 1.4),   # G1 pelvis at ~0.79m; tight range
    "frame_skip": 4,                  # finer control than default 5
    "reset_noise_scale": 0.005,       # start near standing
    "terminate_when_unhealthy": True,
    # All built-in rewards zeroed — G1RewardWrapper handles everything
    "healthy_reward": 0.0,
    "forward_reward_weight": 0.0,
    "ctrl_cost_weight": 0.0,
    "contact_cost_weight": 0.0,
}
```

### What we tried

Getting a humanoid to walk required iterating on several approaches:

| Run | Config | Best Reward | Result |
|-----|--------|-------------|--------|
| 1 | Default PPO, high entropy (0.05), lr=3e-4 | Collapsed | Policy entropy collapse — loss kept decreasing but reward crashed |
| 2 | entropy=0.01, lr=1e-4, clip=0.1 | 614 | Stable but plateaued — stood and shuffled |
| 3 | forward_reward=5.0, healthy=3.0 | 541 | Learned to fall forward for reward (reward hacking) |
| 4 | forward_reward=3.0, healthy=5.0 | 742 | Jittery small steps, moved forward then fell |
| 5 | + obs normalization + parallel envs | 959 | Higher reward but still slow-falling — obs norm made it better at exploiting the simple reward |
| 6 | Custom reward wrapper + tuned PPO hyperparams | TBD | Velocity tracking, death penalty, orientation/stability rewards |

Key lessons:
- **Observation normalization** was the biggest single improvement for training stability
- **Reward function design** matters more than hyperparameter tuning — the agent will always find the easiest way to maximize reward
- **Velocity tracking** (exponential kernel) is fundamentally different from raw forward velocity — it creates a target speed rather than rewarding "go as fast as possible"
- **Death penalty** (-100) makes falling expensive enough that the policy learns to stay upright

## Project structure

```
tutorials/rl-unitree-g1/
├── workflow.py          # Everything — tasks, networks, reward wrapper, training loop, reports
├── requirements.txt     # Python dependencies
└── README.md
```

The G1 MuJoCo model is cloned into the container at build time from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — no local model files needed for remote runs.
