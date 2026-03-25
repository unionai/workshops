# Unitree Go2 Quadruped — RL Locomotion

Train a [Unitree Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2) robot dog to walk using Proximal Policy Optimization (PPO), running on Flyte.

The Go2 model comes from Google DeepMind's [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) and is loaded into Gymnasium's `Ant-v5` environment. Quadrupeds are much easier to train than humanoids — 4 legs provide inherent stability, and 12 DOF (3 per leg) means a simpler action space. Expect convergence in 10-30 iterations vs 100+ for humanoids.

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
flyte run --local workflow.py train_agent --num_iterations 3 --episodes_per_iter 5 --max_steps 100

# Remote — quick demo (should converge)
flyte run workflow.py train_agent --num_iterations 20 --episodes_per_iter 500 --max_steps 1000 --num_workers 5

# Remote — full training run
flyte run workflow.py train_agent --num_iterations 30 --episodes_per_iter 1000 --max_steps 1000 --num_workers 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_iterations` | 30 | Training rounds — each iteration collects rollouts, updates the policy, and evaluates |
| `--episodes_per_iter` | 500 | Rollouts per iteration — each rollout is one full simulation. Split evenly across workers |
| `--max_steps` | 1000 | How long each rollout lasts. With `frame_skip=4`, 1000 steps = 4000 physics timesteps |
| `--num_workers` | 5 | Parallel Flyte tasks for rollout collection — 5 workers × 32 envs = 160 parallel sims |
| `--eval_every` | 10 | How often to save a checkpoint and generate a replay video |
| `--ppo_epochs` | 5 | Gradient passes over the collected rollouts per iteration |
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
  --episodes_per_iter 500 \
  --resume_checkpoint s3://your-bucket/path/to/ppo_checkpoint.pt
```

On retries, `train_agent` automatically resumes from the last completed iteration.

## How it works

### Why quadrupeds are easier

| | Humanoid (G1) | Quadruped (Go2) |
|---|---|---|
| Legs | 2 | 4 (inherently stable) |
| Degrees of freedom | 29 | 12 |
| Balance | Must be learned | Built-in |
| Reward complexity | Needs curriculum, gait shaping | Simple velocity tracking works |
| Training time | 100+ iterations | 10-30 iterations |

### Custom reward function

The `Go2RewardWrapper` uses a simpler reward than the humanoid — quadrupeds don't need gait shaping to walk:

| Term | Weight | Purpose |
|------|--------|---------|
| Velocity tracking | +2.0 | `exp(-(x_vel - 0.6)^2 / 0.25)` — peaks at 0.6 m/s target speed |
| Alive bonus | +1.0 | Reward for not falling |
| Termination penalty | -50 | Cost for falling |
| Orientation penalty | -1.0 | Stay level (don't tilt) |
| Vertical velocity | -1.0 | Don't bounce |
| Angular velocity | -0.1 | Don't roll or tumble |
| Action rate | -0.005 | Smooth movements |
| Control cost | -0.005 | Moderate penalty for large torques |

### Task environments

| Environment | Tasks | Resources | Why |
|-------------|-------|-----------|-----|
| `worker_env` | `collect_rollouts`, `evaluate` | 2 CPU, 8 GiB | Lightweight — many run in parallel |
| `gpu_env` | `ppo_update` | 4 CPU, 16 GiB, 1 GPU | GPU for gradient updates, allocated briefly per iteration |
| `train_env` | `train_agent` | 2 CPU, 8 GiB | Orchestrator — coordinates workers and GPU task |

### PPO hyperparameters

Tuned for quadruped locomotion — more aggressive than humanoid since the problem is simpler:

| Parameter | Value | Why |
|-----------|-------|-----|
| `lr` | 3e-4 (linear decay) | 10x higher than humanoid — quadrupeds tolerate bigger updates |
| `gamma` | 0.99 | Standard discount — longer horizon is fine with 4 legs |
| `clip_eps` | 0.2 | Standard PPO clipping |
| `ppo_epochs` | 5 | Standard |
| `gae_lambda` | 0.95 | Standard GAE |
| `entropy_coef` | 0.005 | Slightly more exploration than humanoid |
| `log_std_init` | -1.0 | Looser than humanoid — quadrupeds can afford more exploration |
| `hidden_dim` | 128 | Smaller network — 12 DOF needs less capacity |

### Environment config

```python
GO2_ENV_KWARGS = {
    "healthy_z_range": (0.15, 0.6),   # Go2 body at ~0.3m standing
    "frame_skip": 4,
    "reset_noise_scale": 0.01,
    "terminate_when_unhealthy": True,
    # All built-in rewards zeroed — Go2RewardWrapper handles everything
    "healthy_reward": 0.0,
    "forward_reward_weight": 0.0,
    "ctrl_cost_weight": 0.0,
    "contact_cost_weight": 0.0,
}
```

## Project structure

```
tutorials/rl-unitree-go2/
├── workflow.py          # Everything — tasks, networks, reward wrapper, training loop, reports
├── requirements.txt     # Python dependencies
└── README.md
```

## Go2 vs G1

This tutorial is designed as a companion to [`rl-unitree-g1`](../rl-unitree-g1/). Start here to see RL locomotion work quickly, then tackle the humanoid for the full challenge:

1. **Go2** (this tutorial) — learn the pipeline, see fast results, understand reward shaping basics
2. **G1** (humanoid) — curriculum learning, gait rewards, longer training, harder reward engineering
