# Unitree G1 Humanoid — RL Locomotion

Train a [Unitree G1](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) humanoid robot to walk using Proximal Policy Optimization (PPO), running on Flyte.

The G1 model comes from Google DeepMind's [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) and is loaded into Gymnasium's `Humanoid-v5` environment. Training runs as a single Flyte task with live progress reports — reward curves, loss charts, and a before/after replay comparison at the end.

## What it does

```
train_agent (orchestrator)
  ├── evaluate(random policy)        → baseline reward
  ├── for each iteration:
  │     ├── collect_rollouts         → run episodes with current policy
  │     ├── ppo_update               → update policy + value networks
  │     └── evaluate                 → measure reward (no rendering)
  ├── evaluate(best checkpoint)      → render best model replay
  ├── evaluate(random policy)        → render baseline replay
  └── final report                   → charts + before/after replays
```

Each box is a **Flyte task** — tracked, logged, and runnable locally or on a cluster. Frame rendering only happens at the end (best model vs random baseline), keeping training fast and disk-friendly.

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
  --max_steps 1000

# Remote — full training run
flyte run workflow.py train_agent \
  --num_iterations 60 \
  --episodes_per_iter 500 \
  --max_steps 1000
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_iterations` | 15 | Number of PPO training iterations |
| `--episodes_per_iter` | 20 | Episodes to collect per iteration |
| `--ppo_epochs` | 15 | Gradient updates per iteration |
| `--max_steps` | 1000 | Max simulation steps per episode |
| `--lr` | 1e-4 | Initial learning rate (decays linearly) |

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

**PPO** (Proximal Policy Optimization) trains two neural networks:
- **Policy network** — maps observations (joint angles, velocities, contacts) to actions (joint torques)
- **Value network** — estimates expected future reward from a given state

Each iteration: collect episodes with the current policy, compute advantages using GAE (Generalized Advantage Estimation), then update both networks with clipped surrogate loss.

**Key hyperparameters** tuned for the G1:
- **clip_eps = 0.1** — tighter than the usual 0.2 because bipedal walkers are sensitive to large policy changes
- **entropy_coef = 0.01** — small exploration bonus; too high and the policy optimizes for randomness over reward
- **lr = 1e-4 with linear decay** — conservative learning rate that decays to 1e-5 by the end of training
- **hidden_dim = 256** — larger network than simpler environments since the G1 has ~70 observation dims and ~20 action dims

**Environment config** (`G1_ENV_KWARGS`):
- `healthy_z_range = (0.5, 1.5)` — the G1 is ~1.27m tall; episode ends if center of mass leaves this range
- `healthy_reward = 5.0` — strong incentive to stay upright
- `forward_reward_weight = 1.25` — reward forward velocity
- `ctrl_cost_weight = 0.1` — penalize large joint torques
- `frame_skip = 5` — repeat each action for 5 simulation steps

## Project structure

```
tutorials/rl-unitree-g1/
├── workflow.py          # Everything — tasks, networks, training loop, reports
└── requirements.txt     # Python dependencies
```

The G1 MuJoCo model is cloned into the container at build time from [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) — no local model files needed for remote runs.
