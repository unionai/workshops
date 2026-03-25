# Unitree Go2 (MJX) — GPU-Accelerated RL Locomotion

Train a [Unitree Go2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go2) robot dog to walk using **MJX** (MuJoCo on GPU) with Brax PPO. 4096 parallel environments on a single GPU — full training in minutes, not hours.

## CPU MuJoCo vs MJX

| | CPU MuJoCo (`rl-unitree-go2`) | MJX (`rl-unitree-go2-mjx`) |
|---|---|---|
| Physics engine | CPU | GPU (JAX/XLA) |
| Parallel envs | 160 (10 workers × 32) | 4,096 (single GPU) |
| Training time | ~30 min | ~5-15 min |
| Framework | PyTorch + Gymnasium | JAX + Brax |
| Parallelism | Flyte workers | `jax.vmap` on GPU |

MJX compiles the entire physics simulation into XLA operations that run on GPU. Combined with Brax's PPO (also JAX-based), the entire training loop — simulation, reward computation, policy update — stays on GPU with zero CPU-GPU transfers.

## What it does

```
train_agent (orchestrator)
  ├── train_on_gpu          → Brax PPO + MJX (4096 envs on 1 GPU)
  ├── render_replay         → CPU MuJoCo rendering (MJX can't render)
  └── final report          → training curves + replay
```

The architecture is simpler than the CPU version because everything fits on one GPU — no need for distributed workers. Replay rendering is the only part that runs on CPU (MJX doesn't support rendering).

## Setup

```bash
cd tutorials/rl-unitree-go2-mjx

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

Requires a CUDA GPU for training (the `train_on_gpu` task).

## Run it

```bash
# Remote — quick demo
flyte run workflow.py train_agent --num_timesteps 5000000 --num_envs 2048

# Remote — full training
flyte run workflow.py train_agent --num_timesteps 20000000 --num_envs 4096

# Remote — maximum speed
flyte run workflow.py train_agent --num_timesteps 50000000 --num_envs 8192
```

| Flag | Default | Description |
|------|---------|-------------|
| `--num_timesteps` | 20,000,000 | Total environment steps across all parallel envs |
| `--num_envs` | 4,096 | Parallel environments on GPU — more = faster but more VRAM |
| `--episode_length` | 1,000 | Steps per episode |
| `--num_evals` | 10 | Number of evaluation checkpoints during training |

## How it works

### MJX — MuJoCo on GPU

MJX compiles MuJoCo physics into JAX/XLA operations. The key pattern:

```python
import mujoco
from mujoco import mjx

# Load model to GPU
mj_model = mujoco.MjModel.from_xml_path("scene_mjx.xml")
mjx_model = mjx.put_model(mj_model)

# Vectorize across 4096 environments with jax.vmap
jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
batch = jit_step(mjx_model, batch_of_data)  # 4096 envs stepped in parallel
```

### Brax PPO

Brax provides a fully JAX-based PPO implementation. The entire training loop is JIT-compiled:
- **Rollout collection** — vmapped environment steps on GPU
- **Advantage computation** — GAE on GPU
- **Policy update** — gradient descent on GPU
- **No CPU-GPU transfers** during training

### Custom reward

Same reward structure as the CPU version but implemented in JAX:

| Term | Weight | Purpose |
|------|--------|---------|
| Velocity tracking | +2.0 | Exponential kernel at 0.6 m/s target |
| Alive bonus | +1.0 | Reward for not falling |
| Termination penalty | -50 | Cost for falling |
| Orientation penalty | -1.0 | Stay level |
| Vertical velocity | -1.0 | Don't bounce |
| Control cost | -0.005 | Smooth actions |

### Why rendering is on CPU

MJX runs physics on GPU but doesn't support rendering (no OpenGL/EGL). Replay frames are rendered using standard CPU MuJoCo after training completes. The trained policy is exported and run on a CPU MuJoCo environment for visualization.

## Project structure

```
tutorials/rl-unitree-go2-mjx/
├── workflow.py          # MJX env, Brax PPO training, replay rendering
├── requirements.txt     # Python dependencies (mujoco-mjx, jax, brax)
└── README.md
```

## Tutorial progression

1. **[rl-unitree-go2](../rl-unitree-go2/)** — CPU MuJoCo, distributed workers, learn the fundamentals
2. **rl-unitree-go2-mjx** (this) — GPU MuJoCo, 100x faster, same robot
3. **[rl-unitree-g1](../rl-unitree-g1/)** — humanoid challenge, curriculum learning, reward engineering
