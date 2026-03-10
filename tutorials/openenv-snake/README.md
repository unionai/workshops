# Snake RL with OpenEnv + Flyte

Train an LLM to play Snake using GRPO (Group Relative Policy Optimization) with a custom OpenEnv environment and visual replay in Flyte reports.

## What's Here

| File | Description |
|------|-------------|
| `snake_rl.py` | Flyte pipeline — baselines, GRPO training, eval, HTML report with visual replay |
| `snake_env/models.py` | Pydantic models (SnakeAction, SnakeObservation, SnakeState) |
| `snake_env/server/environment.py` | Snake game logic implementing OpenEnv's Environment interface |
| `snake_env/server/app.py` | FastAPI server exposing the Snake env via OpenEnv protocol |
| `snake_env/client.py` | EnvClient subclass for connecting to the Snake server |
| `requirements.txt` | Dependencies |

## Concepts

- **Custom OpenEnv Environment** — Build your own RL environment with OpenEnv's Gymnasium-style API
- **GRPO Training** — Group Relative Policy Optimization for LLM RL
- **Reward Shaping** — +1.0 apple, -1.0 death, +/-0.01 distance-based shaping
- **Visual Replay** — HTML canvas replay with frame slider embedded in Flyte reports
- **Eval Tracking** — Score improvement over training with baseline comparisons

## Setup

```bash
cd tutorials/openenv-snake

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Run Locally

**Terminal 1 — Start the Snake env server:**

```bash
cd tutorials/openenv-snake
python -m snake_env.server.app
```

**Terminal 2 — Run the training pipeline:**

```bash
cd tutorials/openenv-snake

# Quick test (small run)
flyte run --local snake_rl.py pipeline \
  --training_steps 2 \
  --rollouts_per_step 4 \
  --eval_episodes 10

# Full local run
flyte run --local snake_rl.py pipeline \
  --training_steps 5 \
  --rollouts_per_step 16 \
  --eval_episodes 30
```

## Deploy & Run Remotely

To run the training pipeline on Union with a GPU, you first need to deploy the Snake env server as a Union app.

### 1. Deploy the Snake env server

The server is a standard FastAPI app. Deploy it using `flyte deploy`:

```bash
flyte deploy deploy_server.py env
```

This deploys `snake_env/server/app.py` as a Union app. Once deployed, grab the app URL from the Union dashboard.

### 2. Run the training pipeline

```bash
flyte run snake_rl.py pipeline \
  --env_url https://tiny-voice-dabc2.apps.demo.hosted.unionai.cloud\
  --training_steps 10 \
  --rollouts_per_step 32
```

## Architecture

```
pipeline
  ├── [env server running on localhost:8000]
  ├── eval_baselines()        → Random + greedy avg scores
  ├── eval_model()            → Pre-training score (step 0)
  ├── for step in 1..N:
  │     ├── train_step()      → Collect rollouts, GRPO update, save checkpoint
  │     └── eval_model()      → Eval checkpoint, record score + best replay
  └── generate HTML report    → Score charts + visual canvas replay with slider
```

## Report Output

The Flyte report includes:
1. **Score Over Training** — Agent avg score vs random/greedy baselines
2. **Training Reward** — Average shaped reward per step
3. **Direction Distribution** — UP/DOWN/LEFT/RIGHT fractions over time
4. **Visual Replay** — Interactive canvas replay with episode selector and frame slider