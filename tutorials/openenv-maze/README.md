# Maze RL with OpenEnv + Flyte

Train an LLM to navigate randomly generated mazes using GRPO (Group Relative Policy Optimization).

The agent learns to find the exit in DFS-generated 8x8 mazes. Improvement is visually obvious — random wandering evolves into directed pathfinding, shown in an interactive canvas replay with path trace overlay.

## Architecture

```
maze_env/                    # OpenEnv environment package
  models.py                  # MazeAction, MazeObservation, MazeState
  server/
    environment.py           # Maze game logic (DFS generation, movement, rewards)
    app.py                   # Local dev server

maze_rl.py                   # Single-file Flyte pipeline (models + client + GRPO + eval + HTML replay)
deploy_server.py             # Union app deployment (FastAPIAppEnvironment)
```

## Setup

```bash
cd tutorials/openenv-maze
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Local Run

Start the environment server:
```bash
python -m maze_env.server.app
```

In another terminal, run the training pipeline:
```bash
flyte run --local maze_rl.py pipeline --training_steps 2 --rollouts_per_step 4 --eval_episodes 10
```

## Remote Deployment

Deploy the maze environment server:
```bash
flyte deploy deploy_server.py env
```

Run the pipeline against the deployed server:
```bash
flyte run maze_rl.py pipeline --env_url <APP_URL> --training_steps 10
```

## Report Output

The pipeline generates a Flyte report with:
- **Results table** — solve rate, avg steps, avg reward for all policies
- **Solve Rate Over Training** — agent line + baseline dashes
- **Avg Steps to Solve** — efficiency improvement (lower = better)
- **Direction Distribution** — UP/DOWN/LEFT/RIGHT fractions over training
- **Interactive Replay** — canvas visualization with path trace, episode selector, frame slider, play/pause

## Performance Tuning

By default the pipeline uses bfloat16 and gradient checkpointing to fit on smaller GPUs and Apple Silicon. On larger hardware you can disable these for faster training:

```bash
# Default (memory-optimized for local dev / small GPUs)
flyte run --local maze_rl.py pipeline --training_steps 3

# Full precision, no checkpointing (faster on A100/H100)
flyte run maze_rl.py pipeline --training_steps 10 --use_bfloat16 false --gradient_checkpointing false
```

| Flag | Default | Effect |
|------|---------|--------|
| `--use_bfloat16` | `true` | Half-precision — halves memory, minimal speed cost on modern GPUs |
| `--gradient_checkpointing` | `true` | Recomputes activations in backward pass — ~30% slower but large memory savings |

## Note: Prebuilt OpenEnv Maze

OpenEnv includes a [prebuilt maze environment](https://meta-pytorch.org/OpenEnv/environments/maze.html) with similar mechanics (8x8 grid, directional movement, exit-finding). This tutorial builds a custom maze from scratch to demonstrate the full workflow — writing the environment, serving it, and deploying it on Union.

If you just want to jump to training, you could also host the prebuilt maze on Union by wrapping it in a `FastAPIAppEnvironment` and pointing the pipeline at it. Any OpenEnv-compatible server works as the `--env_url` target.

## Rewards

| Event | Reward |
|-------|--------|
| Reach exit | +10.0 |
| Move closer (manhattan) | +0.1 |
| Move away | -0.1 |
| Hit wall | -0.3 |
| Revisit cell | -0.2 |
| Max steps (100) | -1.0 |