# OpenEnv Blackjack RL: Train an LLM with GRPO

Train a small language model to play Blackjack using Group Relative Policy Optimization (GRPO) with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) and Flyte. Watch win rates improve over training with baseline comparisons.

## What's Here

| Script | What it does |
|--------|-------------|
| `blackjack_rl.py` | Full RL pipeline — baselines, GRPO training loop, eval at each checkpoint, HTML report with win-rate curves |

## Concepts Covered

- **OpenEnv** — Meta's universal RL environment API (`reset()` / `step()` / `state()`)
- **GRPO** — Group Relative Policy Optimization for LLM fine-tuning
- **Reward shaping** — Transforming raw game outcomes into training signals
- **Eval tracking** — Measuring win-rate improvement at each training checkpoint
- **Flyte reports** — HTML reports with matplotlib charts

## Setup

```bash
cd tutorials/openenv-blackjack-rl

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

### Environment Server

The pipeline connects to an OpenSpiel Blackjack server via OpenEnv. By default it uses the public HF Space:

```
https://openenv-openspiel-env.hf.space
```

To run locally with Docker instead:

```bash
docker run -d -p 8000:8000 -e OPENSPIEL_GAME=blackjack registry.hf.space/openenv-openspiel-env:latest
```

Then pass `--env_url http://localhost:8000` to the pipeline.

---

## Local Development

```bash
# Quick test (small model, few steps)
flyte run --local --tui blackjack_rl.py pipeline --training_steps 2 --rollouts_per_step 8 --eval_episodes 20

# Full local run
flyte run --local --tui blackjack_rl.py pipeline --training_steps 5

# Auto-open the HTML report
flyte run --local --tui blackjack_rl.py pipeline --training_steps 3 --open_report
```

### Browse Past Runs

```bash
flyte start tui
```

---

## Deploy to Production

```bash
# Train on the cluster with GPU
flyte run blackjack_rl.py pipeline --training_steps 10 --rollouts_per_step 32

# Use a larger model
flyte run blackjack_rl.py pipeline --model_id "Qwen/Qwen2.5-1.5B-Instruct" --training_steps 20
```

---

## How It Works

```
pipeline
  |
  +-- eval_baselines()        # Random + heuristic win rates (cached)
  +-- eval_model()            # Pre-training LLM win rate
  +-- for each training step:
  |     +-- train_step()      # Collect rollouts -> GRPO update -> save checkpoint
  |     +-- eval_model()      # Evaluate checkpoint, record win rate
  +-- Generate HTML report    # Win-rate curve + baselines + action distribution
```

**GRPO in a nutshell:**
1. Play a group of blackjack episodes with the LLM
2. Compute group-relative advantages (normalize rewards within the group)
3. Update policy: increase probability of actions from high-reward episodes

---

## Key Concepts

| Feature | Local | Remote |
|---------|-------|--------|
| **Run** | `flyte run --local` | `flyte run` |
| **TUI** | `--tui` flag | Dashboard in UI |
| **Caching** | `cache="auto"` — local SQLite | `cache="auto"` — cluster cache |
| **Reports** | `report=True` — local HTML file | `report=True` — in Flyte UI |
| **Compute** | Your CPU/MPS | `Resources(cpu=2, memory="8Gi", gpu=1)` |
| **Model** | Qwen2.5-0.5B (default) | Scale to 1.5B+ with GPU |

## Resources

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv) — Universal RL environment spec
- [GRPO Paper (arXiv:2402.03300)](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [OpenEnv GRPO Blackjack Tutorial](https://github.com/meta-pytorch/OpenEnv/tree/main/examples/grpo_blackjack) — Official training example