# LLM Fine-Tuning: PPO (Classic RLHF)

Classic RLHF with Proximal Policy Optimization — the technique originally used to train ChatGPT. Train a reward model on human preferences, then optimize a policy model against it.

## How PPO/RLHF Works

PPO is the most involved post-training method. It requires four models working together:

```
                    ┌──────────────┐
  Prompt ──────────►│ Policy Model │──── Generate ────► Response
                    │  (trainable) │                        │
                    └──────────────┘                        │
                                                           ▼
                    ┌──────────────┐                  ┌──────────┐
                    │  Ref. Model  │◄── KL penalty ──│  Reward  │──► Reward Score
                    │   (frozen)   │                  │  Model   │
                    └──────────────┘                  └──────────┘
                                                           │
                    ┌──────────────┐                        │
                    │  Value Head  │◄── Advantage ──────────┘
                    │  (trainable) │
                    └──────────────┘
```

1. **Policy model** generates responses
2. **Reward model** scores them
3. **Reference model** (frozen copy) constrains the policy from drifting too far
4. **Value head** estimates future rewards for PPO's advantage calculation

## Pipeline

This tutorial runs a 4-step pipeline:

1. **Prepare data** — download Anthropic HH-RLHF preference pairs
2. **Train reward model** — learn to score helpful/harmless responses higher
3. **PPO training** — optimize the policy against the reward model
4. **Evaluate** — compare reward scores and responses before/after

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments |
| `workflow.py` | Pipeline: prepare data → train reward model → PPO → evaluate |

## Setup

```bash
cd tutorials/llm-fine-tuning-ppo

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

### Default

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test

```bash
flyte run --local --tui workflow.py pipeline --max_train_samples 100 --max_eval_samples 20 --epochs 1 --ppo_epochs 1
```

### Remote

```bash
flyte run workflow.py pipeline --epochs 2 --ppo_epochs 2
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model |
| `--dataset_name` | `Anthropic/hh-rlhf` | Preference dataset |
| `--epochs` | `2` | Reward model training epochs |
| `--ppo_epochs` | `2` | PPO optimization epochs per batch |
| `--lr` | `1e-5` | Learning rate |
| `--batch_size` | `4` | Batch size |
| `--max_new_tokens` | `128` | Max tokens per generated response |
| `--max_train_samples` | `2000` | Max training pairs |
| `--max_eval_samples` | `500` | Max evaluation pairs |
| `--num_eval_examples` | `30` | Examples for evaluation |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## The Full Post-Training Series

| Tutorial | Technique | Complexity | What's needed |
|----------|-----------|-----------|---------------|
| [SFT (SQL)](../llm-fine-tuning-sql/) | Supervised fine-tuning | Simplest | Input-output pairs |
| [Classification](../bert-fine-tuning-sentiment/) | Encoder fine-tuning | Simple | Labels |
| [DPO](../llm-fine-tuning-dpo/) | Direct preference optimization | Medium | Chosen/rejected pairs |
| [GRPO](../llm-fine-tuning-grpo/) | Group reward optimization | Medium | Verifiable reward function |
| **PPO** (this) | Classic RLHF | Highest | Reward model + policy + value head + ref model |

## PPO vs DPO vs GRPO

| | PPO | DPO | GRPO |
|---|---|---|---|
| **Reward model** | Required (trained separately) | Not needed | Not needed |
| **Models in memory** | 4 (policy, ref, value, reward) | 2 (policy, ref) | 1 (policy) |
| **Training signal** | Reward model scores | Preference pairs directly | Verifiable reward function |
| **Memory usage** | Highest | Medium | Medium |
| **Stability** | Harder to tune | More stable | More stable |
| **When to use** | Complex preferences, existing reward model | Have preference data, want simplicity | Have verifiable outcomes (math, code) |
