# LLM Fine-Tuning: GRPO (Sentiment Steering)

Steer a language model toward generating more positive text using GRPO (Group Relative Policy Optimization). No human preference data needed — just a reward signal from a sentiment classifier.

GRPO generates multiple completions per prompt, scores them with a reward function, and uses relative performance within the group to update the policy. This is the technique behind DeepSeek-R1's reasoning improvements.

## How GRPO Works

```
Prompt: "The restaurant was located on the corner of..."
    ├── Completion 1: "...a busy street and had amazing food"    → reward: +0.95 (positive)
    ├── Completion 2: "...main street but the service was awful" → reward: -0.87 (negative)
    ├── Completion 3: "...the block, a charming little place"    → reward: +0.72 (positive)
    └── Completion 4: "...fifth avenue and was disappointing"    → reward: -0.63 (negative)

GRPO reinforces completions 1 & 3 based on their relative scores within the group.
```

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: prepare prompts → GRPO train (sentiment reward) → evaluate |

## Setup

```bash
cd tutorials/llm-fine-tuning-grpo

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

### Default (SmolLM2-135M)

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test

```bash
flyte run --local --tui workflow.py pipeline --max_train_samples 50 --epochs 1
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline --epochs 2
```

### Bigger model

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model |
| `--epochs` | `2` | Training epochs |
| `--lr` | `1e-5` | Learning rate |
| `--batch_size` | `4` | Prompts per batch (each generates `num_generations` completions) |
| `--num_generations` | `4` | Completions per prompt (the "group" in GRPO) |
| `--max_completion_length` | `64` | Max tokens per completion |
| `--max_train_samples` | `500` | Max training prompts |
| `--max_eval_samples` | `100` | Max evaluation prompts |
| `--num_eval_examples` | `50` | Prompts for before/after comparison |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## Evaluation

The evaluate step generates completions from both models for the same prompts, then scores each with a sentiment classifier:

- **Average sentiment score** (positive = high, negative = low)
- **% positive completions**
- **Side-by-side examples** showing how the GRPO model steers toward positive text

## Why Sentiment?

GRPO needs a continuous reward signal — not just right/wrong. A sentiment classifier provides smooth, gradient-friendly scores that any model can learn from, even small ones like SmolLM2-135M. Math reasoning (GSM8K) requires larger models (0.5B+) since small models can't produce correct answers to learn from.

## How It Differs from SFT

| | SFT (SQL tutorial) | GRPO (this tutorial) |
|---|---|---|
| **Signal** | Ground truth outputs | Reward function (sentiment score) |
| **Training** | Maximize likelihood of correct answer | Reinforce completions that score higher within group |
| **Data** | Input-output pairs | Just prompts + reward function |
| **Strength** | Teaches exact format | Encourages exploration toward a goal |


flyte run workflow.py pipeline --max_train_samples 100 --epochs 1 --num_eval_examples 20  
