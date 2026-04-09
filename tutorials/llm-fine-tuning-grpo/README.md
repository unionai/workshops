# LLM Fine-Tuning: GRPO (Letter Counting)

Teach a language model to count letters in words using GRPO (Group Relative Policy Optimization). No labeled data needed — just a reward function that checks if the answer is right.

LLMs famously struggle with counting letters. GRPO generates multiple completions per prompt, scores them with a reward function, and uses the relative performance within the group to update the policy. This is the technique behind DeepSeek-R1's reasoning improvements.

## Results

With SmolLM2-135M on a T4 GPU (~20 min training):

| | Exact Match | Within ±1 |
|---|---|---|
| **Base model** | 3.3% | 10.0% |
| **GRPO-trained** | 50.0% | 86.7% |

The base model spams "12" for everything. The GRPO model actually learned to count.

## How GRPO Works

```
Prompt: "How many times does 'a' appear in 'banana'? Answer with just the number."
    ├── Completion 1: "The answer is 3."  → reward: 1.0  (exact match)
    ├── Completion 2: "The answer is 2."  → reward: 0.5  (off by 1)
    ├── Completion 3: "The answer is 5."  → reward: 0.0  (too far off)
    └── Completion 4: "The answer is 3."  → reward: 1.0  (exact match)

GRPO reinforces completions 1 & 4 based on their relative scores within the group.
```

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: generate letter-counting examples → GRPO train → evaluate |

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
flyte run --local --tui workflow.py pipeline --max_train_samples 100 --epochs 1
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline --epochs 3
```

### Bigger model (better accuracy)

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `8` | Prompts per batch (must be divisible by `num_generations`) |
| `--num_generations` | `8` | Completions per prompt (the "group" in GRPO) |
| `--max_completion_length` | `32` | Max tokens per completion |
| `--max_train_samples` | `500` | Max training examples |
| `--max_eval_samples` | `100` | Max evaluation examples |
| `--num_eval_examples` | `50` | Examples for before/after comparison |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## Reward Function

The reward uses partial credit based on how close the answer is:

| Accuracy | Reward |
|----------|--------|
| Exact match | 1.0 |
| Off by 1 | 0.5 |
| Off by 2 | 0.25 |
| Anything else | 0.0 |

This gives GRPO a gradient to learn from even when the model isn't exactly right — unlike binary 0/1 rewards where small models often get stuck at all-zeros.

## Reward Hacking

If you train too long, the model discovers an exploit: since most letters appear exactly once in a word, always answering "1" maximizes expected reward. This is **reward hacking** — the model games the reward function instead of learning the actual task.

| Epochs | Exact Match | What happened |
|--------|-------------|---------------|
| 2 | 50.0% | Good balance — learned to count, some diversity in answers |
| 5 | 44.0% | Worse — collapsed to always answering "1" (94% within ±1 but only because 1 is usually close) |

This is the same problem DeepSeek encountered at scale with R1. Mitigations include:
- **Fewer epochs** — stop before the model collapses (2 epochs is the sweet spot here)
- **Balanced training data** — ensure equal representation of counts 0, 1, 2, 3+
- **KL penalty** — penalize the model for drifting too far from the base policy
- **Bigger models** — more capacity to learn the actual rule instead of a shortcut

## Scaling Up

The default params are tuned for a T4 GPU workshop demo (~20 min). For better results with more compute:

```bash
# Bigger model on a larger GPU
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --max_train_samples 1000 --epochs 2 \
  --num_generations 8
```

Larger models are less likely to collapse to a single answer and can learn the actual counting rule.

## How It Differs from SFT

| | SFT (SQL tutorial) | GRPO (this tutorial) |
|---|---|---|
| **Signal** | Ground truth outputs | Reward function (correctness score) |
| **Training** | Maximize likelihood of correct answer | Reinforce completions that score higher within group |
| **Data** | Input-output pairs | Just prompts + verifiable answers |
| **Strength** | Teaches exact format | Encourages exploration toward correct answers |



flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B" --max_train_samples 500 --epochs 2 --num_eval_examples 50 
