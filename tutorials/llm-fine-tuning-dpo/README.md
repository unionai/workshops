# LLM Fine-Tuning: DPO (Preference Alignment)

Align a language model with human preferences using DPO (Direct Preference Optimization). The model learns to prefer helpful, harmless responses over unhelpful or harmful ones — no reward model needed.

## How DPO Works

DPO takes pairs of responses — one **chosen** (preferred) and one **rejected** — and directly optimizes the model to assign higher probability to chosen responses.

```
Prompt: "How do I make a good cup of coffee?"

Chosen:  "Start with fresh beans, grind them right before brewing.
          Use water just off the boil (195-205°F)..."          ← model should prefer this

Rejected: "Just use instant coffee, it's all the same anyway."  ← model should avoid this
```

Unlike RLHF, DPO skips the reward model entirely — it derives the optimal policy directly from the preference data.

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: prepare HH-RLHF pairs → DPO train → evaluate win rate |

## Setup

```bash
cd tutorials/llm-fine-tuning-dpo

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

### Default (SmolLM2-135M on HH-RLHF)

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test

```bash
flyte run --local --tui workflow.py pipeline --max_train_samples 100 --max_eval_samples 20 --epochs 1
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
| `--dataset_name` | `Anthropic/hh-rlhf` | HuggingFace preference dataset |
| `--epochs` | `2` | Training epochs |
| `--lr` | `5e-6` | Learning rate |
| `--batch_size` | `2` | Per-device batch size |
| `--beta` | `0.1` | DPO beta — higher = stay closer to reference policy |
| `--max_length` | `512` | Max sequence length |
| `--max_train_samples` | `2000` | Max training pairs |
| `--max_eval_samples` | `500` | Max evaluation pairs |
| `--num_eval_examples` | `50` | Examples for win rate evaluation |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## Evaluation

The evaluate step measures **preference win rate** — how often the model assigns higher probability to the chosen response over the rejected one:

- **Base model**: ~50% (no preference, essentially random)
- **DPO-trained**: should be significantly higher

Also generates side-by-side responses from both models for qualitative comparison.

## The Fine-Tuning Series

| Tutorial | Technique | Signal | Dataset |
|----------|-----------|--------|---------|
| [SQL](../llm-fine-tuning-sql/) | SFT (LoRA/QLoRA/Full) | Ground truth outputs | text-to-SQL |
| [Sentiment](../bert-fine-tuning-sentiment/) | Classification | Labels | IMDB reviews |
| [GRPO](../llm-fine-tuning-grpo/) | Reward optimization | Verifiable reward | GSM8K math |
| **DPO** (this) | Preference alignment | Chosen vs rejected pairs | Anthropic HH-RLHF |
