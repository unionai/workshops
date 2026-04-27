# LLM Fine-Tuning: GRPO (Code Generation)

Teach a model to write correct Python functions using GRPO (Group Relative Policy Optimization). The reward is simple: does the generated code pass the test cases?

This is the same technique DeepSeek used for R1 — but applied to code instead of math. Multiple valid implementations exist for each problem, so GRPO can explore different solutions rather than memorizing a single "correct" answer.

## How It Works

```
Prompt: "Write a Python function called `reverse_string` that reverses a string.

def reverse_string(s):"

    ├── Completion 1: "return s[::-1]"           → tests pass  → reward: 1.0
    ├── Completion 2: "return ''.join(reversed(s))" → tests pass → reward: 1.0
    ├── Completion 3: "return s[0]"               → tests fail  → reward: 0.0
    └── Completion 4: "return list(s)"            → tests fail  → reward: 0.0

GRPO reinforces completions 1 & 2 — both are valid, different solutions.
```

Unlike SFT which teaches one specific implementation, GRPO rewards *any* code that passes the tests.

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for training |
| `workflow.py` | Pipeline: coding problems → GRPO train → evaluate pass rate |

## Setup

```bash
cd tutorials/llm-fine-tuning-grpo-code

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
flyte run --local --tui workflow.py pipeline --max_train_samples 30 --epochs 1
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline --epochs 3
```

### Bigger model

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-0.5B"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `4` | Prompts per batch (must be divisible by `num_generations`) |
| `--num_generations` | `4` | Completions per prompt |
| `--max_completion_length` | `128` | Max tokens per completion |
| `--max_train_samples` | `200` | Max training problems |
| `--max_eval_samples` | `50` | Max evaluation problems |
| `--num_eval_examples` | `30` | Problems for before/after comparison |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |

## The Problems

30 Python function problems ranging from simple (`double`, `add`) to moderate (`fibonacci`, `remove_duplicates`, `flatten`). Each has 3 test cases. The reward function actually executes the generated code and checks the assertions.

## Reward Function

The reward = fraction of test cases passed:

| Result | Reward |
|--------|--------|
| All 3 tests pass | 1.0 |
| 2 of 3 pass | 0.67 |
| 1 of 3 pass | 0.33 |
| None pass / invalid code | 0.0 |

This gives GRPO a continuous signal — partial credit for partially correct code.

## Training Report

The training report updates live with:
- **Reward chart** — average reward over training batches
- **Pass rate chart** — percentage of completions that pass all tests
- Current stats table

## Why Code Generation for GRPO?

Code is an ideal GRPO task because:
- **Verifiable** — you can run the code and check if it works
- **Multiple valid solutions** — `return s[::-1]` and `return ''.join(reversed(s))` are both correct
- **Continuous reward** — partial credit from individual test cases
- **Practical** — this is how production code models are trained


flyte run workflow.py pipeline --max_train_samples 60 --epochs 1 --num_eval_examples 15 