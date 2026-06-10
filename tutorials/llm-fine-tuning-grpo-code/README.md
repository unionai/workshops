# GRPO Fine-Tuning: Teach a Model to Write Code

Teach a small language model to write correct Python functions using **GRPO (Group Relative Policy Optimization)**. The reward is dead simple: does the generated code actually pass the test cases?

This is the same RL technique DeepSeek used for R1 — applied here to code generation instead of math. Unlike supervised fine-tuning which memorizes one "correct" answer, GRPO generates multiple completions per prompt and reinforces whichever ones pass the tests. The model discovers its own solutions.

## How GRPO Works

For each training prompt, the model generates several completions. A reward function scores each one, and GRPO reinforces the best within the group:

```
Prompt: "Write a function called `reverse_string` that reverses a string.

def reverse_string(s):"

    ├── Completion 1: "return s[::-1]"              → 3/3 tests pass → reward: 1.0
    ├── Completion 2: "return ''.join(reversed(s))"  → 3/3 tests pass → reward: 1.0
    ├── Completion 3: "return s[0]"                  → 0/3 tests pass → reward: 0.0
    └── Completion 4: "return list(s)"               → 0/3 tests pass → reward: 0.0

GRPO computes advantages within the group: completions 1 & 2 get positive
advantage, 3 & 4 get negative. The policy shifts toward code that passes tests.
```

The key insight: there's no single "correct" answer to learn. Any code that passes is good code. This is why RL works better than SFT here — it rewards the *outcome*, not a specific implementation.

## What's in the Pipeline

The workflow runs three steps:

```
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│ Prepare Data │────▶│   GRPO Training   │────▶│  Evaluate  │
│  (CPU task)  │     │   (GPU task)      │     │ (GPU task) │
└──────────────┘     └──────────────────┘     └────────────┘
 MBPP problems        GRPO with sandboxed      Base vs fine-tuned
 from HuggingFace     code execution           sandboxed comparison
                      (LoRA or full)
```

1. **Prepare data** — Downloads the MBPP dataset from HuggingFace and builds prompts with function signatures extracted from reference solutions.
2. **Train with GRPO** — Fine-tunes a model (LoRA by default, or full with `--method full`). For each prompt, generates multiple completions, executes them in a [sandbox](#sandboxed-code-execution), and rewards based on test pass rate.
3. **Evaluate** — Runs the same problems through both the base and fine-tuned model, comparing pass rates side by side.

## Files

| File | What it does |
|------|-------------|
| `workflow.py` | Full pipeline — data prep, GRPO training, evaluation |
| `config.py` | Flyte task environments (CPU/GPU), image config, secrets |
| `report_helpers.py` | SVG chart generation and HTML styling for live reports |
| `requirements.txt` | Python dependencies |

## Setup

```bash
cd tutorials/llm-fine-tuning-grpo-code
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your HuggingFace token (needed for gated models):

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

## Run

### Quick sanity check

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --max_train_samples 10 \
  --epochs 1 \
  --num_generations 2 \
  --batch_size 2 \
  --num_eval_examples 5
```

Small dataset, one epoch, minimal generations — finishes in a few minutes. Good for verifying the pipeline works end to end.

### Standard run

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --method lora \
  --epochs 3 \
  --lr 5e-5 \
  --batch_size 4 \
  --num_generations 4 \
  --max_completion_length 128 \
  --max_train_samples 200 \
  --max_eval_samples 50 \
  --num_eval_examples 30
```

These are the default values shown explicitly. The default SmolLM2-135M is too small to write valid Python for MBPP problems — use Qwen2.5-0.5B or larger for meaningful results.

### Longer training run

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --method lora \
  --epochs 5 \
  --lr 5e-5 \
  --batch_size 4 \
  --num_generations 4 \
  --max_completion_length 192 \
  --max_train_samples 400 \
  --max_eval_samples 50 \
  --num_eval_examples 50
```

### Full fine-tuning (no LoRA)

```bash
flyte run workflow.py pipeline --method full --model_name "Qwen/Qwen2.5-0.5B"
```

By default the pipeline uses LoRA adapters (`--method lora`), which freeze most weights and train small low-rank matrices. Use `--method full` to update all parameters — more expressive but uses more memory and is slower.

### Bigger model

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-1.5B"
```

### Local execution (no cluster)

Add `--local` to run everything on your machine. Useful for debugging but slow without a GPU:

```bash
flyte run --local --tui workflow.py pipeline --max_train_samples 10 --epochs 1
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `HuggingFaceTB/SmolLM2-135M` | HuggingFace model to fine-tune |
| `--method` | `lora` | `lora` for LoRA adapters, `full` for full fine-tuning |
| `--epochs` | `3` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `4` | Prompts per training batch |
| `--num_generations` | `4` | Completions generated per prompt (the "Group" in GRPO) |
| `--max_completion_length` | `128` | Max tokens per generated completion |
| `--max_train_samples` | `200` | Number of training problems |
| `--max_eval_samples` | `50` | Number of held-out eval problems |
| `--num_eval_examples` | `30` | Problems used in the before/after comparison |
| `--lora_r` | `16` | LoRA rank — higher = more capacity, more params |
| `--lora_alpha` | `32` | LoRA scaling factor (effective scale = alpha/r) |

## Dataset

The pipeline uses [**MBPP** (Mostly Basic Python Programming)](https://huggingface.co/datasets/google-research-datasets/mbpp), a standard benchmark of ~970 Python programming problems from Google Research. Each problem has a natural language description, a reference solution, and 3 executable test assertions.

The `prepare_data` task downloads MBPP from HuggingFace, extracts the function signature from each reference solution, and builds a prompt that the model completes:

```
Problem text: "Write a function to find the maximum of two numbers."
Reference:    def max_of_two(a, b): return a if a > b else b

  → Prompt sent to model:
    "Write a function to find the maximum of two numbers.

    def max_of_two(a, b):"

  → Tests:
    assert max_of_two(3, 5) == 5
    assert max_of_two(10, 2) == 10
    assert max_of_two(4, 4) == 4
```

Problems range from simple (`is_even`, `reverse_string`) to moderate (`fibonacci`, `remove_duplicates`) to harder (`heap operations`, `regex matching`). Use `--max_train_samples` and `--max_eval_samples` to control how many are used.

## The Reward Function

The reward is the fraction of test cases the generated code passes:

| Result | Reward |
|--------|--------|
| All 3 tests pass | 1.0 |
| 2 of 3 pass | 0.67 |
| 1 of 3 pass | 0.33 |
| No tests pass / invalid code | 0.0 |

This continuous signal is important — GRPO gets partial credit for partially correct code, which helps it learn incrementally rather than requiring perfect solutions from the start.

## Sandboxed Code Execution

The model generates arbitrary Python which needs to be executed to compute rewards. Rather than using `exec()` in the training process (dangerous with untrusted code), this tutorial uses [Union interactive sandboxes](https://docs.union.ai/docs/v2/union/user-guide/sandboxing/interactive-sandboxes/) (`union.sandbox.on_device`) to run generated code safely:

- **Network blocked** — generated code cannot make network requests
- **Process isolation** — crashes in generated code don't affect the trainer
- **Persistent session** — one sandbox stays open for the entire training run, avoiding per-evaluation setup cost
- **Timeout enforcement** — each code execution has a 5-second timeout, preventing infinite loops

The sandbox session is opened once per task, and the reward function calls into it from the trainer thread using `asyncio.run_coroutine_threadsafe()`. For production workloads with higher isolation requirements, use `sb.session()` (remote sandbox) which runs each session in its own Kubernetes pod.

## Live Training Reports

During training, the Flyte report updates live with:

- **Progress bar** — current step, epoch, and loss
- **Training loss chart** — loss curve over epochs
- **Reward chart** — average reward over training batches (running avg + per-batch)
- **Pass rate chart** — percentage of completions passing all tests
- **Stat grid** — method, dataset size, epochs, learning rate, generations

The evaluation report shows a side-by-side comparison with a bar chart of base vs GRPO pass rates, plus individual code examples with test results.

## Why Code Generation for GRPO?

Code is an ideal task for reinforcement learning because:

- **Verifiable** — run it and check, no subjective judgment needed
- **Multiple valid solutions** — `s[::-1]` and `''.join(reversed(s))` both work
- **Continuous reward signal** — partial credit from individual test cases guides learning
- **Practical** — this is how production code models (Codex, DeepSeek-Coder) are actually trained

## Understanding the Training Loop

Unlike standard SFT where the trainer just does forward/backward passes, GRPO's inner loop is:

```
For each batch of prompts:
  1. Generate `num_generations` completions per prompt (inference)
  2. Score each completion with the reward function (sandbox execution)
  3. Compute advantages within each group (normalize rewards)
  4. Update policy to increase probability of high-reward completions (training)
```

This means each training step involves both **inference** (generating code) and **execution** (running tests), which is why GRPO is slower than SFT. The `num_generations` parameter directly controls the inference cost — `num_generations=4` means 4x the generation work per step.
