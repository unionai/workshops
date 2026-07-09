# GRPO Fine-Tuning: Teach a Model to Solve Math

Teach a small language model to solve grade-school math word problems (**GSM8K**) using **GRPO (Group Relative Policy Optimization)**. The reward is dead simple: did the model reach the correct final answer?

This is the sibling of [`llm-fine-tuning-grpo-code`](../llm-fine-tuning-grpo-code). Same RL technique (the one DeepSeek used for R1), same Flyte pipeline shape — but the task is math instead of code, so:

- **No sandbox.** The reward just parses the model's final number and compares it to the gold answer. Nothing untrusted is executed, so the whole thing is simpler and a bit faster.
- **The dataset is already in the learnable zone.** Qwen2.5-0.5B-**Instruct** already solves ~30–40% of GSM8K, so GRPO has real signal to sharpen from step one — no difficulty-filtering pre-pass required.

> **Why not a harder math set (e.g. DeepMath-103K)?** GRPO can only sharpen capability the base model *already has occasionally*. A 0.5B model scores ~0% on competition-level math — every completion fails, the reward is flat zero, and nothing is learned (the same "all-impossible" wall the code tutorial hits on raw MBPP). GSM8K is the sweet spot for a small model. You can point `--dataset_name` at a harder set to *demonstrate* that failure mode as a teaching contrast.

## How GRPO Works

For each problem, the model generates several full solutions. A reward function scores each, and GRPO reinforces the best within the group:

```
Problem: "Natalia sold clips to 48 friends in April, and half as many in May.
          How many clips did she sell altogether?"

    ├── Completion 1: "...48 + 24 = 72. \boxed{72}"      → correct → reward: 1.0
    ├── Completion 2: "...48 + 24 = 72. \boxed{72}"      → correct → reward: 1.0
    ├── Completion 3: "...48 - 24 = 24. \boxed{24}"      → wrong   → reward: 0.0
    └── Completion 4: "...48 * 2 = 96.  \boxed{96}"      → wrong   → reward: 0.0

GRPO computes advantages within the group: completions 1 & 2 get positive
advantage, 3 & 4 negative. The policy shifts toward reasoning that lands the
right answer — without ever being shown a "correct" derivation to copy.
```

The key insight, same as the code tutorial: there's no single correct *derivation* to learn. Any reasoning chain that reaches the right number is rewarded. RL optimizes the **outcome**, not one specific path.

## What's in the Pipeline

```
┌──────────────┐     ┌──────────────────┐     ┌────────────┐
│ Prepare Data │────▶│   GRPO Training   │────▶│  Evaluate  │
│  (CPU task)  │     │   (GPU task)      │     │ (GPU task) │
└──────────────┘     └──────────────────┘     └────────────┘
 GSM8K question       GRPO with answer-        Base vs fine-tuned
 + gold answer        correctness reward       accuracy on held-out
                      (LoRA or full)           test problems
```

1. **Prepare data** — Downloads GSM8K and extracts each `(question, gold answer)` pair (the gold number is the value after the `####` marker in the reference solution).
2. **Train with GRPO** — Fine-tunes an instruct model (LoRA by default). For each question it generates multiple solutions, parses the final answer, and rewards the ones that match gold. A light **format reward** nudges the model to put its answer in `\boxed{...}` so it's parseable.
3. **Evaluate** — Runs held-out GSM8K problems through both the base and fine-tuned model, comparing accuracy side by side.

## Files

| File | What it does |
|------|-------------|
| `workflow.py` | Full pipeline — data prep, GRPO training, evaluation |
| `config.py` | Flyte task environments (CPU/GPU), image config, secrets |
| `report_helpers.py` | SVG chart generation and HTML styling for live reports |
| `requirements.txt` | Python dependencies |

## Setup

```bash
cd tutorials/llm-fine-tuning-grpo-math
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your HuggingFace token (optional for these ungated models, but keeps things smooth):

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

## Run

### Workshop run (single T4, ~20–30 min)

```bash
flyte run workflow.py pipeline
```

Equivalent to the defaults:

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B-Instruct" \
  --method lora \
  --epochs 1 \
  --lr 1e-5 \
  --batch_size 6 \
  --num_generations 6 \
  --max_completion_length 256 \
  --max_train_samples 200 \
  --max_eval_samples 200 \
  --num_eval_examples 40
```

Use the **Instruct** model, not the base — GSM8K needs the model to follow the "reason then answer" chat format, which the instruct tuning provides.

> **Note:** `--batch_size` must be divisible by `--num_generations` (a GRPO requirement — each optimizer step processes whole generation groups). The defaults use `6` and `6`.

### Keep it from overfitting

GRPO on a small model and a small dataset can **overfit**: the training-time accuracy climbs while held-out accuracy *drops*. Two knobs guard against it, both on by default:

- `--beta 0.04` — KL penalty anchoring the policy to the base model so it can't drift far. Set higher (e.g. `0.1`) if you see the eval regress; `0` disables it.
- `--epochs 1` — more epochs over the same small set is the fastest route to memorization. Add data (`--max_train_samples`) before adding epochs.

### Quick sanity check

```bash
flyte run workflow.py pipeline \
  --max_train_samples 20 --epochs 1 \
  --num_generations 2 --batch_size 2 \
  --max_completion_length 128 --num_eval_examples 8
```

### Full fine-tuning (no LoRA)

```bash
flyte run workflow.py pipeline --method full
```

### Bigger model

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-1.5B-Instruct"
```

### Show the "too hard" failure mode (teaching contrast)

```bash
flyte run workflow.py pipeline --dataset_name "trl-lib/DeepMath-103K"
```

Expect a flat ~0% reward — the base model can't solve competition math, so GRPO has nothing to sharpen. This is the point: **data curation matters more than the algorithm.**

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-0.5B-Instruct` | HuggingFace model to fine-tune (use an **instruct** model) |
| `--method` | `lora` | `lora` for LoRA adapters, `full` for full fine-tuning |
| `--epochs` | `1` | Training epochs (more = higher overfit risk) |
| `--lr` | `1e-5` | Learning rate |
| `--batch_size` | `6` | Completions per training step (must be divisible by `--num_generations`) |
| `--num_generations` | `6` | Completions generated per prompt (the "Group" in GRPO) |
| `--max_completion_length` | `256` | Max tokens per generated solution (math needs room to reason) |
| `--beta` | `0.04` | KL penalty vs. the base model — the main overfit guard |
| `--dataset_name` | `openai/gsm8k` | Dataset to train on |
| `--max_train_samples` | `200` | Number of training problems |
| `--max_eval_samples` | `200` | Number of held-out eval problems |
| `--num_eval_examples` | `40` | Problems used in the before/after comparison |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA scaling factor |

## The Reward Function

Two reward functions, summed with weights `[1.0, 0.2]`:

| Reward | Weight | What it measures |
|--------|--------|------------------|
| **Answer accuracy** | 1.0 | `1.0` if the parsed final answer equals the gold answer, else `0.0` |
| **Format** | 0.2 | `1.0` if the completion contains a parseable `\boxed{...}`, else `0.0` |

The answer is extracted by preferring `\boxed{...}`, then a `#### x` marker, then the last number in the text — then compared numerically to the gold value. Accuracy is the real objective; the small format reward just keeps the output parseable so accuracy can be measured cleanly.

> This is a good example of a **multi-part reward** (dominant objective + light nudge). For how to reason about reward design in general — sparse vs. dense, reward hacking, and why the right reward depends on your use case — see the code tutorial's [How to Think About Rewards](../llm-fine-tuning-grpo-code#how-to-think-about-rewards).

## Why Math for GRPO?

Math is an ideal RL task for the same reasons code is:

- **Verifiable** — check the final answer, no subjective judgment.
- **Multiple valid solution paths** — many chains of reasoning reach the same number.
- **Objective reward** — a correct answer is an unambiguous, unhackable signal.
- **In-reach for small models** — on GSM8K specifically, a 0.5B instruct model has enough of a foothold for RL to lift it.

## Further Reading

- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — introduces GRPO (group-relative advantages, no value network), originally *for math*.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — scales the same RL recipe up into a frontier reasoning model.
- [DAPO](https://arxiv.org/abs/2503.14476) — "Dynamic Sampling" (filtering all-pass/all-fail groups) and other GRPO refinements; the [code tutorial's](../llm-fine-tuning-grpo-code#background--further-reading) learnability filter is a cached, offline version of it.
- [KL / `beta` regularization](https://arxiv.org/abs/2402.03300) — why the KL anchor to the base model keeps a small model from drifting and overfitting (train reward up, held-out down). Standard small-model GSM8K values are `beta ≈ 0.001–0.005` (this tutorial defaults to `0.005`).

The [code tutorial's Background section](../llm-fine-tuning-grpo-code#background--further-reading) has a fuller table mapping each design choice to its paper — most of it applies here too, minus the sandbox and the learnability filter.

## Math vs. Code — Which to Run?

| | Code (`grpo-code`) | Math (`grpo-math`) |
|--|--|--|
| Dataset | MBPP | GSM8K |
| Reward | Tests pass (sandboxed) | Answer correct (parsed) |
| Sandbox | Yes (`union.sandbox`) | No |
| Extra pre-pass | Learnability filter (MBPP is too hard raw) | None (GSM8K is already learnable) |
| Best for | The "cool" demo — real code execution in a sandbox | A simple, reliable backup that trains fast |

The code tutorial is the more impressive demo (watching sandboxed code get graded live). This math tutorial is the dependable fallback: fewer moving parts, no sandbox, and a dataset that's already in the model's learnable zone.
