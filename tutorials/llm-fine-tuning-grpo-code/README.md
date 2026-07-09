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

The workflow runs four steps:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────┐
│ Prepare Data │──▶│    Filter    │──▶│   GRPO Training   │──▶│  Evaluate  │
│  (CPU task)  │   │  (GPU task)  │   │   (GPU task)      │   │ (GPU task) │
└──────────────┘   └──────────────┘   └──────────────────┘   └────────────┘
 MBPP candidate     Keep only the      GRPO with sandboxed    Base vs fine-tuned
 pool from HF       learnable          code execution         sandboxed comparison
                    problems (cached)  (LoRA or full)
```

1. **Prepare data** — Downloads the MBPP dataset from HuggingFace and builds a *candidate pool* of prompts with function signatures extracted from reference solutions.
2. **Filter for learnability** — Samples the base model over the candidate pool and keeps only the problems it solves *sometimes but not always* — the zone where GRPO's within-group advantage is non-zero. [Why this matters ↓](#the-learnability-filter). This task is cached, so it runs once and later runs reuse the filtered set.
3. **Train with GRPO** — Fine-tunes a model (LoRA by default, or full with `--method full`). For each prompt, generates multiple completions, executes them in a [sandbox](#sandboxed-code-execution), and rewards code that passes *all* tests.
4. **Evaluate** — Runs held-out problems through both the base and fine-tuned model, comparing pass rates side by side.

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

### Workshop run (single T4, ~20–30 min)

The defaults are tuned for a single **NVIDIA T4** — the whole pipeline (data → GRPO
train → eval) finishes in roughly 20–30 minutes and the live reward/pass-rate
charts climb visibly during training. Just run:

```bash
flyte run workflow.py pipeline
```

That's equivalent to the defaults shown explicitly:

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --method lora \
  --epochs 1 \
  --lr 5e-5 \
  --batch_size 6 \
  --num_generations 6 \
  --max_completion_length 128 \
  --max_candidate_samples 300 \
  --max_train_samples 100 \
  --filter_samples 4 \
  --max_eval_samples 50 \
  --num_eval_examples 20
```

Qwen2.5-0.5B is the sweet spot: small enough to train fast on a T4, strong enough
to write valid Python (SmolLM2-135M can't). The T4 (Turing) has no bf16 support, so
the workflow automatically trains in **fp16** rather than falling back to slow fp32.

> **First run pays for the filter.** The learnability filter (step 2) samples the
> base model over the candidate pool, which takes a few minutes. It's cached on its
> inputs, so the *first* run does the work and every later run with the same model
> and pool reuses the filtered dataset instantly — spend a run before the workshop
> to warm the cache, then the live run goes straight to training.

> **Note:** `--batch_size` must be divisible by `--num_generations` (a GRPO
> requirement — each optimizer step processes whole generation groups). The
> defaults use `6` and `6`.

### Quick sanity check

```bash
flyte run workflow.py pipeline \
  --max_candidate_samples 20 \
  --max_train_samples 6 \
  --filter_samples 3 \
  --epochs 1 \
  --num_generations 2 \
  --batch_size 2 \
  --max_completion_length 64 \
  --num_eval_examples 4
```

Tiny dataset, minimal generations — finishes in a few minutes once the image is
built. Good for verifying the pipeline works end to end before a longer run.

### Longer training run (bigger GPU)

On an L40s / A100 you can push harder for stronger results:

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-0.5B" \
  --method lora \
  --epochs 5 \
  --lr 5e-5 \
  --batch_size 8 \
  --num_generations 8 \
  --max_completion_length 192 \
  --max_candidate_samples 800 \
  --max_train_samples 400 \
  --filter_samples 6 \
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
flyte run --local --tui workflow.py pipeline \
  --max_candidate_samples 15 --max_train_samples 8 --filter_samples 2 --epochs 1
```

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-0.5B` | HuggingFace model to fine-tune |
| `--method` | `lora` | `lora` for LoRA adapters, `full` for full fine-tuning |
| `--epochs` | `1` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `6` | Completions per training step (must be divisible by `--num_generations`) |
| `--num_generations` | `6` | Completions generated per prompt (the "Group" in GRPO) |
| `--max_completion_length` | `128` | Max tokens per generated completion |
| `--max_candidate_samples` | `300` | Size of the candidate pool the filter draws from |
| `--max_train_samples` | `100` | Target cap on *learnable* problems to keep (for Qwen-0.5B only ~25% of the pool qualifies, so the pool size is usually the real driver) |
| `--filter_samples` | `4` | Base-model samples per candidate; keep if `1 ≤ all-pass < filter_samples` |
| `--max_eval_samples` | `50` | Number of held-out eval problems |
| `--num_eval_examples` | `20` | Problems used in the before/after comparison |
| `--beta` | `0.04` | KL penalty vs. the base model — the main overfit/drift guard (see below) |
| `--lora_r` | `16` | LoRA rank — higher = more capacity, more params |
| `--lora_alpha` | `32` | LoRA scaling factor (effective scale = alpha/r) |

## Tuning Guide — What the Knobs Actually Do

The table above lists the flags; this is how to *reason* about the ones that matter for GRPO specifically (the LoRA/lr knobs behave like they do in any fine-tune). GRPO's failure modes are subtle — the training reward can climb while the model gets *worse* on held-out problems — so it helps to know which knob fixes which symptom.

### `--beta` — the KL leash

GRPO adds a penalty for drifting away from the **base model's** behavior, scaled by `beta`. Think of it as a leash:

- **`beta` too low (→ 0):** long leash. The policy is free to wander wherever the reward points — including into degenerate or memorized behavior. Symptom: **train reward climbs, held-out accuracy drops**, output style drifts (e.g. chatty code that gets truncated), entropy collapses.
- **`beta` too high:** short leash. The policy can barely move from the base, so it can't learn. Symptom: **flat 0pp** — the fine-tuned model is nearly identical to the base.
- **The sweet spot is narrow.** For a small model on a small dataset, `0.005–0.02` is a reasonable range; the tutorial defaults to `0.04` conservatively. If you see a regression, this is the *first* knob to reach for — before touching the model or the data.

### `--num_generations` — the group size

This is the "Group" in **G**RPO. Advantages are computed *within* the group of completions for one prompt, so the group needs **variety of outcomes** to produce a gradient. If all N completions get the same reward, that prompt teaches nothing that step.

- **Higher** (8–16): better advantage estimates, more exploration, more stable — but every step costs proportionally more generation time (the GRPO bottleneck).
- **Lower** (2–4): faster, noisier, more prone to zero-variance groups.
- It must divide `--batch_size`.

### `--filter_samples` — how hard the difficulty filter looks

How many times the base model attempts each candidate during the [learnability filter](#the-learnability-filter). A problem is kept only if it's solved `1 ≤ x < filter_samples` times.

- **Higher** (6–8): a sharper "is this in the learnable zone?" estimate, and a wider learnable band — at linear cost to the (cached) filter pass.
- **Lower** (3): cheaper, coarser. Below 3 the "sometimes" signal is too noisy to be meaningful.

### `--epochs` — how many passes over the (small) training set

More epochs over the *same* problems is the fastest route to **memorization**: the model over-fits the training set and generalizes worse. On a small filtered set, prefer **1–2 epochs**. If you want more training signal, add *data* (`--max_candidate_samples`) before adding epochs.

### Symptom → knob cheat-sheet

| You see... | Most likely cause | Reach for |
|---|---|---|
| GRPO **worse** than base, output drifted/verbose | Policy drift, too little regularization | Raise `--beta`, lower `--epochs` |
| GRPO **identical** to base (0pp) | Policy can't move | Lower `--beta`, raise `--epochs`/`--lr` |
| Reward stuck near 0 the whole run | No learnable signal (all problems too hard) | Bigger `--max_candidate_samples`, check the filter kept anything |
| Reward jumps to a degenerate constant | Reward is hackable | Make the reward stricter (see below) |
| Training very slow | Generation cost | Lower `--num_generations` / `--max_completion_length` |

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

Problems range from simple (`is_even`, `reverse_string`) to moderate (`fibonacci`, `remove_duplicates`) to harder (`heap operations`, `regex matching`). `--max_candidate_samples` sets the size of the pool the [learnability filter](#the-learnability-filter) draws from, `--max_train_samples` the number of learnable problems kept for training, and `--max_eval_samples` the held-out eval set.

## The Learnability Filter

GRPO computes advantages **within a group** of completions for the same prompt, so
it only learns where that group has *reward variance*. Two kinds of problems teach
it nothing:

- **Impossible** — the base model never solves it, so every completion scores 0. Worse, these are exactly where reward hacking takes over: a constant like `return True` or `return -1` grabs a few asserts and, in an otherwise all-zero group, becomes the *highest-advantage* completion. GRPO then reinforces degenerate constants, and the fine-tuned model gets **worse**.
- **Trivial** — the base model always solves it, so every completion scores 1 and the advantage is zero.

A raw MBPP pool for a 0.5B model is mostly impossible problems, so unfiltered
training collapses onto constants. The `filter_learnable` task fixes this at the
data level: it samples the base model `filter_samples` times per candidate and keeps
only the problems solved *sometimes but not always* (`1 ≤ all-pass count < filter_samples`)
— the learnable middle where every group has a real gradient. The task is cached on
its inputs, so the filtering cost is paid once.

This is the RLVR lesson worth demonstrating: **RL can only sharpen what the model can already do occasionally.** Curating for that zone is half the job — often more impactful than any hyperparameter.

## The Reward Function

The reward is **binary** — all or nothing:

| Result | Reward |
|--------|--------|
| All tests pass | 1.0 |
| Any test fails / invalid code | 0.0 |

Binary reward is deliberate. Partial credit (`passed/total`) is trivially hackable:
a constant that returns the right *type* passes a fraction of the asserts, and on an
impossible problem that fraction beats the genuine (failing) attempts — so GRPO
learns to emit `return True`. All-or-nothing removes that gradient entirely. It only
works because the [learnability filter](#the-learnability-filter) guarantees each
retained problem still has variance in its group (some completions fully pass, some
don't), so there's always a real signal to learn from.

## How to Think About Rewards

The reward function *is* the task definition. GRPO doesn't know what "good code" means — it only knows the number your reward returns, and it will find the **shortest path to a high number**, whether or not that path is what you intended. Most of the work in an RL project is reward design, not RL. A few principles worth internalizing:

**1. The reward is a proxy — the model optimizes the proxy, literally.** This is the central hazard, usually called *reward hacking*. If a partially-correct constant scores 0.33 and the honest attempts score 0.0, the model learns to emit the constant. The fix isn't a better optimizer; it's a reward with no cheap exploit. Ask of any reward: *"what's the laziest output that scores well here?"* — and if that output isn't what you want, the reward is wrong.

**2. Sparse vs. dense is a real trade-off.** A **binary** reward (all-or-nothing) is unhackable but *sparse* — on hard problems every attempt scores 0, so there's no gradient. A **dense** reward (partial credit) gives signal on near-misses but opens a hacking surface. This tutorial resolves the tension by moving the density into the **data** instead of the reward: the [learnability filter](#the-learnability-filter) guarantees each retained problem produces a mix of 0s and 1s, so a *binary* reward still yields a gradient. (If you genuinely need a dense reward, weight the parts so the cheap wins can't dominate — see the VeRPO reference below.)

**3. Multi-part rewards encode multiple goals — weight them carefully.** The sibling [math tutorial](../llm-fine-tuning-grpo-math) uses `1.0 × correctness + 0.2 × format`: correctness is the goal, format is a gentle nudge to keep output parseable. Keep the "real" objective dominant, or the model will farm the cheap secondary reward (e.g. perfect formatting around wrong answers).

**4. Prefer verifiable rewards when you can get them.** Code (run the tests) and math (check the answer) give an **objective, ungameable oracle** — no learned reward model to drift or be gamed. That's why they're the poster children for RL fine-tuning. The catch, as this tutorial shows, is that "verifiable" only covers *what counts as correct* — you still design the *shaping*, the *data*, and the *regularization*.

**Why the reward changes with the use case.** There's no universal reward — it's a function of what you can measure and what you're willing to accept:

| Use case | Natural reward signal | Watch out for |
|---|---|---|
| Code generation | Unit tests pass | Constants that pass a subset; tests that are gameable |
| Math / reasoning | Final answer matches | "Right answer, nonsense reasoning"; format parsing |
| SQL / tool use | Query runs & returns expected rows / call succeeds | Syntactically-valid-but-wrong; empty results scoring "no error" |
| Summarization / writing | **No cheap verifier** → LLM-as-judge or rubric | Judge bias, verbosity/sycophancy hacking |
| Style / safety constraints | Regex / classifier / format checks | Model satisfies the checker while violating the intent |

The rule of thumb: **if a cheap program can verify the outcome, use it and keep the reward strict; if it can't, you're now also in the business of building (and defending) a reward model** — a much harder problem, and where most reward hacking horror stories come from. Start strict and binary, add density or sub-rewards only when the model can't learn — and every time you add a term, re-ask *"what's the laziest way to score well now?"*

## Sandboxed Code Execution

The model generates arbitrary Python which needs to be executed to compute rewards. Rather than using `exec()` in the training process (dangerous with untrusted code), this tutorial uses [Union interactive sandboxes](https://docs.union.ai/docs/v2/union/user-guide/sandboxing/interactive-sandboxes/) (`union.sandbox.on_device`) to run generated code safely:

- **Network blocked** — generated code cannot make network requests
- **Process isolation** — crashes in generated code don't affect the trainer
- **Persistent session** — one sandbox stays open for the entire training run, avoiding per-evaluation setup cost
- **Timeout enforcement** — each code execution has a 5-second timeout, preventing infinite loops

The sandbox session is opened once per task, and the reward function calls into it from the trainer thread using `asyncio.run_coroutine_threadsafe()`.

## What to Expect on a 0.5B — and Why Labs Get Real Coding Results

Be warned, and read this before you judge the numbers: on a **small model with ~100 problems, this example plateaus.** You'll see the *training* pass-rate climb nicely (GRPO genuinely learns the training problems), but **held-out** pass rate barely moves — and much of the apparent gain is the model learning to emit clean, runnable code rather than becoming a better *coder*. That's not a bug, and it's not a knock on GRPO for code. It's the **regime**.

"But companies use GRPO for coding all the time!" — they do, and code is one of the *best* RLVR targets because it's cheaply verifiable. The difference isn't the domain, it's the scale:

| | This tutorial | What labs actually do |
|---|---|---|
| **Base model** | 0.5B–3B | 7B–70B+, already a competent coder |
| **Learnable zone** | tiny — a 0.5B solves almost nothing *sometimes* → dead groups | huge — a strong base solves lots *sometimes* → dense gradient everywhere |
| **Dataset size** | ~100 MBPP problems | 10⁴–10⁶ problems (contests, repos, synthetic) |
| **Transfer** | poor at N=100 (each problem an island) | emerges from *breadth* — at 100k problems, "write correct Python" becomes one broad, transferable skill |
| **Reward harness** | binary tests + learnability filter | test suites + difficulty curricula + dedup + anti-hacking |

Two things fail *simultaneously* in the 0.5B-on-100-problems toy setting: the **learnable zone** is nearly empty (the base rarely succeeds, so most groups give zero gradient), and **transfer** doesn't happen (100 diverse problems are 100 islands, not one skill). Both are fixed by scale — a stronger base fills in the learnable zone, and enough problems (10⁴+) turn "write correct Python" into a single broad skill with endless instances. Going 0.5B → 3B alone helps the first knob but not the second; you need **both** a bigger base and far more data.

So this tutorial is the right place to learn the *mechanics* (sandboxed verification, the learnability filter, binary reward, the training loop) — but to get genuinely *better code*, revisit it with a 7B+ base and thousands of problems on a bigger GPU. For a task engineered to give a clean win even in the toy regime (one skill, guaranteed-solvable, always-verifiable), and a full framework for scoring your own task, see the [Countdown tutorial's task-selection guide](../llm-fine-tuning-grpo-countdown#choosing-a-task--and-a-model--for-grpo).

## Scaling This to Production

This demo collapses generation, verification, and training into a single GPU task with one in-process sandbox — deliberately, so the whole pipeline fits on one T4 and stays easy to follow. In real RLVR pipelines the wall-clock is dominated by **rollout generation** and **reward verification** (running the code), not the gradient step — so that's where you scale, and you can do it *without changing how code is executed*:

- **Warm worker pools** — put a `flyte.ReusePolicy` on the GPU environment so the policy model loads once and stays hot across rollout batches instead of cold-starting a container per step (`throughput = replicas × concurrency`).
- **Fan out the rollouts** — the group dimension (`num_generations`) and the prompt batch are embarrassingly parallel. Spread them across the warm pool with `flyte.map(...)` / `asyncio.gather()` — each action runs in its own container — and bound it with an `asyncio.Semaphore` or `flyte.map(concurrency=...)` to respect GPU quota.
- **Keep verification on-device** — each fanned-out worker verifies its own shard of completions with its own `sb.on_device.session()`. No shared bottleneck, and the same reward function you see here — just running on more boxes.

What Union does *not* do for you is the RL-engine internals: syncing updated policy weights to inference workers each step, off-policy correction, matching generation throughput to training. That lives in your trainer↔inference integration (e.g. TRL's vLLM mode, veRL). Union gives you the distributed substrate and the secure verifier; you plug in the RL engine.

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
- **Objective reward** — a passing test suite is an unambiguous, unhackable signal (as long as you [filter out the impossible problems](#the-learnability-filter))
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

## Background & Further Reading

None of the design choices here are arbitrary — each maps to a published finding. If you want to go deeper, these are the papers behind the knobs:

| Choice in this tutorial | Why | Reference |
|---|---|---|
| **Learnability filter** — keep only problems the base solves *sometimes* | A group where every completion gets the same reward (all-pass or all-fail) has zero advantage and zero gradient. Filtering to intermediate difficulty (~0.5 success rate) maximizes the informativeness of each update. This is exactly what DAPO calls **Dynamic Sampling**. | [DAPO](https://arxiv.org/abs/2503.14476) · [Competence–Difficulty Alignment](https://arxiv.org/abs/2505.17652) |
| **Binary reward** — 1.0 only if all tests pass | On small models, partial pass-rate (`passed/total`) is *hackable*: constants like `return True` grab the "easy" tests (a skew the VeRPO paper calls **cardinality bias**), and partial credit doesn't beat binary at convergence anyway. | [Beyond Binary / VeRPO](https://arxiv.org/abs/2601.03525) |
| **KL penalty** (`--beta`) | With too little KL regularization the policy drifts from the base and reward-hacks/overfits — train reward climbs while held-out accuracy *drops*. Small-model GRPO configs typically use `beta ≈ 0.001–0.005`. | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) |
| **GRPO itself** — group-relative advantages, no value network | Introduced for math reasoning, then scaled up as the core of DeepSeek-R1. | [DeepSeekMath](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |

A few practical notes drawn from that work:

- **Our filter is a cached, *offline* approximation of DAPO's *online* dynamic sampling.** DAPO re-filters every batch as the model improves (so problems that become trivial drop out mid-training); we measure difficulty once up front and cache it. Cheaper and workshop-friendly, but less adaptive.
- **Filtering isn't free lunch.** It pays off when the data is genuinely too hard (raw MBPP for a 0.5B model). On data that's *already* in the learnable zone, difficulty-filtering can strip useful signal — so the sibling [math tutorial](../llm-fine-tuning-grpo-math) on GSM8K skips it.
- **Truncated completions are worth masking.** Verbose outputs cut off at `max_completion_length` become un-runnable; masking their reward (TRL's `mask_truncated_completions`, DAPO's "overlong filtering") avoids training on that garbage.
- **The training curve can lie.** In every failed run here the *reward chart went up* — the reward-hacking and overfitting were only visible in the held-out eval and the actual generated code. Always read the completions, not just the loss/reward.

TODO:

Deploy model and slect lora or no lora or change out lora from run id or something. 
