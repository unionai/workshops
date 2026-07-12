# GRPO Fine-Tuning: Teach a Model to Write Code

Teach a language model to write correct Python functions using **GRPO (Group Relative Policy Optimization)**. The reward is dead simple: does the generated code actually pass the test cases?

This is the same RL technique DeepSeek used for R1 — applied here to code generation instead of math. Unlike supervised fine-tuning, which memorizes one "correct" answer, GRPO generates multiple completions per prompt and reinforces whichever ones pass the tests. The model discovers its own solutions.

## How GRPO Works

For each training prompt, the model generates several completions. A reward function scores each one, and GRPO reinforces the best within the group:

```
Prompt: "Write a function called `reverse_string` that reverses a string.

def reverse_string(s):"

    ├── Completion 1: "return s[::-1]"               → 3/3 tests pass → reward: 1.0
    ├── Completion 2: "return ''.join(reversed(s))"  → 3/3 tests pass → reward: 1.0
    ├── Completion 3: "return s[0]"                  → 0/3 tests pass → reward: 0.0
    └── Completion 4: "return list(s)"               → 0/3 tests pass → reward: 0.0

GRPO computes advantages within the group: completions 1 & 2 get positive
advantage, 3 & 4 get negative. The policy shifts toward code that passes tests.
```

The key insight: there's no single "correct" answer to learn. Any code that passes is good code. This is why RL works better than SFT here — it rewards the *outcome*, not a specific implementation.

**One consequence drives every choice in this tutorial:** the advantage is computed *within* a group, so a group only teaches the model something if its completions **disagree**. Sixteen completions that all fail produce exactly as much gradient as sixteen that all pass — none. Keep that in mind; it explains the base model, the reward, and the filter. See [Choosing a Base Model ↓](#choosing-a-base-model).

## What's in the Pipeline

```
┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌────────────┐
│ Prepare Data │──▶│   Filter?    │──▶│   GRPO Training  │──▶│  Evaluate  │
│  (CPU task)  │   │  (optional)  │   │    (GPU task)    │   │ (GPU task) │
└──────────────┘   └──────────────┘   └──────────────────┘   └────────────┘
 MBPP candidate     Opt-in via         GRPO with sandboxed    Base vs fine-tuned
 pool from HF       --use_filter       code execution         sandboxed comparison
                                       (LoRA or full)
```

1. **Prepare data** — Downloads the MBPP dataset from HuggingFace and builds a *candidate pool* of prompts, with function signatures extracted from the reference solutions.
2. **Filter for learnability** *(optional)* — Samples the base model over the pool and keeps only problems it solves *sometimes but not always* — the zone where GRPO's within-group advantage is non-zero. Enable with `--use_filter`; it's cached, so it runs once and later runs reuse the result. [Details ↓](#the-learnability-filter)
3. **Train with GRPO** — For each prompt, generates multiple completions, executes them in a [sandbox](#sandboxed-code-execution), and rewards code that passes *all* tests.
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

### Devbox — start it with `--gpu`

If you run against a local Flyte devbox, it **must** be started with `--gpu`:

```bash
flyte start devbox --gpu
```

The flag adds `--gpus all` to the underlying `docker run` *and* selects the
`flyte-devbox:gpu-latest` image. Both are fixed when the container is created, so a
devbox started without it can't be repaired by restarting — replace it:

```bash
docker rm -f flyte-devbox && flyte start devbox --gpu
```

Verify the cluster can actually see the GPU before launching anything:

```bash
kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'   # want: 1
```

An empty result means Kubernetes doesn't know the GPU exists. Training pods then sit in
`Pending` **forever** with `0/1 nodes are available: 1 Insufficient nvidia.com/gpu`, and
because no container ever starts there are no task logs to explain it — it looks exactly
like a slow queue. `kubectl describe pod -n flyte <pod>` is what tells the two apart.

> **Note:** the devbox holds your secrets, so replacing the container **wipes them**.
> Re-create `HF_TOKEN` (below) after any `docker rm -f flyte-devbox`.

### HuggingFace token

`config.py` declares `HF_TOKEN` as a [`flyte.Secret`](https://www.union.ai/docs/flyte/user-guide/secrets/)
on both task environments, so it has to exist in **two** places — they are not
interchangeable.

**1. Locally** — used by `flyte run --local` and by the CLI itself:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

**2. In the Flyte backend** — used by remote task pods:

```bash
flyte create secret HF_TOKEN     # prompts for the value; nothing hits your shell history
flyte get secret                 # verify
```

A `.env` alone is **not** enough for a remote run. The secret is injected into the pod by
an admission webhook, and if the backend doesn't have it the webhook refuses to create the
pod at all — the run fails before any container starts, so there are no task logs. The only
trace is a cluster event:

```bash
kubectl get events -n flyte --sort-by=.lastTimestamp | grep -i secret
# admission webhook "flyte-pod-webhook.flyte.org" denied the request:
# none of the secret managers injected secret [key:"HF_TOKEN" ...]
```

## Run

```bash
flyte run workflow.py pipeline
```

That's the default configuration, shown explicitly:

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-Coder-7B" \
  --method lora \
  --no-use_chat_template \
  --epochs 2 \
  --lr 5e-5 \
  --batch_size 8 \
  --num_generations 8 \
  --max_completion_length 192 \
  --max_candidate_samples 300 \
  --num_eval_examples 50
```

**Requirements:** one large-memory GPU (~40 GB+; a 128 GB DGX Spark is comfortable) and
roughly **2–3 hours** end to end. The live reward and pass-rate charts climb visibly
during training, and the evaluation report compares base vs. fine-tuned on held-out
problems.

Three things about this configuration are load-bearing.

**The base model is a 7B, and that is the point.** GRPO's gradient comes entirely from
groups where *some* completions pass and some fail. A sub-2B model almost never passes a
raw MBPP problem, so nearly every group comes back all-zero and teaches nothing — the run
looks busy while barely learning. A 7B code-pretrained base passes often enough to produce
that variance. This is the single biggest lever on the task, bigger than any
hyperparameter. [Choosing a Base Model ↓](#choosing-a-base-model) covers how to reason
about it when you swap models.

**Use the base, not `-Instruct`, and turn the chat template off.** Qwen ships a ChatML
`chat_template` in its *base* tokenizer too, so `--use_chat_template` would wrap a base
model in a chat format it was never trained on, and it will ramble instead of completing
the function. `--no-use_chat_template` makes this a pure completion task — continue the
MBPP stub — which is what a base model is good at. The flag applies to **both** training
and evaluation, so the before/after comparison stays apples-to-apples either way. Use an
`-Instruct` model *with* the chat template if you prefer; just keep the two consistent.

**Budget for the KL reference model.** With `--beta` above 0, TRL keeps a frozen copy of
the base model to compute the KL term, so plan on roughly **2× the weights** — about 28 GB
for a 7B in bf16, plus LoRA and activations. That's what makes a 7B comfortable and a 32B
dense model impractical here.

### Quick sanity check

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-Coder-0.5B" \
  --max_candidate_samples 20 \
  --max_train_samples 6 \
  --epochs 1 \
  --num_generations 2 \
  --batch_size 2 \
  --max_completion_length 64 \
  --num_eval_examples 4
```

Tiny model, tiny dataset — finishes in a few minutes on a small GPU. This verifies the
pipeline runs end to end (data → sandbox → training → eval). It is **not** expected to
produce a learning signal; a 0.5B on 6 problems is far below the threshold where GRPO has
anything to work with. Use it to check plumbing, not results.

### Longer run

More data and more generations per group, which sharpens the advantage estimates:

```bash
flyte run workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-Coder-7B" \
  --no-use_chat_template \
  --epochs 2 \
  --batch_size 16 \
  --num_generations 16 \
  --max_completion_length 192 \
  --max_candidate_samples 800 \
  --num_eval_examples 50
```

Doubling `--num_generations` doubles the odds that any given prompt yields a mixed
pass/fail group, so it buys more *usable* gradient — but every step costs proportionally
more generation time, and generation is the bottleneck. Keep `--batch_size` equal to
`--num_generations` (one prompt group per step) unless you have memory to spare.

### Full fine-tuning (no LoRA)

```bash
flyte run workflow.py pipeline --method full --model_name "Qwen/Qwen2.5-Coder-0.5B"
```

By default the pipeline trains LoRA adapters, which freeze most weights and learn small
low-rank matrices. `--method full` updates all parameters — more expressive, but far more
memory and much slower. At 7B, full fine-tuning needs weights + gradients + optimizer
state + the KL reference in memory at once; prefer LoRA unless you have the headroom.

`--method qlora` loads the base in 4-bit, which fits a 7B on a 16 GB GPU at the cost of
slow generation. It saves the adapter rather than a merged model.

### Local execution (no cluster)

```bash
flyte run --local --tui workflow.py pipeline \
  --model_name "Qwen/Qwen2.5-Coder-0.5B" \
  --max_candidate_samples 15 --max_train_samples 8 --epochs 1
```

Runs everything on your machine. Useful for debugging; slow without a GPU.

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `Qwen/Qwen2.5-Coder-7B` | HuggingFace model to fine-tune |
| `--method` | `lora` | `lora` for LoRA adapters, `full` for all parameters, `qlora` for a 4-bit base |
| `--epochs` | `2` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `8` | Completions per training step (must be divisible by `--num_generations`) |
| `--num_generations` | `8` | Completions generated per prompt (the "Group" in GRPO) |
| `--max_completion_length` | `192` | Max tokens per generated completion |
| `--use_chat_template` | `False` | Wrap prompts in the tokenizer's chat template. Keep off for **base** models, on for `-Instruct`. Applied to training *and* eval |
| `--use_filter` | `False` | Opt in to the [learnability filter](#the-learnability-filter). Off by default — training uses the whole candidate pool |
| `--max_candidate_samples` | `300` | Size of the candidate pool (also what the filter draws from) |
| `--max_train_samples` | `100` | *(only with `--use_filter`)* Cap on the *learnable* subset kept by the filter. **Ignored without `--use_filter`** — the training pool is then just `--max_candidate_samples` |
| `--filter_samples` | `4` | *(only with `--use_filter`)* Base-model samples per candidate; keep if `1 ≤ all-pass < filter_samples` |
| `--max_eval_samples` | `50` | Size of the held-out eval pool |
| `--num_eval_examples` | `50` | Problems used in the before/after comparison |
| `--eval_k` | `4` | Samples per problem at eval; the score is mean pass@1 over these |
| `--beta` | `0.04` | KL penalty vs. the base model — the main overfit/drift guard |
| `--lora_r` | `16` | LoRA rank — higher = more capacity, more params |
| `--lora_alpha` | `32` | LoRA scaling factor (effective scale = alpha/r) |
| `--precision` | `auto` | `auto` picks bf16 where supported, else fp16. Force `fp32` if generation crashes with `inf/nan` logits |

> **`--batch_size` must be divisible by `--num_generations`** — a GRPO requirement, since
> each optimizer step processes whole generation groups.

> **If sampling crashes with `probability tensor contains inf/nan`** (which can surface as
> a CUDA device-side assert rather than an obvious numerical error), pass
> `--precision fp32`. Low-precision generation can produce `nan` logits on some newer GPUs;
> fp32 costs ~2× memory and is slower, but is numerically bulletproof.

## Choosing a Base Model

This is the most important decision in the whole pipeline, and it is easy to get wrong in a
way that looks like a hyperparameter problem.

**GRPO can only sharpen what the model can already do occasionally.** Advantages are
computed within a group of completions for one prompt. If all of them fail, the group is
flat and contributes no gradient. If all of them pass, likewise. Learning happens *only* in
the middle — problems the base solves **sometimes**. So the useful question about a base
model is not "how good is it?" but **"how often does it land in the middle?"**

That fraction collapses fast as models get smaller:

| Base | Behavior on raw MBPP | Result |
|---|---|---|
| Sub-2B general | Almost never passes | Nearly all groups all-zero → **no gradient**, flat run |
| Sub-2B code-pretrained | Rarely passes | Mostly dead groups → weak, noisy signal |
| **7B code-pretrained** | Passes *sometimes* | Plenty of mixed groups → **real gradient** |
| 7B+ instruct / larger | Passes often | Signal, plus more trivial (all-pass) groups |

`Qwen2.5-Coder-7B` is the default because it's the smallest model in this pipeline that
reliably produces a **held-out** improvement on MBPP. Smaller models will run to completion
and produce charts — the training reward may even drift upward — while the held-out pass
rate barely moves.

**If you swap in a different model, check this first.** Before blaming `--beta`, `--lr`, or
the filter, look at whether the groups have any variance at all:

- **Reward pinned near 0 for the whole run** → the base almost never solves anything. The
  groups are dead. Reach for a **bigger or more code-pretrained base**, not a knob.
- **Reward pinned near 1** → the problems are trivial for this base. Feed it harder data.
- **Reward in the middle and moving** → you have signal; *now* the hyperparameters matter.

This is the RLVR lesson worth internalizing, and it generalizes well beyond code: **RL
amplifies existing competence, it doesn't create it.** Whether you get that variance from a
stronger base or by curating the data for it, it's the thing that makes GRPO work — usually
more impactful than any hyperparameter.

Scale is the other half of the story. Labs get large coding gains from RLVR because they
pair a strong base with **10⁴–10⁶ problems**; at a few hundred problems, each one is more
of an island than an instance of a general skill, so transfer is limited even with a good
base. This tutorial is sized to teach the mechanics — sandboxed verification, binary
reward, the group, the training loop — on hardware you can actually book. For a task
engineered to give a clean win even at small scale, and a framework for scoring your own
task, see the [Countdown tutorial's task-selection guide](../llm-fine-tuning-grpo-countdown#choosing-a-task--and-a-model--for-grpo).

## Tuning Guide — What the Knobs Actually Do

The table above lists the flags; this is how to *reason* about the ones that matter for
GRPO specifically (the LoRA and learning-rate knobs behave as they do in any fine-tune).
GRPO's failure modes are subtle — training reward can climb while the model gets *worse* on
held-out problems — so it helps to know which knob addresses which symptom.

### `--beta` — the KL leash

GRPO penalizes drift away from the **base model's** behavior, scaled by `beta`. Think of it
as a leash:

- **`beta` too low (→ 0):** long leash. The policy wanders wherever the reward points,
  including into degenerate or memorized behavior. Symptom: **train reward climbs, held-out
  accuracy drops**; output style drifts (chatty code that gets truncated); entropy collapses.
- **`beta` too high:** short leash. The policy can barely move from the base, so it can't
  learn. Symptom: **flat 0pp** — the fine-tuned model is nearly identical to the base.
- **The sweet spot is narrow.** `0.005–0.04` is a reasonable range here. If you see a
  regression, this is the first knob to reach for — before touching the model or the data.

### `--num_generations` — the group size

This is the "Group" in **G**RPO. Advantages are computed *within* the group of completions
for one prompt, so the group needs **variety of outcomes** to produce a gradient. If all N
completions get the same reward, that prompt teaches nothing that step.

- **Higher** (8–16): better advantage estimates, more exploration, more stable — and a
  higher chance any given prompt yields a mixed group. Every step costs proportionally more
  generation time, which is the GRPO bottleneck.
- **Lower** (2–4): faster, noisier, far more prone to zero-variance groups.
- It must divide `--batch_size`.

### `--epochs` — how many passes over the training set

More epochs over the *same* problems is the fastest route to **memorization**: the model
overfits and generalizes worse. Prefer **1–2 epochs**. If you want more training signal, add
*data* (`--max_candidate_samples`) before adding epochs.

### `--filter_samples` — how hard the difficulty filter looks

*(Only relevant with `--use_filter`.)* How many times the base model attempts each candidate
during the [learnability filter](#the-learnability-filter). A problem is kept only if it's
solved `1 ≤ x < filter_samples` times.

- **Higher** (6–8): a sharper estimate of the learnable zone, and a wider band — at linear
  cost to the (cached) filter pass.
- **Lower** (3): cheaper, coarser. Below 3, the "sometimes" signal is too noisy to mean much.

### Symptom → knob cheat-sheet

| You see... | Most likely cause | Reach for |
|---|---|---|
| Reward stuck near 0 the whole run | No learnable signal — the base can't solve these | A **bigger / more code-pretrained base** ([above](#choosing-a-base-model)); more data |
| GRPO **worse** than base, output drifted/verbose | Policy drift, too little regularization | Raise `--beta`, lower `--epochs` |
| GRPO **identical** to base (0pp) | Policy can't move | Lower `--beta`, raise `--epochs`/`--lr` |
| Reward jumps to a degenerate constant | Reward is hackable | Make the reward stricter (see below) |
| Training very slow | Generation cost | Lower `--num_generations` / `--max_completion_length` |
| Held-out eval flat but noisy | Too few eval problems | Raise `--num_eval_examples` / `--eval_k` |

## Dataset

The pipeline uses [**MBPP** (Mostly Basic Python Programming)](https://huggingface.co/datasets/google-research-datasets/mbpp),
a standard benchmark of ~970 Python programming problems from Google Research. Each problem
has a natural-language description, a reference solution, and 3 executable test assertions.

The `prepare_data` task downloads MBPP, extracts the function signature from each reference
solution, and builds a prompt for the model to complete:

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

Problems range from simple (`is_even`, `reverse_string`) to moderate (`fibonacci`,
`remove_duplicates`) to harder (heap operations, regex matching). `--max_candidate_samples`
sets the pool size, `--max_train_samples` caps the problems kept for training (the
*learnable* subset when the filter is enabled, otherwise the pool itself), and
`--max_eval_samples` sets the held-out eval set.

## The Reward Function

The reward is **binary** — all or nothing:

| Result | Reward |
|--------|--------|
| All tests pass | 1.0 |
| Any test fails / invalid code | 0.0 |

This is deliberate. Partial credit (`passed/total`) is trivially hackable: a constant that
returns the right *type* passes a fraction of the asserts, and on an impossible problem that
fraction beats the genuine (failing) attempts — so GRPO learns to emit `return True`.
All-or-nothing removes that gradient entirely: an impossible problem produces an all-zero
group with **no advantage and no gradient**, so it is inert rather than hackable.

That's what makes it safe to train without the [learnability filter](#the-learnability-filter).
Impossible problems don't corrupt training; they just don't contribute. The signal comes from
the learnable groups — problems where some completions pass and some don't — which is exactly
what a strong enough base supplies.

## The Learnability Filter

*Optional (`--use_filter`), off by default.*

Two kinds of problem teach the model nothing:

- **Impossible** — the base never solves it; every completion scores 0. With a binary reward
  these contribute no gradient, so they're harmless — they only **waste compute**.
- **Trivial** — the base always solves it; every completion scores 1, so the advantage is zero.

With `--use_filter`, the `filter_learnable` task samples the base model `filter_samples`
times per candidate and keeps only the problems solved *sometimes but not always*
(`1 ≤ all-pass count < filter_samples`) — the learnable middle, where every group carries a
real gradient. The task is cached on its inputs, so the cost is paid once.

The filter is an **efficiency** lever, not a correctness requirement: it concentrates compute
on the groups that matter. It is most valuable when the data is genuinely too hard for the
base — and least valuable when the pool is already mostly in the learnable zone, where
difficulty-filtering can strip useful signal. Turn it on when a run shows most groups are
dead (reward pinned near 0) *and* you've already confirmed the base is strong enough.

## How to Think About Rewards

The reward function *is* the task definition. GRPO doesn't know what "good code" means — it
only knows the number your reward returns, and it will find the **shortest path to a high
number**, whether or not that path is what you intended. Most of the work in an RL project is
reward design, not RL.

**1. The reward is a proxy — the model optimizes the proxy, literally.** This is the central
hazard, usually called *reward hacking*. If a partially-correct constant scores 0.33 and the
honest attempts score 0.0, the model learns to emit the constant. The fix isn't a better
optimizer; it's a reward with no cheap exploit. Ask of any reward: *"what's the laziest output
that scores well here?"* If that output isn't what you want, the reward is wrong.

**2. Sparse vs. dense is a real trade-off.** A **binary** reward is unhackable but *sparse* —
on hard problems every attempt scores 0, so there's no gradient. A **dense** reward (partial
credit) gives signal on near-misses but opens a hacking surface. This tutorial keeps the reward
binary and gets its gradient from the **data**: the signal lives in problems the base solves
sometimes. If you genuinely need a dense reward, weight the parts so the cheap wins can't
dominate.

**3. Multi-part rewards encode multiple goals — weight them carefully.** The sibling
[math tutorial](../llm-fine-tuning-grpo-math) uses `1.0 × correctness + 0.2 × format`:
correctness is the goal, format is a gentle nudge to keep output parseable. Keep the real
objective dominant, or the model will farm the cheap secondary reward (perfect formatting
around wrong answers).

**4. Prefer verifiable rewards when you can get them.** Code (run the tests) and math (check
the answer) give an **objective, ungameable oracle** — no learned reward model to drift or be
gamed. That's why they're the poster children for RL fine-tuning. The catch is that
"verifiable" only covers *what counts as correct* — you still design the shaping, the data,
and the regularization.

**Why the reward changes with the use case.** There's no universal reward — it's a function of
what you can measure and what you're willing to accept:

| Use case | Natural reward signal | Watch out for |
|---|---|---|
| Code generation | Unit tests pass | Constants that pass a subset; gameable tests |
| Math / reasoning | Final answer matches | "Right answer, nonsense reasoning"; format parsing |
| SQL / tool use | Query runs & returns expected rows | Syntactically-valid-but-wrong; empty results scoring "no error" |
| Summarization / writing | **No cheap verifier** → LLM-as-judge or rubric | Judge bias, verbosity/sycophancy hacking |
| Style / safety constraints | Regex / classifier / format checks | Model satisfies the checker while violating the intent |

The rule of thumb: **if a cheap program can verify the outcome, use it and keep the reward
strict; if it can't, you're now also in the business of building (and defending) a reward
model** — a much harder problem, and where most reward-hacking horror stories come from. Start
strict and binary; add density or sub-rewards only when the model can't learn — and every time
you add a term, re-ask *"what's the laziest way to score well now?"*

## Sandboxed Code Execution

The model generates arbitrary Python that has to be executed to compute rewards. Rather than
calling `exec()` in the training process (dangerous with untrusted code), this tutorial uses
[Union interactive sandboxes](https://docs.union.ai/docs/v2/union/user-guide/sandboxing/interactive-sandboxes/)
(`union.sandbox.on_device`):

- **Network blocked** — generated code cannot make network requests
- **Process isolation** — crashes in generated code don't affect the trainer
- **Persistent session** — one sandbox stays open for the whole training run, avoiding
  per-evaluation setup cost
- **Timeout enforcement** — each execution has a 5-second timeout, preventing infinite loops

The sandbox session is opened once per task, and the reward function calls into it from the
trainer thread with `asyncio.run_coroutine_threadsafe()`.

## Reading the Reports

During training, the Flyte report updates live:

- **Progress bar** — current step, epoch, and loss
- **Training loss chart** — loss curve over epochs
- **Reward chart** — average reward over training batches (running average + per-batch)
- **Pass rate chart** — percentage of completions passing all tests
- **Stat grid** — method, dataset size, epochs, learning rate, generations

The evaluation report scores each problem as **mean pass@1 over `--eval_k` samples**, not a
single greedy draw. This matters: GRPO raises *P(correct)* — it turns a problem the model
solves 1-in-4 times into one it solves 3-in-4. A single greedy sample collapses that
probability back into one pass/fail bit and throws most of the improvement away, which is how
a real training gain can show up as a flat eval.

Example problems are grouped so the comparison stays honest rather than cherry-picked:

- **GRPO improved it** — passes more often after training (sorted by largest gain)
- **Both solved it** — the base already had these
- **Neither solved it** — still out of reach
- **GRPO regressed** — the base passed more often; watch this bucket for drift

Each card shows a *passing* sample where one exists, and labels it as such — so the code you
read is the code the badge is talking about.

**The training curve can lie.** Reward climbing is not proof of learning: a policy that drifts
into a degenerate style can push training reward up while held-out accuracy falls. Always read
the held-out eval and the actual generated code, not just the loss and reward charts.

## Understanding the Training Loop

Unlike standard SFT, where the trainer just does forward/backward passes, GRPO's inner loop is:

```
For each batch of prompts:
  1. Generate `num_generations` completions per prompt (inference)
  2. Score each completion with the reward function (sandbox execution)
  3. Compute advantages within each group (normalize rewards)
  4. Update the policy to increase the probability of high-reward completions (training)
```

Every training step involves both **inference** (generating code) and **execution** (running
tests), which is why GRPO is slower than SFT. Generation dominates the wall-clock and is
memory-bandwidth-bound, so cost scales with the bytes of weights read per token —
`num_generations` and `max_completion_length` are the two knobs that move it most.

## Scaling This to Production

This tutorial collapses generation, verification, and training into a single GPU task with one
in-process sandbox — deliberately, so the pipeline fits on one machine and stays easy to
follow. In real RLVR pipelines the wall-clock is dominated by **rollout generation** and
**reward verification**, not the gradient step, so that's where you scale — and you can do it
without changing how code is executed:

- **Warm worker pools** — put a `flyte.ReusePolicy` on the GPU environment so the policy model
  loads once and stays hot across rollout batches instead of cold-starting a container per step
  (`throughput = replicas × concurrency`).
- **Fan out the rollouts** — the group dimension (`num_generations`) and the prompt batch are
  embarrassingly parallel. Spread them across the warm pool with `flyte.map(...)` /
  `asyncio.gather()` — each action in its own container — bounded by an `asyncio.Semaphore` or
  `flyte.map(concurrency=...)` to respect GPU quota.
- **Keep verification on-device** — each fanned-out worker verifies its own shard of completions
  with its own `sb.on_device.session()`. No shared bottleneck, and the same reward function you
  see here, just running on more boxes.

What Union does *not* do for you is the RL-engine internals: syncing updated policy weights to
inference workers each step, off-policy correction, matching generation throughput to training.
That lives in your trainer↔inference integration (TRL's vLLM mode, veRL). Union gives you the
distributed substrate and the secure verifier; you plug in the RL engine.

## Why Code Generation for GRPO?

Code is an ideal RL task because it is:

- **Verifiable** — run it and check; no subjective judgment needed
- **Multi-solution** — `s[::-1]` and `''.join(reversed(s))` both work, and GRPO rewards either
- **Objectively scored** — a passing test suite is an unambiguous signal
- **Practical** — this is how production code models are actually trained

## Background & Further Reading

None of the design choices here are arbitrary — each maps to a published finding.

| Choice in this tutorial | Why | Reference |
|---|---|---|
| **A base strong enough to sometimes succeed** | A group where every completion gets the same reward has zero advantage and zero gradient. Learning concentrates where the success rate is intermediate (~0.5). | [DAPO](https://arxiv.org/abs/2503.14476) · [Competence–Difficulty Alignment](https://arxiv.org/abs/2505.17652) |
| **Learnability filter** — keep only problems the base solves *sometimes* | The same principle applied to the *data* instead of the model. This is what DAPO calls **Dynamic Sampling**. | [DAPO](https://arxiv.org/abs/2503.14476) |
| **Binary reward** — 1.0 only if all tests pass | Partial pass-rate (`passed/total`) is *hackable*: constants like `return True` grab the easy tests (a skew VeRPO calls **cardinality bias**), and partial credit doesn't beat binary at convergence anyway. | [Beyond Binary / VeRPO](https://arxiv.org/abs/2601.03525) |
| **KL penalty** (`--beta`) | With too little KL regularization the policy drifts from the base and reward-hacks — train reward climbs while held-out accuracy *drops*. | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) |
| **GRPO itself** — group-relative advantages, no value network | Introduced for math reasoning, then scaled up as the core of DeepSeek-R1. | [DeepSeekMath](https://arxiv.org/abs/2402.03300) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) |

A few practical notes drawn from that work:

- **This filter is a cached, *offline* approximation of DAPO's *online* dynamic sampling.** DAPO
  re-filters every batch as the model improves, so problems that become trivial drop out
  mid-training; here difficulty is measured once up front and cached. Cheaper, but less adaptive.
- **Filtering isn't a free lunch.** It pays off when the data is genuinely too hard. On data
  already in the learnable zone it can strip useful signal — which is why the sibling
  [math tutorial](../llm-fine-tuning-grpo-math) on GSM8K skips it too.
- **Truncated completions are worth masking.** Verbose outputs cut off at
  `max_completion_length` become un-runnable; masking their reward (TRL's
  `mask_truncated_completions`, DAPO's "overlong filtering") avoids training on that garbage.
