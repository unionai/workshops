# Distributed GRPO — Fanning Out Rollouts and Verification

The [single-GPU GRPO tutorial](../llm-fine-tuning-grpo-code) puts generation, verification,
and the gradient step in one task on one card. That is the right way to learn the mechanics.
It is also, past a certain size, the wrong way to run it.

This tutorial takes that pipeline apart and spreads it across the cluster, in two levels:

| | What moves off the learner | GPUs | Ships |
|---|---|---|---|
| **Level 1** (`workflow.py`) | Verification → reusable CPU pool | 1× L40s | Real speedup, small diff, keeps TRL |
| **Level 2** (`distributed.py`) | Verification **and** generation → two pools | 1× L40s learner + 1–4× L40s rollout | The full RL loop, written out |

Level 1 stands on its own. Start there.

---

## Why: the reward function is the bottleneck

Look at the reward function in the single-GPU tutorial. It is a serial loop:

```python
for completion, p, t, setup in zip(completions, func_prompt, tests, setup_code):
    future = asyncio.run_coroutine_threadsafe(run_tests_sandboxed(sbx, full_code, test_list), loop)
    all_passed, passed, total = future.result(timeout=10)     # one at a time
```

At the default `batch_size=8 × num_generations=8`, every training step does **64 sequential
sandbox executions**, each with a 5s timeout. The GPU sits idle for essentially all of it.
This is the general shape of RLVR: wall-clock is dominated by rollout generation and reward
verification, not by the gradient step.

The two levels here attack those two terms in that order.

---

## Level 1 — fan out verification

The only change is the body of the reward function. It shards its completions and dispatches
them to a warm pool:

```python
shards = chunk(items, shard_size)
future = asyncio.run_coroutine_threadsafe(
    asyncio.gather(*(verify_shard(s, verify_timeout_s) for s in shards)), loop)
results = [r for shard in future.result(timeout=timeout) for r in shard]
```

TRL's `GRPOTrainer`, the binary reward, the KL anchor, and every hyperparameter are untouched,
so any speedup is attributable to the fan-out alone. Throughput becomes `replicas ×
concurrency` — up to **160 concurrent shards** with the defaults in `config.py`.

### The constraint that shapes everything: reuse vs. bubblewrap

The single-GPU tutorial sandboxes with `backend="bubblewrap"`, the strongest backend. The
verifier pool **cannot**, and the reason is a hard constraint rather than an oversight:

1. bwrap needs `CAP_SYS_ADMIN` + unconfined AppArmor on the pod.
2. The only way to request those is `flyte.PodTemplate().allow_nested_sandboxing()`.
3. A reusable `TaskEnvironment` **cannot set `pod_template` at all**:

```
ValueError: Cannot set pod_template when environment is reusable.
```

So "warm pool" and "bubblewrap" are mutually exclusive today. The pool runs `backend="userns"`
(userns-lite), which needs no extra pod capabilities — which is exactly why it works here.

**userns is weaker isolation than bwrap.** It relies on unprivileged user namespaces without
the seccomp/AppArmor posture bwrap gets. Combined with `network_mode="blocked"` (verified —
outbound requests raise `URLError`) it is a reasonable choice for scoring model-generated MBPP
solutions. It is *not* the right default for genuinely adversarial code.

If you need bwrap-grade isolation, take the other side of the trade — drop `reusable`, add the
pod template back, and fan out with `flyte.map`:

```python
verify_env = flyte.TaskEnvironment(
    name="grpo-dist-verify",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
)   # no `reusable=` — these two cannot coexist
```

You keep the parallelism and the strong sandbox, and pay container cold-start on every shard
instead of amortizing it across a pool. **That is the actual trade: isolation strength vs.
cold start.** Pick deliberately; don't drift into one by accident.

`verify.py` will never silently downgrade to `backend="none"`. If no sandbox is available it
raises. Running model-generated code with no isolation is not a degraded mode of this
pipeline — it is a different and much worse thing.

### Checking the pool is actually warm

Each verifier process stamps a `worker_id` into its results, and the training report shows the
distinct count. This is the number that tells you reuse is engaging:

- **~`replicas` distinct ids** across many shards → the pool is warm, working as intended.
- **one id per shard** → every call cold-started and you got none of the benefit.

---

## Level 2 — disaggregate the whole loop

Level 1 still generates on the training GPU, so that card holds weights, optimizer state,
activations *and* a KV cache for `batch_size × num_generations` sequences. On a 48GB L40s with
a 14B base, that is the binding constraint.

Level 2 moves generation out onto its own warm pool, so the learner card holds only training
memory. The learner drives one round at a time; each round fans out twice — once to generate,
once to verify — and rolls the results back up into a single gradient step:

```
                  ┌───────────────────────────────────────────────┐
                  │  learner   (1× L40s, NOT reusable)             │
                  │  holds base + LoRA + optimizer state,          │
                  │  one long-lived task driving every round       │
                  └───────────────┬───────────────────────────────┘
                                  │
        (1) save current LoRA adapter ──► flyte.io.Dir   (tens of MB, not the 28GB base)
                                  │
        (2) split `prompts_per_round` prompts into `rollout_workers` chunks
                                  │
              ┌───────────────────┴───────────────────┐
              ▼                                        ▼
   ┌─────────────────────┐                 ┌─────────────────────┐   rollout_env
   │  generate_rollouts   │                 │  generate_rollouts   │   REUSABLE, GPU
   │  vLLM engine (warm,  │        …        │  vLLM engine (warm,  │   replicas=(1,4)
   │  cached in-process)  │                 │  cached in-process)  │   concurrency=1
   │  + hot-swap adapter  │                 │  + hot-swap adapter  │   1× L40s each
   └──────────┬──────────┘                 └──────────┬──────────┘
              │  n=`num_generations` completions per prompt        │
              └────────────────────┬───────────────────────────────┘
                                   │
        (3) flatten to (prompt, completion, logprob) rows,
            re-chunk into shards of `shard_size`
                                   │
        ┌──────────────┬───────────┴──────────┬─────────── … up to replicas×concurrency
        ▼              ▼                       ▼
  ┌───────────┐  ┌───────────┐          ┌───────────┐          verify_env
  │verify_shard│ │verify_shard│    …     │verify_shard│          REUSABLE, CPU
  │ userns box │ │ userns box │          │ userns box │          replicas=(2,20)
  │ run tests  │ │ run tests  │          │ run tests  │          concurrency=8
  └─────┬──────┘ └─────┬──────┘          └─────┬──────┘          (≤160 shards at once)
        │  reward = 1.0 iff ALL tests pass, else 0.0             │
        └──────────────────────┬────────────────────────────────┘
                               │
        (4) regroup rewards by prompt (groups of `num_generations`)
                               │
        (5) learner: within-group advantages → GRPO loss → ONE optimizer step
                               │
                               └────────────► next round (back to step 1)
```

The learner is one long-lived task, not one task per round. It loads the base once and keeps
optimizer state in memory; per-round tasks would reload the model every round and either lose
Adam's moments or serialize them to blob storage each time.

**Weight sync is the part no orchestrator does for you**, and here it is about six lines: the
learner writes the adapter (step 1), the rollout workers load it by path (step 2). That is cheap
only because we train LoRA — the adapter is tens of MB while the base is ~28GB, and the base
never moves.

### How the two fan-outs work

Both pools are the same idea — a warm set of workers you dispatch shards to with
`asyncio.gather` — but they shard *different* things and isolate for *different* reasons.

**Rollout sharding (by prompt).** `prompts_per_round` prompts are split into `rollout_workers`
chunks (`distributed.py` uses `chunk(requests, ceil(len/workers))`), and each chunk becomes one
`generate_rollouts` task call. A worker asks its vLLM engine for `num_generations` completions
per prompt in one batched `generate()` — vLLM's continuous batching keeps the GPU busy across the
whole chunk. Each completion comes back with the summed log-prob of its sampled tokens, which the
learner needs for the importance ratio. Workers are **stateful on purpose**: the vLLM engine is
cached in a process global and reused across rounds (see "Reuse" below), so only the LoRA adapter
changes round to round.

**Verify sharding (by completion).** All completions from all rollout workers are flattened and
re-chunked into shards of `shard_size` — a *different* partition, because verification cost scales
with the number of completions, not prompts. Each shard is one `verify_shard` call that opens a
single `userns` sandbox and runs every completion's test suite sequentially inside it, returning a
binary reward per completion. Workers here are **stateless on purpose**: a fresh sandbox per shard,
so code from one completion can't observe or clobber another's files. The two knobs are
independent — `rollout_workers` sets generation parallelism, `shard_size` sets verification
parallelism (`ceil(total_completions / shard_size)` shards, capped by the pool's `replicas ×
concurrency`).

### Reuse is what makes the rollout pool viable

`rollout.py` caches the vLLM engine in a module global that survives across task invocations —
and across whole *runs* — because a reusable replica outlives the run that created it:

```python
_ENGINE = None
_ENGINE_BASE = None   # the REMOTE path of the base model the engine was built from

if _ENGINE is None or _ENGINE_BASE != base_dir.path:
    _ENGINE = LLM(model=..., enable_lora=True, max_lora_rank=_MAX_LORA_RANK, dtype=...)
    _ENGINE_BASE = base_dir.path
```

Building the engine and loading 28GB takes minutes. On a *non*-reusable environment every call is
a fresh container, `_ENGINE` is always `None`, and this would be strictly slower than generating
in-process. Reuse is not an optimization here; it is the enabling condition. **Measured on a
7B/L40s run: rollout `118s → 6s` from round 0 to round 1** — the cold start is paid once, then
every later round reuses the warm engine.

### Stale state is the price of reuse — key every cache on identity

A warm replica outliving its run is the whole point, but it makes stale in-process state the
central hazard. Everything cached must be keyed on something that changes when the work changes,
never on "have I built anything yet." Two real traps, both fixed in `rollout.py`:

**1. The engine, keyed on the base model.** An earlier, sloppier version guarded only on
`_ENGINE is None`. When a 14B run landed on a replica still warm from a 0.5B run, it reused the
0.5B engine — which was built with `max_lora_rank=8` and rejected the 14B run's rank-16 adapter
(`ValueError: LoRA rank 16 is greater than max_lora_rank 8`). That error was lucky: without it,
the run would have **silently generated from the 0.5B weights** and trained against garbage.
Keying on `base_dir.path` (and freeing the old engine before building the new one) fixes it —
`download_model` is cached per `model_name`, so the same model reuses the engine and a different
model rebuilds.

**2. The adapter, keyed on its path.** vLLM caches LoRA adapters by integer id and serves the
cached weights whenever an id repeats. Numbering adapters by `round_id` is unsafe across runs:
run B's round 0 collides with run A's round 0 on a warm replica and serves A's stale adapter.
`rollout.py` instead assigns every *distinct adapter remote path* a fresh monotonic id, so a new
adapter always gets a new id — within a run and across runs. **Symptom if you get this wrong:**
mean reward wanders around its round-0 value forever while nothing errors.

---

## The GRPO objective, written out

Level 1 hides this inside TRL. Level 2 makes it explicit in `learner.py`, and two details there
are worth more than the rest of the file.

### Ratios are per-token, not per-sequence

```python
ratio = torch.exp(torch.clamp(token_lp_new - token_lp_old, min=-20.0, max=20.0))
```

A sequence-level ratio is `exp(Σ per-token diffs)`. On a 192-token completion, a drift of only
0.05 nats per token gives `exp(9.6) ≈ 15000`; at 1024 tokens, `exp(51) ≈ 1.7e22`. The clip band
is blown past instantly and the gradient is garbage. Token-level ratios stay near 1 regardless
of completion length. Measured, with 0.05 nats/token of drift:

| completion length | token-level ratio | sequence-level ratio |
|---|---|---|
| 4 | 1.0513 | 1.22 |
| 192 | 1.0513 | 1.48e+04 |
| 1024 | 1.0513 | 1.72e+22 |

Each sequence is also normalized by its own length before averaging (the `1/|o_i|` in the GRPO
objective). Summing instead would weight long completions more heavily and hand the model a
length bias to exploit.

### Dropout must be off

This is a correctness requirement, not a tuning choice. The ratio `exp(logp_new − logp_old)` is
only meaningful if the two forward passes differ *because the parameters changed*. With dropout
active they also differ by a fresh random mask. Measured on an unchanged batch:

| | mean ratio | KL vs. an **identical** reference |
|---|---|---|
| dropout active | 1.0198 | 0.0055 |
| dropout disabled | **1.0000** | **0.0000** |

That KL is the damning number: a spurious penalty applied every step, from pure noise.
`_disable_dropout()` zeroes every dropout layer at load, and `lora_dropout=0.0` (not the 0.05
you would use for supervised fine-tuning).

### Reading the loss

With `inner_epochs=1` the policy loss is **~0.0 at every step**, and that is correct.
Advantages are zero-mean within a group and the on-policy ratio is exactly 1, so the loss
*value* is `-mean(A) = 0` while its *gradient* is not. **Do not tune against the loss.** Watch
mean reward and the live-group count.

### Live groups is the metric that matters

A group where every completion got the same reward — all passed, or far more often all failed —
has zero variance, therefore zero advantage, therefore zero gradient. It costs a full generation
and verification cycle and teaches nothing. `AdvantageStats.dead_groups` counts them, and the
report shows live/total per round.

If live groups collapse toward zero, **no hyperparameter will save the run.** Either the base
model is too weak to land in the middle band or the problems are too hard. Change the model or
the data, not the learning rate.

---

## Choosing a base model

Unchanged from the single-GPU tutorial, and still the most important decision: **GRPO can only
sharpen what the model can already do occasionally.** The useful question about a base is not
"how good is it?" but "how often does it land in the middle?"

| Base | Behavior on raw MBPP | Result |
|---|---|---|
| Sub-2B general | Almost never passes | Nearly all groups all-zero → **no gradient** |
| Sub-2B code-pretrained | Rarely passes | Mostly dead groups → weak, noisy signal |
| 7B code-pretrained | Passes *sometimes* | Plenty of mixed groups → **real gradient** |
| **14B code-pretrained** | Passes often enough | More live groups per round; the default here |

The default is `Qwen/Qwen2.5-Coder-14B`. Level 2 is what makes 14B comfortable — see below.

## Memory: 14B LoRA on a 48GB L40s

An L40s advertises 48GB but only ~44.4GB is usable after the driver. A 14B in bf16 is ~28GB of
frozen weights, leaving ~16GB for everything else:

| | Level 1 | Level 2 |
|---|---|---|
| base weights (bf16) | ~28 GB | ~28 GB |
| LoRA + optimizer | ~1 GB | ~1 GB |
| activations (grad ckpt) | scales with `micro_batch_size` | scales with `micro_batch_size` |
| **KV cache for generation** | **on the same card** | **on the rollout workers** |
| verdict | tight — lower `batch_size`/`num_generations` if you OOM | fits, but mind `micro_batch_size` |

Level 2 removes generation from the learner card, which is what makes 14B viable at all — but it
is not automatically comfortable. Two settings are load-bearing, and both are defaults here:

- **`micro_batch_size` is the activation-memory knob.** At the tutorial defaults, a 14B update
  OOM'd on the L40s at `micro_batch_size=4` (42.8GB live, ~1GB stranded to fragmentation). The
  default is **2**; drop to **1** if you still OOM, or if you widen `max_completion_length`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (set on `learner_env`) lets the
  allocator reclaim fragmented arenas. Without it, a 14B update can OOM with a gigabyte
  "reserved but unallocated" — memory that would fit, stranded by fragmentation.

If you are running **Level 1** at 14B and hitting OOM, that is generation and training sharing
the card — drop to 7B, shrink the group, or move to Level 2.

## Results

A real Level 2 run on the documented target (L40s, 8 rounds, 16 prompts/round × 8 generations,
50-problem held-out eval at pass@1 over 4 samples):

| Base | Base pass@1 | GRPO pass@1 | Improvement |
|---|---|---|---|
| Qwen2.5-Coder-7B | 38.0% | 40.5% | **+2.5pp** |

Of 50 eval problems, 21 sat in the movable band (base solves them *sometimes*) — that is the
mass GRPO can act on, and the never-solved count stayed flat (22 → 22), exactly as theory
predicts. The per-round training reward rose from ~0.48 to a ~0.61 peak with the expected RL
noise, and the warm rollout engine made every round after the first ~20× cheaper to generate
(`118s → 6s`). This is a modest-but-real gain at small scale; labs get large coding gains by
pairing a strong base with 10⁴–10⁶ problems, and the pipeline here is what lets you scale the
rollout/verify halves to get there.

---

## Setup

```bash
cd tutorials/llm-fine-tuning-grpo-distributed
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

A HuggingFace token is needed for the model download:

```bash
echo "HF_TOKEN=hf_..." > .env
```

`requirements.txt` is for the local venv. The remote images are built from the package lists in
`config.py` via `.with_pip_packages()` — **keep the two in sync by hand.** (`.with_requirements()`
stores a relative path and re-resolves it at runtime, which breaks anywhere the working
directory isn't yours.)

Override the accelerator if you aren't on L40s — note the lowercase `s`, `"L40S:1"` is not a
valid accelerator string:

```bash
export GRPO_GPU="A100:1"
```

## Run

```bash
# Level 1 — 14B on one L40s, verification on the pool
flyte run workflow.py pipeline

# Level 2 — full disaggregation
flyte run distributed.py distributed_pipeline

# Level 2, smaller and cheaper
flyte run distributed.py distributed_pipeline --rounds 3 --prompts_per_round 16 \
    --rollout_workers 2
```

### Wiring check first

Before spending GPU hours, confirm the plumbing on a small model. This produces **no learning
signal** — a 0.5B generates dead groups by design — it only proves the pipeline runs:

```bash
flyte run workflow.py pipeline --model_name "Qwen/Qwen2.5-Coder-0.5B" \
    --max_candidate_samples 20 --epochs 1 --num_generations 2 \
    --batch_size 2 --num_eval_examples 4
```

### Local development

`userns` is Linux-only. On macOS, `verify.py` resolves to `sandbox-exec` with a warning — fine
for a wiring check, not the deployed configuration. Force a backend explicitly with:

```bash
export GRPO_SANDBOX_BACKEND=bubblewrap
```

It errors if that backend isn't actually available, rather than downgrading.

## Parameters

Shared:

| Flag | Default | What it does |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-Coder-14B` | Base model. The load-bearing choice. |
| `--num_generations` | `8` | Group size. More groups with mixed pass/fail = more gradient. |
| `--shard_size` | `4` | Completions per verifier task call. Smaller = more parallelism, more dispatch overhead. |
| `--beta` | `0.04` | KL leash to the base policy. 0 disables. |
| `--lora_r` / `--lora_alpha` | `16` / `32` | Adapter capacity. `lora_r` must be ≤ vLLM's `max_lora_rank`. |
| `--eval_k` | `4` | Samples per problem at eval. `1` = greedy, and throws away most of the measured gain. |

Level 2 only:

| Flag | Default | What it does |
|---|---|---|
| `--rounds` | `8` | Policy updates. One rollout + verify + step cycle each. |
| `--prompts_per_round` | `16` | Problems sampled per round. Completions per round = this × `num_generations`. |
| `--rollout_workers` | `2` | Parallel `generate_rollouts` calls. Capped by `rollout_env` max replicas. |
| `--inner_epochs` | `1` | Gradient steps per rollout batch. `1` is strictly on-policy (ratio ≡ 1, clipping never engages). `>1` is where the clipped surrogate starts doing work, at the cost of stale data. |
| `--micro_batch_size` | `8` | Sequences per forward pass. The knob that keeps 14B inside memory. |

## Files

| File | Role |
|---|---|
| `config.py` | The four task environments and the reuse policies |
| `common.py` | Prompting, code assembly, sandboxed execution, dataset, model download |
| `verify.py` | The verifier pool — backend resolution and `verify_shard` |
| `workflow.py` | **Level 1**: TRL trainer with a fanned-out reward, plus evaluation |
| `rollout.py` | **Level 2**: vLLM generation workers with LoRA hot-swap |
| `learner.py` | **Level 2**: advantages, the clipped surrogate, the KL term |
| `distributed.py` | **Level 2**: the disaggregated loop |

## Symptom → cause

| Symptom | Look at |
|---|---|
| Mean reward never moves off its round-0 value | The `round_id` LoRA-cache trap — rollouts are serving a stale adapter |
| Live groups near zero every round | Base model too weak, or problems too hard. Not a hyperparameter problem. |
| Policy loss is 0.0 every step | Expected with `inner_epochs=1`. Not a bug. Read reward instead. |
| One verifier `worker_id` per shard | Pool is cold-starting; reuse isn't engaging |
| Mean ratio far from 1.0 with `inner_epochs=1` | Dropout is active somewhere, or generation/training dtypes disagree |
| Train reward climbs, held-out pass rate drops | Classic reward drift — raise `--beta` |
| OOM on the learner at 14B | Level 1 shares the card with generation. Shrink the group or move to Level 2. |

## Background

| Choice | Why | Reference |
|---|---|---|
| A base strong enough to sometimes succeed | A group where every completion scores the same has zero advantage and zero gradient. | [DAPO](https://arxiv.org/abs/2503.14476) |
| Binary reward (all tests, or nothing) | Partial credit is hackable: `return True` grabs the easy asserts and becomes the highest-advantage completion in an otherwise-failing group. | [Beyond Binary / VeRPO](https://arxiv.org/abs/2601.03525) |
| KL penalty (`--beta`) | Without it the policy drifts and reward-hacks — train reward up, held-out down. | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) |
| Per-token ratio, length-normalized | Sequence-level ratios explode with completion length; token sums create a length bias. | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) |

## See also

- [`../llm-fine-tuning-grpo-code`](../llm-fine-tuning-grpo-code) — the single-GPU version. Read it first.
- [`../llm-fine-tuning-grpo-countdown`](../llm-fine-tuning-grpo-countdown) — choosing a task and model for GRPO.
