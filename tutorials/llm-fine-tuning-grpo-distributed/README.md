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

Level 2 moves generation out. Each round:

```
1. learner  : save the current LoRA adapter        -> flyte.io.Dir
2. rollouts : fan out to the vLLM pool             (rollout.py, L40s, reusable)
3. verify   : fan out to the sandbox pool          (verify.py, CPU, reusable)
4. roll up  : (prompt, completion, logprob, reward)
5. learner  : group advantages -> GRPO loss -> one optimizer step
```

The learner is one long-lived task that drives all of it, not one task per round. It loads a
14B base once and keeps optimizer state in memory; per-round tasks would reload the model every
round and either lose Adam's moments or serialize them to blob storage each time.

**Weight sync is the part no orchestrator does for you**, and here it is about six lines: the
learner writes the adapter, the rollout workers load it by path. That is cheap only because we
train LoRA — the adapter is tens of MB while the base is ~28GB, and the base never moves.

### Reuse is what makes the rollout pool viable

`rollout.py` caches the vLLM engine in a module global:

```python
_ENGINE = None   # survives across task invocations on a reusable replica

if _ENGINE is None:
    _ENGINE = LLM(model=_BASE_PATH, enable_lora=True, max_lora_rank=lora_rank, ...)
```

Building the engine and loading 28GB takes minutes. On a *non*-reusable environment every call
is a fresh container, `_ENGINE` is always `None`, and this would be strictly slower than
generating in-process. Reuse is not an optimization here; it is the enabling condition.

### The trap: vLLM caches LoRA adapters by integer id

```python
lora_request=LoRARequest(f"policy-r{round_id}", round_id + 1, adapter_path)
```

`round_id` **must** increase every round. vLLM keys its adapter cache on the integer id — reuse
an id and it serves the *previously cached* adapter and silently ignores the new weights on
disk. Training then looks like it runs fine while every round rolls out the round-0 policy.

This is the nastiest failure mode in the design because nothing errors. **Symptom:** mean reward
wanders around its round-0 value forever. If you see that, check this before anything else.

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

| | Level 1 | Level 2 |
|---|---|---|
| base weights (bf16) | ~28 GB | ~28 GB |
| LoRA + optimizer | ~1 GB | ~1 GB |
| activations (grad ckpt) | ~4–8 GB | ~4–8 GB |
| **KV cache for generation** | **on the same card** | **on the rollout workers** |
| verdict | tight — lower `batch_size`/`num_generations` if you OOM | comfortable |

Generation and training competing for one card is exactly what Level 2 removes. If you are
running Level 1 at 14B and hitting OOM, that is the constraint talking — drop to 7B, shrink the
group, or move to Level 2.

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
