"""Rollout workers — vLLM generation on a warm GPU pool.

At Level 1 the learner still generates its own completions, so the training GPU
holds model weights, optimizer state, activations *and* a KV cache for
`batch_size x num_generations` sequences. On a 48GB L40s with a 14B base that is
the binding constraint.

Level 2 moves generation here. The learner card then holds only training memory,
and generation scales by adding replicas rather than by shrinking the batch.

Two things make this worth doing:

  * **vLLM** batches continuously and uses paged attention, so it generates a
    group of 8 completions far faster than `model.generate()` in a training loop.
  * **Reuse** keeps the engine alive between rounds. Building a vLLM engine and
    loading 28GB of weights takes minutes; doing that once per replica instead of
    once per round is the difference between viable and not.

The engine lives in a module global. That only works because `rollout_env` is
reusable — the process survives across task invocations, so `_ENGINE` is still
there on the next call. On a non-reusable environment every call would be a fresh
container, `_ENGINE` would always be None, and this would be strictly slower than
generating in-process.
"""

import logging
import os
from dataclasses import dataclass

import flyte
import flyte.io

from config import rollout_env

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Process-global engine cache. Survives across task calls on a reusable replica —
# and, crucially, across *separate runs*, because a warm replica outlives the run
# that created it (idle_ttl). That longevity is the whole point (skip cold starts)
# but it makes stale state the central hazard: everything cached here must be keyed
# on identity that changes when the work changes, never on "have I built anything".
#
# `_ENGINE_BASE` is the *remote* path of the base model the engine was built from
# (Dir.path). download_model is cached per model_name, so the same model yields the
# same remote path across runs (reuse the engine, good) and a different model yields
# a different path (rebuild, essential — otherwise a 7B run silently generates from
# a warm replica's 0.5B engine).
_ENGINE = None
_ENGINE_BASE: str | None = None

# vLLM caches LoRA adapters by integer id and serves the cached weights if an id
# repeats. `round_id` alone is unsafe across runs: run B's round 0 would collide
# with run A's round 0 on a warm replica and serve A's stale adapter. Instead we
# assign every *distinct adapter remote path* a fresh monotonic id, so a new
# adapter always gets a new id — within a run and across runs.
_ADAPTER_IDS: dict[str, int] = {}
_NEXT_ADAPTER_ID = 1

# A generous ceiling so any reasonable LoRA rank reuses the same engine (an engine
# built with max_lora_rank=R serves every adapter of rank <= R). Rebuilding the
# engine per rank would defeat reuse; this decouples the two.
_MAX_LORA_RANK = 64

_WORKER_ID = f"{os.getpid()}"


@dataclass
class RolloutRequest:
    """One prompt to generate a group for."""
    index: int      # position in the round's prompt list — used to regroup after fan-out
    prompt: str     # fully built (chat-templated if applicable) prompt text


@dataclass
class RolloutSample:
    """One completion, with the log-prob it was generated under."""
    index: int          # the RolloutRequest.index it belongs to
    completion: str
    logprob: float      # sum of sampled-token logprobs under the *rollout* policy
    num_tokens: int


@rollout_env.task
async def generate_rollouts(
    requests: list[RolloutRequest],
    base_dir: flyte.io.Dir,
    adapter_dir: flyte.io.Dir,
    round_id: int,
    num_generations: int = 8,
    max_completion_length: int = 192,
    temperature: float = 0.9,
    top_p: float = 0.95,
    lora_rank: int = 16,
) -> list[RolloutSample]:
    """Generate `num_generations` completions per prompt under the current policy.

    Args:
        round_id: kept for logging/telemetry only. Adapter identity is derived from
            the adapter's remote path, not this number, so correctness no longer
            depends on round_id being fresh (see `_ADAPTER_IDS`). vLLM caches LoRA
            adapters by integer id and would serve a stale adapter if an id
            repeated; keying the id on the adapter path removes that trap entirely.
        lora_rank: validated against the engine's fixed `_MAX_LORA_RANK` ceiling.
    """
    global _ENGINE, _ENGINE_BASE, _NEXT_ADAPTER_ID

    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    if lora_rank > _MAX_LORA_RANK:
        # Fail loudly rather than let vLLM reject the adapter mid-generation with a
        # confusing "rank > max_lora_rank" deep in EngineCore.
        raise ValueError(
            f"lora_rank={lora_rank} exceeds the rollout engine's max_lora_rank "
            f"({_MAX_LORA_RANK}). Raise _MAX_LORA_RANK in rollout.py to use a larger rank."
        )

    # Build (or rebuild) the engine keyed on the base model's REMOTE identity, not
    # on "have I built anything". A warm replica from a previous run may already
    # hold an engine for a *different* base model; reusing it would silently
    # generate from the wrong weights. Comparing base_dir.path catches that.
    base_key = base_dir.path
    if _ENGINE is None or _ENGINE_BASE != base_key:
        # If a warm replica already holds an engine for a different base model, free
        # its GPU memory before building the new one — otherwise the old (e.g. 0.5B)
        # engine's weights + KV cache linger and can crowd out the new (e.g. 7B) one.
        if _ENGINE is not None:
            log.info(f"[rollout {_WORKER_ID}] base changed ({_ENGINE_BASE} -> {base_key}); freeing old engine")
            del _ENGINE
            _ENGINE = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        local_base = await base_dir.download()
        # bf16 only exists on Ampere+ (A100/L40s/H100…). Turing cards (T4, V100)
        # have no bf16 path, and passing dtype="bfloat16" there aborts engine
        # startup. Detect and drop to fp16 — the learner (distributed.py) does the
        # same, so generation and training stay on the same dtype.
        engine_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        log.info(
            f"[rollout {_WORKER_ID}] building vLLM engine (cold start), "
            f"dtype={engine_dtype}, base={base_key}"
        )
        _ENGINE = LLM(
            model=local_base,
            enable_lora=True,
            max_lora_rank=_MAX_LORA_RANK,
            dtype=engine_dtype,
            # Leave headroom: the adapter, CUDA graphs, and NCCL buffers all live
            # outside the fraction vLLM reserves for weights + KV cache.
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            enforce_eager=False,
        )
        _ENGINE_BASE = base_key
        # A fresh engine has an empty LoRA cache, so our path->id map must reset too,
        # or a stale id from the previous engine could be reused against a new one.
        _ADAPTER_IDS.clear()
        _NEXT_ADAPTER_ID = 1
    else:
        log.info(f"[rollout {_WORKER_ID}] reusing warm engine for round {round_id}")

    # Give each distinct adapter path a fresh, stable int id (see _ADAPTER_IDS).
    adapter_key = adapter_dir.path
    adapter_id = _ADAPTER_IDS.get(adapter_key)
    if adapter_id is None:
        adapter_id = _NEXT_ADAPTER_ID
        _ADAPTER_IDS[adapter_key] = adapter_id
        _NEXT_ADAPTER_ID += 1

    # The adapter changes every round; the engine does not.
    adapter_path = await adapter_dir.download()

    sampling = SamplingParams(
        n=num_generations,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_completion_length,
        # logprobs=0 returns the log-prob of each *sampled* token (and no top-k
        # alternatives). We only need the sampled path to compute the importance
        # ratio in the learner.
        logprobs=0,
    )

    prompts = [r.prompt for r in requests]
    outputs = _ENGINE.generate(
        prompts,
        sampling,
        # Unique int id per distinct adapter path (see _ADAPTER_IDS), so vLLM never
        # serves a stale cached adapter — the correctness fix that replaces the old
        # "round_id must be fresh" contract.
        lora_request=LoRARequest(f"policy-{adapter_id}", adapter_id, adapter_path),
    )

    samples: list[RolloutSample] = []
    for req, out in zip(requests, outputs):
        for comp in out.outputs:
            # comp.logprobs is a list (one entry per generated token) of
            # {token_id: Logprob}. Sum the sampled token's logprob at each step to
            # get the sequence log-prob under the rollout policy.
            total_lp = 0.0
            if comp.logprobs:
                for tok_id, lp_map in zip(comp.token_ids, comp.logprobs):
                    lp = lp_map.get(tok_id)
                    if lp is not None:
                        total_lp += lp.logprob
            samples.append(
                RolloutSample(
                    index=req.index,
                    completion=comp.text,
                    logprob=total_lp,
                    num_tokens=len(comp.token_ids),
                )
            )

    log.info(
        f"[rollout {_WORKER_ID}] round {round_id}: {len(requests)} prompts "
        f"-> {len(samples)} completions"
    )
    return samples
