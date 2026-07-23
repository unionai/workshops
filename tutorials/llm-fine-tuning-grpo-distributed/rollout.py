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


# Process-global engine + the base-model path it was built for. Survives across
# task calls on a reusable replica; None on the first call to each replica.
_ENGINE = None
_ENGINE_MODEL_PATH: str | None = None
_BASE_PATH: str | None = None
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
        round_id: MUST increase every round. vLLM caches LoRA adapters by integer
            id — if you reuse an id, it serves the *previously cached* adapter and
            silently ignores the new weights on disk. Training then looks like it
            runs fine while every round rolls out the round-0 policy. This is the
            single nastiest failure mode in this design, and it fails silently.
    """
    global _ENGINE, _ENGINE_MODEL_PATH, _BASE_PATH

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Download the base weights once per replica, not once per round.
    if _BASE_PATH is None:
        _BASE_PATH = await base_dir.download()
        log.info(f"[rollout {_WORKER_ID}] base weights -> {_BASE_PATH}")

    if _ENGINE is None or _ENGINE_MODEL_PATH != _BASE_PATH:
        log.info(f"[rollout {_WORKER_ID}] building vLLM engine (cold start)")
        _ENGINE = LLM(
            model=_BASE_PATH,
            enable_lora=True,
            max_lora_rank=lora_rank,
            dtype="bfloat16",
            # Leave headroom: the adapter, CUDA graphs, and NCCL buffers all live
            # outside the fraction vLLM reserves for weights + KV cache.
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            enforce_eager=False,
        )
        _ENGINE_MODEL_PATH = _BASE_PATH
    else:
        log.info(f"[rollout {_WORKER_ID}] reusing warm engine for round {round_id}")

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
        # Fresh name AND fresh int id per round — see the round_id docstring above.
        lora_request=LoRARequest(f"policy-r{round_id}", round_id + 1, adapter_path),
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
