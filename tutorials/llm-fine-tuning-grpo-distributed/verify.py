"""The verifier pool — sandboxed test execution on warm, reusable containers.

This is the Level 1 change in one file. In the single-GPU tutorial the reward
function is a serial `for` loop that pushes one completion at a time through one
in-process sandbox on the training GPU. At batch_size=8 x num_generations=8 that
is 64 sequential sandbox executions per step, and verification — not the gradient —
dominates wall-clock.

Here each shard of completions becomes a task call against a warm pool sized
`replicas x concurrency` (up to 160 concurrent shards with the defaults in
config.py). The reward logic is unchanged; only *where* it runs is different.


Why userns and not bubblewrap
-----------------------------
The single-GPU tutorial runs `backend="bubblewrap"`, the strongest backend. We
can't here, and the reason is worth understanding rather than working around:

  * bwrap needs CAP_SYS_ADMIN + unconfined AppArmor on the pod.
  * The only way to request those is `flyte.PodTemplate().allow_nested_sandboxing()`.
  * A reusable TaskEnvironment cannot set `pod_template` at all — Flyte raises
    `ValueError("Cannot set pod_template when environment is reusable.")`.

So "warm pool" and "bubblewrap" are mutually exclusive today. `backend="userns"`
(userns-lite) needs no extra pod capabilities, which is exactly why it works here.
It is *weaker isolation* — it relies on unprivileged user namespaces without the
seccomp/AppArmor posture bwrap gets. Combined with `network_mode="blocked"` it is
a reasonable choice for scoring model-generated solutions to MBPP problems.

It is NOT the right default for genuinely adversarial code. If you need bwrap-grade
isolation, drop `reusable` from `verify_env` and add the pod template back:

    verify_env = flyte.TaskEnvironment(
        name="grpo-dist-verify",
        resources=flyte.Resources(cpu=2, memory="4Gi"),
        pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
    )                       # no `reusable=` — these two cannot coexist

...then fan out with `flyte.map(verify_shard, shards)`. You keep the parallelism and
the strong sandbox, and you pay container cold-start on every shard instead of
amortizing it across the pool. That is the actual trade: isolation strength vs.
cold start. Pick deliberately.
"""

import logging
import os
import uuid
from dataclasses import dataclass

from common import assemble_code, extract_test_list, run_tests_sandboxed
from config import verify_env

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# Module-global, so it is created once per *process*. Because verify_env is
# reusable, that process outlives a single task call — which makes this a cheap
# way to prove reuse is actually engaging. Across many shards you should see
# roughly `replicas` distinct worker ids, not one per call. If every shard reports
# a unique id, the pool is cold-starting and you are getting none of the benefit.
_WORKER_ID = f"{os.getpid()}-{uuid.uuid4().hex[:6]}"
_SHARDS_SERVED = 0
_BACKEND: str | None = None

# Explicit override, e.g. GRPO_SANDBOX_BACKEND=bubblewrap when you have swapped
# verify_env to a non-reusable environment with allow_nested_sandboxing().
_BACKEND_OVERRIDE = os.getenv("GRPO_SANDBOX_BACKEND")


def resolve_backend() -> str:
    """Pick the sandbox backend for this pod, loudly.

    The sandbox library deliberately does no auto-detection: an unavailable
    backend fails rather than silently downgrading to weaker isolation. We keep
    that property and add only what local development needs:

      * an explicit override always wins, and errors if it isn't actually usable;
      * `userns` is the pool default (see the module docstring for why not bwrap);
      * `sandbox-exec` is accepted on macOS so `flyte run --local` works on a
        laptop, with a warning that it is not the deployed configuration.

    What it will NOT do is fall through to `backend="none"`. Running
    model-generated code with no isolation at all is not a degraded mode of this
    pipeline, it is a different and much worse thing — so we refuse instead.
    """
    from union import sandbox as sb

    available = dict(sb.on_device.available_backends())
    if not available:
        raise RuntimeError(
            "union.sandbox reports no backends (native extension unavailable). "
            "Refusing to execute model-generated code unsandboxed."
        )

    if _BACKEND_OVERRIDE:
        if not available.get(_BACKEND_OVERRIDE, False):
            raise RuntimeError(
                f"GRPO_SANDBOX_BACKEND={_BACKEND_OVERRIDE!r} is not available here. "
                f"Available: {sorted(k for k, v in available.items() if v)}"
            )
        return _BACKEND_OVERRIDE

    if available.get("userns"):
        return "userns"

    if available.get("sandbox-exec"):
        log.warning(
            "[verify] userns unavailable (expected on macOS) — falling back to "
            "sandbox-exec for local development. This is NOT the configuration "
            "that runs on the cluster."
        )
        return "sandbox-exec"

    raise RuntimeError(
        "No usable sandbox backend. Available: "
        f"{sorted(k for k, v in available.items() if v)}. Refusing to run "
        "model-generated code without isolation — set GRPO_SANDBOX_BACKEND "
        "explicitly if you know what you are doing."
    )


@dataclass
class VerifyItem:
    """One completion to score."""
    func_def: str     # the function signature, from the last line of the MBPP prompt
    completion: str   # raw model output
    setup: str        # optional setup code (e.g. `import math`)
    tests: str        # newline-joined assert statements


@dataclass
class VerifyResult:
    """Binary reward plus the partial-credit detail, for reporting only.

    The reward is all-or-nothing on purpose. Partial credit (passed/total) is
    hackable on this task: a constant like `return True` grabs a fraction of the
    asserts and, in a group where the genuine attempts all score 0, becomes the
    highest-advantage completion — so GRPO reinforces degenerate constants.
    `passed`/`total` are carried for the report, never for the gradient.
    """
    reward: float
    passed: int
    total: int
    worker_id: str


@verify_env.task
async def verify_shard(items: list[VerifyItem], timeout_s: float = 5.0) -> list[VerifyResult]:
    """Score a shard of completions in a sandbox. Runs on the warm pool.

    Must be `async`: ReusePolicy(concurrency > 1) is only supported for async
    tasks, and concurrency is where most of the pool's throughput comes from.

    A fresh session is opened per shard rather than cached in a module global.
    Caching one would save the session setup, but every shard on a replica would
    then share a sandbox work dir — code from one completion could observe or
    clobber files written by another, which is precisely the property the sandbox
    exists to provide. Shard-level isolation is worth the setup cost; raise
    `shard_size` if you want to amortize it further.
    """
    global _SHARDS_SERVED, _BACKEND

    from union import sandbox as sb

    if _BACKEND is None:
        _BACKEND = resolve_backend()
        log.info(f"[verify {_WORKER_ID}] sandbox backend: {_BACKEND}")

    _SHARDS_SERVED += 1
    log.info(f"[verify {_WORKER_ID}] shard #{_SHARDS_SERVED}, {len(items)} completions")

    results: list[VerifyResult] = []
    async with sb.on_device.session(network_mode="blocked", backend=_BACKEND) as sbx:
        for it in items:
            test_list = extract_test_list(it.tests)
            if not it.completion.strip() or not test_list:
                results.append(VerifyResult(0.0, 0, len(test_list), _WORKER_ID))
                continue
            code = assemble_code(it.func_def, it.completion, it.setup)
            try:
                all_passed, passed, total = await run_tests_sandboxed(
                    sbx, code, test_list, timeout_s=timeout_s
                )
            except Exception as e:
                # A sandbox failure is a failed completion, not a failed training
                # step. Never let one bad generation take down the run.
                log.debug(f"[verify {_WORKER_ID}] sandbox error: {e}")
                all_passed, passed, total = False, 0, len(test_list)
            results.append(
                VerifyResult(1.0 if all_passed else 0.0, passed, total, _WORKER_ID)
            )

    return results


def chunk(items: list, size: int) -> list[list]:
    """Split a list into shards of at most `size`."""
    size = max(1, size)
    return [items[i:i + size] for i in range(0, len(items), size)]
