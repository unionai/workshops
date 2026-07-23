"""Smoke test for the verifier pool — run this before spending any GPU time.

It answers the three questions that decide whether the rest of the pipeline can
work at all, and none of them need a GPU:

  1. Does the sandbox come up on the cluster, and on which backend?
     (`userns` on Linux — bubblewrap is unavailable to a reusable environment.)
  2. Does it score code correctly, and is the network actually blocked?
  3. Is the pool actually being *reused*, or is every shard cold-starting?

Question 3 is the one worth running remotely. Locally every call lands in one
process so reuse looks perfect no matter what; only on the cluster does the
distinct-worker count mean anything.

Usage:
    flyte run smoke_test.py pool_smoke_test
    flyte run smoke_test.py pool_smoke_test --shards 40
"""

import asyncio
import json
import logging

import flyte

from config import verify_env
from verify import VerifyItem, verify_shard

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


SIG = "def add_two(a, b):"
TESTS = "assert add_two(1, 2) == 3\nassert add_two(5, 5) == 10"

CASES = [
    ("correct",        "def add_two(a, b):\n    return a + b\n",                    1.0, 2),
    ("wrong",          "def add_two(a, b):\n    return a * b\n",                    0.0, 0),
    ("raises",         "def add_two(a, b):\n    raise RuntimeError('boom')\n",      0.0, 0),
    ("partial",        "def add_two(a, b):\n    return 3 if (a,b)==(1,2) else 0\n", 0.0, 1),
    ("empty",          "",                                                          0.0, 0),
    ("fenced",         "```python\ndef add_two(a, b):\n    return a + b\n```",       1.0, 2),
    ("body-only",      "    return a + b\n",                                         1.0, 2),
    ("infinite-loop",  "def add_two(a, b):\n    while True:\n        pass\n",        0.0, 0),
]

NET_PROBE = (
    "def add_two(a, b):\n"
    "    import urllib.request\n"
    "    urllib.request.urlopen('http://example.com', timeout=3)\n"
    "    return a + b\n"
)


# The driver runs on verify_env itself — same lean image, nothing else to build.
# A reusable environment needs >= 2 replicas precisely so a parent task holding
# one slot cannot starve the children it is waiting on.
@verify_env.task(report=True)
async def pool_smoke_test(shards: int = 24) -> str:
    """Fan out `shards` identical shards and check scoring, isolation, and reuse."""
    # -- 1 & 2: correctness + isolation --
    items = [VerifyItem(SIG, code, "", TESTS) for _, code, _, _ in CASES]
    results = await verify_shard(items)

    failures = []
    for (name, _, want_reward, want_passed), got in zip(CASES, results):
        ok = got.reward == want_reward and got.passed == want_passed
        log.info(
            f"  {name:14} reward={got.reward:.1f} passed={got.passed}/{got.total} "
            f"{'' if ok else f'<-- EXPECTED reward={want_reward} passed={want_passed}'}"
        )
        if not ok:
            failures.append(f"{name}: reward={got.reward} passed={got.passed}")

    net = (await verify_shard([VerifyItem(SIG, NET_PROBE, "", TESTS)]))[0]
    log.info(f"  {'network-probe':14} reward={net.reward:.1f} (must be 0.0 — egress blocked)")
    if net.reward != 0.0:
        failures.append("SECURITY: sandboxed code reached the network")

    backend_worker = results[0].worker_id

    # -- 3: is the pool warm? --
    # Fire many small shards at once. Capacity is replicas x concurrency, so with
    # reuse working these land on a handful of long-lived processes.
    probe = [VerifyItem(SIG, CASES[0][1], "", TESTS)]
    fanned = await asyncio.gather(*(verify_shard(probe) for _ in range(shards)))
    workers = [r[0].worker_id for r in fanned]
    distinct = sorted(set(workers))

    log.info(f"\n{shards} shards -> {len(distinct)} distinct workers")
    for w in distinct:
        log.info(f"  {w}: {workers.count(w)} shards")

    reuse_ratio = shards / max(1, len(distinct))
    verdict = (
        "REUSE WORKING" if len(distinct) < shards
        else "NO REUSE — every shard cold-started"
    )
    if len(distinct) >= shards:
        failures.append("pool is not reusing containers")

    summary = {
        "scoring_failures": failures,
        "shards": shards,
        "distinct_workers": len(distinct),
        "shards_per_worker": round(reuse_ratio, 1),
        "verdict": verdict,
        "sample_worker_id": backend_worker,
    }

    await flyte.report.replace.aio(
        f"<h2>Verifier Pool Smoke Test</h2>"
        f"<p><b>{verdict}</b> — {shards} shards across {len(distinct)} workers "
        f"({reuse_ratio:.1f} shards/worker)</p>"
        f"<p>Scoring failures: {failures or 'none'}</p>"
        f"<pre>{json.dumps(summary, indent=2)}</pre>",
        do_flush=True,
    )

    if failures:
        raise RuntimeError(f"smoke test failed: {failures}")

    log.info(f"\n{verdict} — all checks passed")
    return json.dumps(summary)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pool_smoke_test)
    print(run.url)
