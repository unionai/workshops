# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
#    "kubernetes",
# ]
# main = "batch"
# ///

"""Batch inference on a market for whole MI355X nodes that survives preemption.

The market sells whole 8-GPU nodes at a market price. When the price goes
above your limit, the node is taken back, and when it drops you get one again.

- The driver runs on a CPU node and splits the prompts into shards.
- Each shard runs on a reusable GPU environment where one replica is one full
  node. The replica starts a vLLM server once, and its weights stream from
  object storage straight into GPU memory.
- Shards are cached, and every sub-batch inside a shard is a trace, so a retry
  after a preemption replays the finished sub-batches.
- Replicas scale from one node up to `MAX_NODES`. With more granted nodes
  the run goes faster, and with none it waits in the queue.

gpt-oss-120b runs as eight engines per node (`--tensor-parallel 1`). A model
that needs the whole node, such as Kimi K3, uses `--tensor-parallel 8`.

    uv run batch_inference.py --model <prefetched model dir> --num-prompts 2000
"""

import argparse
import asyncio
import html
import json
import os
import random
import socket
import subprocess
import sys
import time
from contextvars import ContextVar

import flyte
import flyte.report
from flyte.io import Dir, File

from common import (
    GPUS_PER_NODE,
    QUEUE,
    full_node_pod_template,
    prebuild_and_reexec,
    with_registry,
)

VLLM_ROCM_BASE = (
    "docker.io/vllm/vllm-openai-rocm:nightly-rocm100-96710245ece8fe57a0df3ea18cdc79cbb8ce93cf"
)
SERVER_PORT = 8000
SERVED_NAME = "model"
SUB_BATCH = 128

# {{docs-fragment images}}
vllm_image_spec = (
    with_registry(flyte.Image.from_base(VLLM_ROCM_BASE), "spot-vllm-rocm")
    # The vLLM ROCm image ships Python 3.12; without this the build targets the local interpreter.
    .clone(python_version=(3, 12), extendable=True).with_pip_packages(
        "flyteplugins-vllm>=2.13.1", "unionai-reuse>=0.1.11", "openai", "async-lru"
    )
)
driver_image_spec = with_registry(
    flyte.Image.from_debian_base(python_version=(3, 12)), "spot-batch-driver"
).with_pip_packages("kubernetes")  # common.py builds the GPU pod template when it is imported, here too

# `__main__` builds the images without a queue and passes their URIs in, see common.prebuild_and_reexec.
#
# Use the public worker image from docker/ (VLLM_IMAGE=...) rather than the spec above: the stock vLLM image starts
# `vllm serve` as its ENTRYPOINT, which takes the reusable worker's launcher for a model name and exits with code 2.
# Flyte's image builder cannot clear an entrypoint, and drops a `command` set in the pod template.
vllm_image = os.getenv("VLLM_IMAGE") or vllm_image_spec
driver_image = os.getenv("DRIVER_IMAGE") or driver_image_spec
# {{/docs-fragment images}}

# {{docs-fragment envs}}
max_nodes = int(os.getenv("MAX_NODES", "2"))

worker_env = flyte.TaskEnvironment(
    name="spot-vllm-worker",
    image=vllm_image,
    resources=flyte.Resources(cpu="32", memory="256Gi", shm="64Gi"),
    pod_template=full_node_pod_template(),
    queue=QUEUE,
    interruptible=True,
    # One replica per granted node. Several shards share a replica so the
    # server always has requests queued.
    reusable=flyte.ReusePolicy(
        replicas=(1, max_nodes), concurrency=4, idle_ttl=900, scaledown_ttl=300
    ),
)

driver_env = flyte.TaskEnvironment(
    name="spot-batch-driver",
    image=driver_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    queue=QUEUE,
    # The driver re-imports this module to submit shards; give it the same URIs.
    env_vars={k: v for k in ("VLLM_IMAGE", "DRIVER_IMAGE") if (v := os.getenv(k))},
    depends_on=[worker_env],
)
# {{/docs-fragment envs}}


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

TASKS = [
    "parses an ISO 8601 duration string into seconds",
    "merges overlapping intervals in a list",
    "finds the longest palindromic substring",
    "implements an LRU cache with O(1) get and put",
    "validates a Sudoku board",
    "serializes and deserializes a binary tree",
    "computes the edit distance between two strings",
    "detects a cycle in a directed graph",
    "rate limits calls with a token bucket",
    "evaluates a reverse Polish notation expression",
    "groups anagrams from a list of words",
    "returns the k most frequent elements",
    "topologically sorts tasks with dependencies",
    "finds the median of a data stream",
    "decodes a run-length encoded string",
    "checks whether a string has balanced brackets",
]
LANGUAGES = ["Python", "TypeScript", "Go", "Rust"]
STYLES = [
    "Include unit tests.",
    "Explain the time and space complexity.",
    "Handle invalid input explicitly.",
    "Keep it under 40 lines.",
]


def make_prompts(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [
        {
            "id": i,
            "prompt": f"Write a {rng.choice(LANGUAGES)} function that {rng.choice(TASKS)}. {rng.choice(STYLES)}",
        }
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# vLLM server, one per replica
# ─────────────────────────────────────────────────────────────────────────────

_server: dict = {}
_server_lock = asyncio.Lock()


async def ensure_server(model_path: str, tensor_parallel: int, extra_args: str) -> float:
    """Start vLLM once per replica and return how long the load took."""
    async with _server_lock:
        if _server.get("proc") is not None and _server["proc"].poll() is None:
            return _server["load_seconds"]

        import httpx
        from flyteplugins.vllm.model_streamer import engine_args

        start = time.time()
        args = await engine_args(model_path)
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args["model"],
            "--served-model-name",
            SERVED_NAME,
            "--load-format",
            args["load_format"],
            "--model-loader-extra-config",
            json.dumps(args["model_loader_extra_config"]),
            "--tensor-parallel-size",
            str(tensor_parallel),
            "--data-parallel-size",
            str(GPUS_PER_NODE // tensor_parallel),
            "--port",
            str(SERVER_PORT),
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.92",
            *extra_args.split(),
        ]
        print("starting vLLM:", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        async with httpx.AsyncClient() as client:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM exited with code {proc.returncode} while loading"
                    )
                try:
                    if (
                        await client.get(f"http://localhost:{SERVER_PORT}/health", timeout=5)
                    ).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(5)

        _server.update(proc=proc, load_seconds=time.time() - start)
        print(f"vLLM ready in {_server['load_seconds']:.0f}s", flush=True)
        return _server["load_seconds"]


# Counts the sub-batches a shard invocation actually generates. A replayed trace
# skips the function body, so replays don't add to it. A context variable keeps
# the count per invocation, since several shards share a reusable replica.
_generated: ContextVar[list[int] | None] = ContextVar("_generated", default=None)


# {{docs-fragment generate}}
@flyte.trace
async def generate(prompts: list[dict], max_tokens: int) -> list[dict]:
    """One sub-batch. Traced, so a retried shard replays it instead of rerunning it."""
    from openai import AsyncOpenAI

    counter = _generated.get()
    if counter is not None:
        counter.append(len(prompts))

    client = AsyncOpenAI(base_url=f"http://localhost:{SERVER_PORT}/v1", api_key="none")

    async def one(p: dict) -> dict:
        resp = await client.chat.completions.create(
            model=SERVED_NAME,
            messages=[{"role": "user", "content": p["prompt"]}],
            max_tokens=max_tokens,
        )
        return {
            "id": p["id"],
            "completion": resp.choices[0].message.content,
            "tokens": resp.usage.completion_tokens,
        }

    return list(await asyncio.gather(*(one(p) for p in prompts)))


# {{/docs-fragment generate}}


def gpu_view() -> dict:
    """What this replica was allocated: its node, its GPUs and how full their memory is, read from the kernel."""
    import glob

    def _read(p: str) -> str:
        try:
            return open(p).read().strip()
        except OSError:
            return ""

    used, total = [], []
    for card in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
        t, u = _read(card), _read(card.replace("vram_total", "vram_used"))
        if t.isdigit() and u.isdigit():
            total.append(int(t) / 2**30)
            used.append(int(u) / 2**30)
    return {"node": os.getenv("NODE_NAME") or "?", "gpus": len(total),
            "vram_used_gib": round(sum(used), 1), "vram_total_gib": round(sum(total), 1)}


async def count_attempt() -> int:
    """Count this shard's attempts in the run's storage.

    `flyte.ctx().attempt_number` comes from the process environment, which a
    reusable replica shares across invocations, so each shard keeps its own
    counter under its action name, which stays the same across retries.
    """
    import flyte.storage as storage

    ctx = flyte.ctx()
    path = f"{ctx.run_base_dir.rstrip('/')}/attempts/{ctx.action.name}"
    count = 0
    if await storage.exists(path):
        count = int(b"".join([chunk async for chunk in storage.get_stream(path)]))
    count += 1
    if "://" not in path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    await storage.put_stream(str(count).encode(), to_path=path)
    return count


# {{docs-fragment shard}}
@worker_env.task(cache="auto", retries=10)
async def infer_shard(
    model: Dir,
    shard: File,
    tensor_parallel: int = 1,
    max_tokens: int = 1024,
    extra_args: str = "",
    crash: bool = False,
) -> File:
    attempt = await count_attempt()
    load_seconds = await ensure_server(model.path, tensor_parallel, extra_args)
    shard_started = time.time()

    async with shard.open("rb") as f:
        prompts = json.loads(bytes(await f.read()))  # File.open().read() returns obstore Bytes
    batches = [prompts[i : i + SUB_BATCH] for i in range(0, len(prompts), SUB_BATCH)]

    generated: list[int] = []
    _generated.set(generated)
    results: list[dict] = []
    for i, batch in enumerate(batches):
        results.extend(await generate(batch, max_tokens))
        # Demo switch: on the first attempt, kill the replica halfway through.
        # The retry lands on a replica (this one restarted, or another node),
        # replays the finished sub-batches from their traces and carries on.
        if crash and attempt == 1 and i == (len(batches) - 1) // 2:
            print(f"crashing after sub-batch {i}", flush=True)
            os._exit(137)

    summary = {
        "load_seconds": load_seconds,
        "replica": socket.gethostname(),
        "attempt": attempt,
        "batches": len(batches),
        "replayed": len(batches) - len(generated),
        "started_at": shard_started,
        **gpu_view(),
        "results": results,
    }
    out = File.new_remote()
    async with out.open("wb") as f:
        await f.write(json.dumps(summary).encode())
    return out


# {{/docs-fragment shard}}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────


# A replica counts as active if it finished a shard within this window.
ACTIVE_WINDOW_S = 120


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th style='text-align:left;padding:4px 12px'>{h}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td style='padding:4px 12px'>{html.escape(c)}</td>" for c in row)
        + "</tr>"
        for row in rows
    )
    return f"<table style='border-collapse:collapse'><tr>{head}</tr>{body}</table>"


# Optional: what a granted node costs per hour, for the cost estimate in the report.
PRICE_PER_NODE_HOUR = float(os.getenv("PRICE_PER_NODE_HOUR", "0") or 0)


def _allocated(now: float, shards: dict[int, dict]) -> str:
    """The compute the market granted, as the shards saw it: one row per node, plus node-hours and cost."""
    nodes: dict[str, dict] = {}
    for s in shards.values():
        began = s.get("started_at", s["finished_at"]) - (s["load_seconds"] if s["attempt"] >= 1 else 0)  # loading is billed time too
        n = nodes.setdefault(s.get("node", "?"), {"replicas": set(), "shards": 0, "first": began,
                                                  "last": 0.0, "gpus": 0, "used": 0.0, "total": 0.0})
        n["replicas"].add(s["replica"])
        n["shards"] += 1
        n["first"] = min(n["first"], began)
        if s["finished_at"] >= n["last"]:  # the most recent shard's view of GPU memory
            n.update(last=s["finished_at"], gpus=s.get("gpus", 0), used=s.get("vram_used_gib", 0.0), total=s.get("vram_total_gib", 0.0))
    if not nodes:
        return ""
    active = {k: v for k, v in nodes.items() if now - v["last"] <= ACTIVE_WINDOW_S}
    node_hours = sum((v["last"] - v["first"]) / 3600 for v in nodes.values())
    cards = [
        ("Nodes in use", f"{len(active)} of {len(nodes)} seen"),
        ("GPUs allocated", f"{sum(v['gpus'] for v in active.values())} now · {sum(v['gpus'] for v in nodes.values())} seen"),
        ("GPU memory in use", f"{sum(v['used'] for v in active.values()):,.0f} / {sum(v['total'] for v in active.values()):,.0f} GiB"
         if active else f"last seen {sum(v['used'] for v in nodes.values()):,.0f} / {sum(v['total'] for v in nodes.values()):,.0f} GiB"),
        ("Node-hours used", f"{node_hours:.2f}"),
    ]
    if PRICE_PER_NODE_HOUR:
        cards.append(("Est. cost", f"${node_hours * PRICE_PER_NODE_HOUR:,.2f}"))
    items = "".join(
        f"<div style='padding:12px 16px;border:1px solid #ddd;border-radius:8px'>"
        f"<div style='color:#666;font-size:12px'>{html.escape(k)}</div>"
        f"<div style='font-size:22px'>{html.escape(v)}</div></div>"
        for k, v in cards
    )
    rows = [
        [node, ", ".join(sorted(v["replicas"]))[:60], str(v["gpus"]), f"{v['used']:,.0f} / {v['total']:,.0f} GiB", str(v["shards"]),
         f"{(now - v['first']) / 60:.1f} min ago", "active" if node in active else f"quiet {(now - v['last']) / 60:.1f} min"]
        for node, v in sorted(nodes.items())
    ]
    return ("<h3>Allocated compute</h3>"
            f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{items}</div>"
            "<p style='color:#666'>One row per granted node, as the shards saw it. A node that stops reporting has most "
            "likely been taken back by the market; node-hours count from a node's first shard to its last.</p>"
            + _table(["Node", "Replica", "GPUs", "GPU memory", "Shards", "First work", "Status"], rows))


def _render(total: int, start: float, shards: dict[int, dict]) -> str:
    """Render the live report from the summaries of the finished shards."""
    now = time.time()
    elapsed = max(now - start, 1e-6)
    tokens = sum(r["tokens"] or 0 for s in shards.values() for r in s["results"])
    recent_tokens = sum(
        r["tokens"] or 0
        for s in shards.values()
        if now - s["finished_at"] <= ACTIVE_WINDOW_S
        for r in s["results"]
    )
    prompts_done = sum(len(s["results"]) for s in shards.values())
    loads = sorted(s["load_seconds"] for s in shards.values())
    retried = {i: s for i, s in shards.items() if s["attempt"] > 1}

    replicas: dict[str, dict] = {}
    for s in shards.values():
        r = replicas.setdefault(s["replica"], {"shards": 0, "last": 0.0})
        r["shards"] += 1
        r["last"] = max(r["last"], s["finished_at"])
    active = sum(1 for r in replicas.values() if now - r["last"] <= ACTIVE_WINDOW_S)

    cards = [
        ("Shards done", f"{len(shards)} / {total}"),
        ("Prompts", f"{prompts_done:,}"),
        ("Tokens / s, overall", f"{tokens / elapsed:,.0f}"),
        ("Tokens / s, last 2 min", f"{recent_tokens / min(elapsed, ACTIVE_WINDOW_S):,.0f}"),
        ("Elapsed", f"{elapsed / 60:.1f} min"),
        ("Active replicas", str(active)),
        ("Retried shards", str(len(retried))),
        ("Batches replayed", str(sum(s["replayed"] for s in shards.values()))),
        ("Model load (median)", f"{loads[len(loads) // 2]:.0f} s" if loads else "-"),
    ]
    items = "".join(
        f"<div style='padding:12px 16px;border:1px solid #ddd;border-radius:8px'>"
        f"<div style='color:#666;font-size:12px'>{html.escape(k)}</div>"
        f"<div style='font-size:22px'>{html.escape(v)}</div></div>"
        for k, v in cards
    )
    out = (
        "<h2>Batch inference on a GPU market</h2>"
        f"<div style='display:flex;gap:12px;flex-wrap:wrap'>{items}</div>"
        f"<p style='color:#666'>A replica is active if it finished a shard in the last "
        f"{ACTIVE_WINDOW_S // 60} minutes.</p>"
    )
    out += _allocated(now, shards)
    if replicas:
        rows = [
            [name, str(r["shards"]), f"{(now - r['last']) / 60:.1f} min ago"]
            for name, r in sorted(replicas.items())
        ]
        out += "<h3>Replicas</h3>" + _table(["Replica", "Shards finished", "Last shard"], rows)
    if retried:
        rows = [
            [str(i), str(s["attempt"]), f"{s['replayed']} of {s['batches']}", s["replica"]]
            for i, s in sorted(retried.items())
        ]
        out += "<h3>Retried shards</h3>" + _table(
            ["Shard", "Attempts", "Batches replayed", "Finished on"], rows
        )
    return out


@driver_env.task(report=True)
async def batch(
    model: Dir,
    num_prompts: int = 2000,
    shard_size: int = 256,
    tensor_parallel: int = 1,
    max_tokens: int = 1024,
    extra_args: str = "",
    crash_shard: int = -1,
) -> Dir:
    prompts = make_prompts(num_prompts)
    shards = []
    for i in range(0, len(prompts), shard_size):
        f = File.new_remote()
        async with f.open("wb") as fh:
            await fh.write(json.dumps(prompts[i : i + shard_size]).encode())
        shards.append(f)

    start = time.time()
    finished: dict[int, dict] = {}
    outputs: list[File] = []

    async def run_one(i: int, shard: File) -> None:
        out = await infer_shard(
            model, shard, tensor_parallel, max_tokens, extra_args, crash=(i == crash_shard)
        )
        async with out.open("rb") as fh:
            summary = json.loads(bytes(await fh.read()))
        summary["finished_at"] = time.time()
        finished[i] = summary
        outputs.append(out)
        await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)

    async def refresh() -> None:
        # Keep "Active replicas" and the elapsed time moving while nothing finishes,
        # for example while every node is taken back.
        while True:
            await asyncio.sleep(30)
            await flyte.report.replace.aio(
                _render(len(shards), start, finished), do_flush=True
            )

    await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)
    refresher = asyncio.create_task(refresh())
    try:
        await asyncio.gather(*(run_one(i, s) for i, s in enumerate(shards)))
    finally:
        refresher.cancel()
    await flyte.report.replace.aio(_render(len(shards), start, finished), do_flush=True)

    local = "/tmp/results"
    os.makedirs(local, exist_ok=True)
    for j, out in enumerate(outputs):
        await out.download(os.path.join(local, f"shard-{j:05d}.json"))
    return await Dir.from_local(local)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Remote path of the prefetched model directory"
    )
    parser.add_argument("--num-prompts", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--extra-args", default="")
    parser.add_argument("--crash-shard", type=int, default=-1)
    a = parser.parse_args()

    prebuild_and_reexec(
        {"VLLM_IMAGE": vllm_image_spec, "DRIVER_IMAGE": driver_image_spec}
    )
    flyte.init_from_config()
    run = flyte.with_runcontext(queue=QUEUE).run(
        batch,
        model=Dir.from_existing_remote(a.model),
        num_prompts=a.num_prompts,
        shard_size=a.shard_size,
        tensor_parallel=a.tensor_parallel,
        max_tokens=a.max_tokens,
        extra_args=a.extra_args,
        crash_shard=a.crash_shard,
    )
    print(run.name, run.url)
