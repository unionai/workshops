"""Measure the served model from Flyte tasks: torch on `mps` behind the app, and MLX natively, side by side.

    flyte run benchmark.py benchmark                 # hit the app with concurrent requests
    flyte run mlx_native.py mlx_generate             # the same prompts through mlx-lm on the Mac's GPU
    flyte run benchmark.py compare                   # both, one report

`benchmark` is an ordinary CPU task: it stays in its container and reaches the app over the devbox
network, the way any other service would. `mlx_generate` (mlx_native.py) asks for gpu="AppleSilicon"
and runs on the Mac, where MLX (Apple's array framework) is available; it is the Mac-native way to run
an LLM, and the comparison shows what that buys over PyTorch's mps backend. It lives in its own module
because its host environment is built from its own pip list (mlx-lm only): importing this module's
`serve` there would fail.
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

import flyte
import flyte.report

from config import MLX_MODEL_NAME, MODEL_NAME, PROPAGATED, reach
from mlx_native import PROMPTS, mlx_env, mlx_generate
from report_helpers import make_bar_chart, wrap_report
from serve import app_env

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


client_env = flyte.TaskEnvironment(
    name="apple-silicon-llm-client",
    image=flyte.Image.from_debian_base().with_pip_packages("httpx", "python-dotenv", "fastapi", "uvicorn", "torch>=2.4", "transformers>=4.51"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    env_vars=PROPAGATED,
    # `compare` launches tasks in both of these from inside its own container.
    depends_on=[app_env, mlx_env],
)

def _stat(value, label: str) -> str:
    return f"<div class='stat'><div class='value'>{value}</div><div class='label'>{label}</div></div>"


@client_env.task(report=True)
async def benchmark(prompts: int = 8, concurrency: int = 4, max_tokens: int = 128) -> dict:
    """Send `prompts` chat requests to the app, `concurrency` at a time; report tokens/s and latency."""
    base = reach(app_env)
    log.info(f"app endpoint: {base}")
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=600) as http:
        health = (await http.get(f"{base}/health")).json()
        log.info(f"health: {health}")

        async def one(i: int, prompt: str):
            async with sem:
                t = time.perf_counter()
                r = await http.post(f"{base}/v1/chat/completions", json={"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens})
                r.raise_for_status()
                body = r.json()
                latency = time.perf_counter() - t
                rows.append({"i": i, "prompt": prompt, "completion_tokens": body["usage"]["completion_tokens"], "latency_s": round(latency, 2), "gen_tokens_per_s": body["timing"]["tokens_per_second"], "reply": body["choices"][0]["message"]["content"]})

        start = time.perf_counter()
        await asyncio.gather(*(one(i, PROMPTS[i % len(PROMPTS)]) for i in range(prompts)))
        wall = time.perf_counter() - start

    rows.sort(key=lambda r: r["i"])
    total_tokens = sum(r["completion_tokens"] for r in rows)
    result = {
        "backend": f"torch {health.get('device')} (app)",
        "model": MODEL_NAME,
        "requests": prompts,
        "concurrency": concurrency,
        "wall_seconds": round(wall, 1),
        "aggregate_tokens_per_s": round(total_tokens / wall, 1),
        "mean_gen_tokens_per_s": round(sum(r["gen_tokens_per_s"] for r in rows) / len(rows), 1),
        "mean_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
        "p95_latency_s": sorted(r["latency_s"] for r in rows)[int(0.95 * (len(rows) - 1))],
    }
    log.info(result)
    table = "".join(f"<tr><td>{r['prompt']}</td><td>{r['completion_tokens']}</td><td>{r['latency_s']}s</td><td>{r['gen_tokens_per_s']}</td><td>{r['reply'][:160]}</td></tr>" for r in rows)
    body = (
        f"<h2>{MODEL_NAME} behind the app, {health.get('device')}</h2><div class='stats'>"
        + _stat(result["aggregate_tokens_per_s"], "tokens/s, aggregate")
        + _stat(result["mean_gen_tokens_per_s"], "tokens/s, per request while generating")
        + _stat(f"{result['mean_latency_s']}s", f"mean latency ({concurrency} concurrent)")
        + _stat(f"{result['p95_latency_s']}s", "p95 latency")
        + "</div><div class='card'>The server holds one lock around the GPU, so concurrent requests queue: per-request generation speed "
        "stays flat while latency grows with concurrency. That is the trade a single-model, single-GPU box makes.</div>"
        "<table><tr><th>Prompt</th><th>Tokens</th><th>Latency</th><th>tok/s</th><th>Reply</th></tr>" + table + "</table>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return result


@client_env.task(report=True)
async def compare(prompts: int = 8, concurrency: int = 4, max_tokens: int = 128) -> dict:
    """Run both benchmarks and put them side by side."""
    torch_result = await benchmark(prompts=prompts, concurrency=concurrency, max_tokens=max_tokens)
    mlx_result = await mlx_generate(prompts=prompts, max_tokens=max_tokens)
    labels = [torch_result["backend"], mlx_result["backend"]]
    body = (
        "<h2>PyTorch on mps vs MLX, same Mac, same prompts</h2><div class='stats'>"
        + _stat(torch_result["mean_gen_tokens_per_s"], f"tok/s, {MODEL_NAME} on torch mps")
        + _stat(mlx_result["mean_gen_tokens_per_s"], f"tok/s, {MLX_MODEL_NAME} on MLX")
        + _stat(f"{mlx_result['mean_gen_tokens_per_s'] / max(torch_result['mean_gen_tokens_per_s'], 0.1):.1f}x", "MLX speedup")
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"tokens/s while generating": [torch_result["mean_gen_tokens_per_s"], mlx_result["mean_gen_tokens_per_s"]]}, title="Generation speed", colors=["#5a7db5", "#0f3460"])
        + "</div><div class='card'>Two different things are being compared on purpose: a bf16 PyTorch model behind an HTTP server "
        "(portable to any CUDA GPU with one env var) and a 4-bit MLX model called in-process (Mac only). MLX is what you reach for "
        "when the Mac is the deployment target; torch is what you reach for when it is the dev box for a cluster.</div>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return {"torch": torch_result, "mlx": mlx_result}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(compare)
    print(run.url)
    run.wait()
    print(run.outputs())
