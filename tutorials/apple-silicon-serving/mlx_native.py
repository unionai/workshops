"""Generate with mlx-lm, natively on the Mac's GPU, from a Flyte task.

    flyte run mlx_native.py mlx_generate

MLX is Apple's array framework; mlx-lm runs LLMs on it. Neither has Linux wheels, so the pip marker
keeps the task's Linux image buildable while the Mac environment the bridge builds gets the real
thing. This module imports nothing from `serve`: the host environment for this task only has mlx-lm.
"""

from __future__ import annotations

import logging
import time

import flyte
import flyte.report

from config import MLX_MODEL_NAME, PROPAGATED, SHAPES
from report_helpers import wrap_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROMPTS = [
    "Write a haiku about a laptop.",
    "Explain what a GPU does, in three sentences.",
    "List five uses for a paperclip.",
    "What is the capital of Australia, and why isn't it Sydney?",
    "Write a limerick about a cat who learned SQL.",
    "Give me a two-sentence summary of how photosynthesis works.",
    "Name three differences between TCP and UDP.",
    "Describe the taste of coffee to someone who has never had it.",
]

mlx_env = flyte.TaskEnvironment(
    name="apple-silicon-mlx",
    # mlx only has macOS arm64 wheels; the marker keeps the Linux image buildable.
    image=flyte.Image.from_debian_base().with_pip_packages("mlx-lm; sys_platform == 'darwin'", "python-dotenv"),
    resources=SHAPES["apple"],
    env_vars=PROPAGATED,
)


def _stat(value, label: str) -> str:
    return f"<div class='stat'><div class='value'>{value}</div><div class='label'>{label}</div></div>"


@mlx_env.task(report=True)
async def mlx_generate(prompts: int = 8, max_tokens: int = 128) -> dict:
    """Generate for the same prompts with mlx-lm, natively on the Mac's GPU. Sequential, one model load."""
    import mlx.core as mx
    from mlx_lm import generate, load

    model, tokenizer = load(MLX_MODEL_NAME)
    rows = []
    for i in range(prompts):
        prompt = PROMPTS[i % len(PROMPTS)]
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.perf_counter()
        reply = generate(model, tokenizer, prompt=text, max_tokens=max_tokens, verbose=False)
        mx.synchronize() if hasattr(mx, "synchronize") else None
        seconds = time.perf_counter() - t
        n = len(tokenizer.encode(reply))
        rows.append({"prompt": prompt, "completion_tokens": n, "latency_s": round(seconds, 2), "gen_tokens_per_s": round(n / seconds, 1), "reply": reply})
        log.info(f"{n} tokens in {seconds:.1f}s = {n / seconds:.0f} tok/s")
    total = sum(r["completion_tokens"] for r in rows)
    wall = sum(r["latency_s"] for r in rows)
    result = {
        "backend": "mlx (native)",
        "model": MLX_MODEL_NAME,
        "requests": prompts,
        "wall_seconds": round(wall, 1),
        "aggregate_tokens_per_s": round(total / wall, 1),
        "mean_gen_tokens_per_s": round(sum(r["gen_tokens_per_s"] for r in rows) / len(rows), 1),
        "mean_latency_s": round(wall / len(rows), 2),
        "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2) if hasattr(mx, "get_peak_memory") else None,
    }
    table = "".join(f"<tr><td>{r['prompt']}</td><td>{r['completion_tokens']}</td><td>{r['latency_s']}s</td><td>{r['gen_tokens_per_s']}</td><td>{r['reply'][:160]}</td></tr>" for r in rows)
    body = (
        f"<h2>{MLX_MODEL_NAME} with mlx-lm, natively on the Mac</h2><div class='stats'>"
        + _stat(result["mean_gen_tokens_per_s"], "tokens/s while generating")
        + _stat(f"{result['mean_latency_s']}s", "mean latency, sequential")
        + _stat(result["peak_memory_gb"], "peak GB")
        + "</div><table><tr><th>Prompt</th><th>Tokens</th><th>Latency</th><th>tok/s</th><th>Reply</th></tr>" + table + "</table>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(mlx_generate)
    print(run.url)
    run.wait()
    print(run.outputs())
