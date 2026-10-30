"""Measure the endpoint from Flyte tasks: does it answer correctly, and how fast?

    flyte run benchmark.py sanity                      # 12 questions with known answers, scored
    flyte run benchmark.py benchmark                   # concurrent requests: tokens/s and latency
    flyte run benchmark.py benchmark --thinking        # the same, with Qwen's thinking on
    flyte run benchmark.py compare                     # thinking off vs on, one report

These are ordinary CPU tasks: they reach the app over the network the way any client would. The
point of `sanity` is the habit, not the questions: an endpoint is not "up" until something has
scored its answers, and the score and the cost (tokens, seconds) belong on the same row.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import flyte
import flyte.report

from config import APP_NAME, MODEL_ID, PROPAGATED, SECRETS, api_key, endpoint_for
from report_helpers import make_bar_chart, wrap_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

client_env = flyte.TaskEnvironment(
    name="qwen35-client",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("openai>=1.50", "python-dotenv"),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    env_vars=PROPAGATED,
    secrets=SECRETS,
)

PROMPTS = [
    "Explain what a Kubernetes pod is in three sentences.",
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Give me five names for a coffee shop run by cats.",
    "What is the difference between a process and a thread?",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Convert 72 degrees Fahrenheit to Celsius and show the formula.",
    "List three trade-offs of microservices versus a monolith.",
    "Write a haiku about a GPU that is out of memory.",
]

# Gold answers a regex can check. Not a benchmark: a smoke test with a number attached.
QUESTIONS: list[tuple[str, str]] = [
    ("What is 17 * 23? Reply with the number only.", "391"),
    ("What is the capital of Australia? Reply with the city only.", "Canberra"),
    ("How many days are in a leap year? Reply with the number only.", "366"),
    ("What is the chemical symbol for gold? Reply with the symbol only.", "Au"),
    ("What is 2 to the power of 10? Reply with the number only.", "1024"),
    ("Which planet is closest to the Sun? Reply with the name only.", "Mercury"),
    ("What is the square root of 144? Reply with the number only.", "12"),
    ("In what year did the first moon landing happen? Reply with the year only.", "1969"),
    ("How many bits are in a byte? Reply with the number only.", "8"),
    ("What is the boiling point of water in Celsius at sea level? Reply with the number only.", "100"),
    ("What is the largest ocean on Earth? Reply with the name only.", "Pacific"),
    ("What is 15% of 200? Reply with the number only.", "30"),
]


def _client():
    from openai import AsyncOpenAI

    base = endpoint_for(APP_NAME)
    log.info(f"endpoint: {base}")
    return AsyncOpenAI(base_url=base, api_key=api_key(), timeout=600, max_retries=1)


def _thinking(on: bool) -> dict:
    return {"chat_template_kwargs": {"enable_thinking": on}}


async def _chat(client, prompt: str, max_tokens: int, thinking: bool) -> dict:
    """One streamed request; returns the text plus time-to-first-token, tokens and duration."""
    t0 = time.perf_counter()
    first = None
    text, reasoning = "", ""
    usage = None
    stream = await client.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
        stream=True, stream_options={"include_usage": True}, extra_body=_thinking(thinking),
    )
    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            first = first or time.perf_counter()
            text += delta.content
        r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if r:
            first = first or time.perf_counter()
            reasoning += r
    end = time.perf_counter()
    out_tokens = usage.completion_tokens if usage else 0
    return {
        "prompt": prompt, "text": text, "reasoning_chars": len(reasoning),
        "ttft": round((first or end) - t0, 2), "seconds": round(end - t0, 2),
        "output_tokens": out_tokens, "input_tokens": usage.prompt_tokens if usage else 0,
        "tokens_per_s": round(out_tokens / max(end - (first or t0), 1e-6), 1),
    }


def _stat(value, label: str) -> str:
    return f"<div class='stat'><div class='value'>{value}</div><div class='label'>{label}</div></div>"


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", s.lower())


def is_correct(answer: str, gold: str) -> bool:
    """Numbers: any number in the answer equal to the gold. Text: the gold appears in the answer."""
    try:
        target = float(gold)
    except ValueError:
        return _normalize(gold) in _normalize(answer)
    for found in re.findall(r"-?\d[\d,]*\.?\d*", answer):
        try:
            if abs(float(found.replace(",", "")) - target) < 1e-6:
                return True
        except ValueError:
            continue
    return False


@client_env.task(report=True)
async def sanity(thinking: bool = False, max_tokens: int = 64, concurrency: int = 4) -> dict:
    """Ask the endpoint 12 questions with known answers; score them; report accuracy next to cost."""
    client = _client()
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str):
        async with sem:
            return await _chat(client, q, max_tokens, thinking)

    results = await asyncio.gather(*(one(q) for q, _ in QUESTIONS))
    for r, (_, gold) in zip(results, QUESTIONS):
        r["gold"], r["correct"] = gold, is_correct(r["text"], gold)
    correct = sum(r["correct"] for r in results)
    summary = {
        "model": MODEL_ID, "thinking": thinking, "questions": len(results), "correct": correct,
        "accuracy": round(correct / len(results), 3),
        "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2),
        "mean_output_tokens": round(sum(r["output_tokens"] for r in results) / len(results), 1),
    }
    log.info(json.dumps(summary))
    rows = "".join(
        f"<tr><td>{'✅' if r['correct'] else '❌'}</td><td>{r['prompt']}</td><td>{r['gold']}</td>"
        f"<td>{(r['text'] or '(empty)')[:120]}</td><td>{r['output_tokens']}</td><td>{r['seconds']}s</td></tr>"
        for r in results
    )
    await flyte.report.replace.aio(
        wrap_report(
            f"<h2>Sanity check: {MODEL_ID}{' (thinking)' if thinking else ''}</h2><div class='stat-grid'>"
            + _stat(f"{summary['accuracy']:.0%}", f"correct ({correct}/{len(results)})")
            + _stat(summary["mean_seconds"], "seconds per question")
            + _stat(summary["mean_output_tokens"], "output tokens per question")
            + "</div><table><tr><th></th><th>Question</th><th>Gold</th><th>Answer</th><th>Tokens</th><th>Time</th></tr>"
            + rows + "</table>"
            "<div class='note'>Two numbers per row: whether it was right, and what it cost. Keep them together; "
            "an endpoint that is right and slow and one that is fast and wrong are different problems.</div>"
        ),
        do_flush=True,
    )
    return summary


@client_env.task(report=True)
async def benchmark(prompts: int = 8, concurrency: int = 4, max_tokens: int = 256, thinking: bool = False) -> dict:
    """Send `prompts` chat requests, `concurrency` at a time; report tokens/s, time to first token, latency."""
    client = _client()
    sem = asyncio.Semaphore(concurrency)
    todo = (PROMPTS * ((prompts + len(PROMPTS) - 1) // len(PROMPTS)))[:prompts]

    async def one(p: str):
        async with sem:
            return await _chat(client, p, max_tokens, thinking)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(p) for p in todo))
    wall = time.perf_counter() - t0
    total_out = sum(r["output_tokens"] for r in results)
    summary = {
        "model": MODEL_ID, "thinking": thinking, "requests": len(results), "concurrency": concurrency,
        "wall_seconds": round(wall, 1),
        "throughput_tokens_per_s": round(total_out / wall, 1),
        "per_request_tokens_per_s": round(sum(r["tokens_per_s"] for r in results) / len(results), 1),
        "mean_ttft": round(sum(r["ttft"] for r in results) / len(results), 2),
        "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 2),
        "mean_output_tokens": round(total_out / len(results), 1),
    }
    log.info(json.dumps(summary))
    labels = [f"#{i + 1}" for i in range(len(results))]
    body = (
        f"<h2>{MODEL_ID}{' (thinking)' if thinking else ''}: {len(results)} requests, {concurrency} concurrent</h2><div class='stat-grid'>"
        + _stat(summary["throughput_tokens_per_s"], "tokens/s, aggregate")
        + _stat(summary["per_request_tokens_per_s"], "tokens/s, per request")
        + _stat(f"{summary['mean_ttft']}s", "time to first token")
        + _stat(f"{summary['mean_seconds']}s", "per request")
        + _stat(summary["mean_output_tokens"], "output tokens")
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"tokens/s": [r["tokens_per_s"] for r in results]}, title="Generation speed per request", colors=["#5a7db5"])
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"TTFT": [r["ttft"] for r in results], "total": [r["seconds"] for r in results]}, title="Seconds per request", colors=["#06d6a0", "#0f3460"])
        + "</div><h3>Replies</h3>"
        + "".join(f"<div class='card'><b>{r['prompt']}</b><br><pre style='white-space:pre-wrap'>{r['text'][:600]}</pre>"
                  f"<small>{r['output_tokens']} tokens · {r['seconds']}s" + (f" · {r['reasoning_chars']} chars of thinking" if r["reasoning_chars"] else "") + "</small></div>" for r in results[:4])
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return summary


@client_env.task(report=True)
async def compare(prompts: int = 8, concurrency: int = 4, max_tokens: int = 512) -> dict:
    """Thinking off vs on: the same prompts, side by side. Thinking buys quality on hard tasks and costs tokens on easy ones."""
    off = await benchmark(prompts, concurrency, max_tokens, False)
    on = await benchmark(prompts, concurrency, max_tokens, True)
    off_s, on_s = await sanity(False, 64, concurrency), await sanity(True, max_tokens, concurrency)
    labels = ["thinking off", "thinking on"]
    body = (
        f"<h2>{MODEL_ID}: thinking off vs on</h2><div class='chart-container'>"
        + make_bar_chart(labels, {"accuracy %": [off_s["accuracy"] * 100, on_s["accuracy"] * 100]}, title="Sanity questions correct (%)", colors=["#0f3460"], y_max_cap=100)
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"output tokens": [off["mean_output_tokens"], on["mean_output_tokens"]]}, title="Output tokens per request (benchmark prompts)", colors=["#5a7db5"])
        + "</div><div class='chart-container'>"
        + make_bar_chart(labels, {"seconds": [off["mean_seconds"], on["mean_seconds"]]}, title="Seconds per request", colors=["#06d6a0"])
        + "</div><div class='note'>Thinking is a per-request switch (<code>chat_template_kwargs.enable_thinking</code>), so the same "
        "endpoint serves both. Measure before you default to it: on short factual questions it mostly costs tokens.</div>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return {"thinking_off": {**off, "sanity": off_s}, "thinking_on": {**on, "sanity": on_s}}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(sanity)
    print(run.url)
    run.wait()
    print(run.outputs())
