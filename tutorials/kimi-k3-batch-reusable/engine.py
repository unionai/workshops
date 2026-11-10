"""Batch inference against an OpenAI-compatible endpoint: concurrency, retries, timing, tokens, scoring.

One function does the work: `run_rows` sends every row of a shard through the endpoint `concurrency` at a time,
parses and scores each reply with the task module, and records what each request cost (input, output and
prefix-cached tokens; seconds; attempts). A progress callback lets the Flyte task that owns the shard re-render its
report while the batch runs. Nothing here knows about Flyte, clinical notes or receipts: it is the same file in the
text and the vision tutorials.

Why the client retries when Flyte already retries the task: a Flyte retry restarts the shard from row one, which is
the right remedy for a dead pod and the wrong one for a single 503 from an autoscaling endpoint. The client absorbs
the transient failures row by row; the task retry stays for the pod.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Awaitable, Callable

from config import DEFAULT_EFFORT, MODEL_ID, PRICE_IN, PRICE_OUT, api_key, endpoint_for, extra_body

log = logging.getLogger(__name__)

ATTEMPTS = 4  # per request, with exponential backoff and jitter


def client(timeout: float = 900):
    from openai import AsyncOpenAI

    base = endpoint_for()
    log.info(f"endpoint: {base}")
    return AsyncOpenAI(base_url=base, api_key=api_key(), timeout=timeout, max_retries=0)


def cost(input_tokens: int, output_tokens: int) -> float:
    """Dollars at the configured list prices; 0 when no price is configured."""
    return (input_tokens * PRICE_IN + output_tokens * PRICE_OUT) / 1e6


async def one(oai, task, row: dict, effort: str, max_tokens: int, temperature: float | None = None, n: int = 1) -> dict:
    """One row through the endpoint: reply text, reasoning length, tokens (incl. prefix-cache hits), seconds, attempts."""
    t0 = time.time()
    last_error = ""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            kwargs = dict(model=MODEL_ID, messages=task.messages(row), max_tokens=max_tokens, extra_body=extra_body(effort), n=n)
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = await oai.chat.completions.create(**kwargs)
            choices = resp.choices
            texts = [c.message.content or "" for c in choices]
            reasoning = sum(len(getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None) or "") for c in choices)
            usage = resp.usage
            details = getattr(usage, "prompt_tokens_details", None) if usage else None
            cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
            end = time.time()
            return {
                "id": row["id"], "text": texts[0], "texts": texts, "reasoning_chars": reasoning,
                "input_tokens": usage.prompt_tokens if usage else 0, "output_tokens": usage.completion_tokens if usage else 0,
                "cached_tokens": cached, "seconds": round(end - t0, 3), "started_at": t0, "finished_at": end,
                "attempts": attempt, "error": "", "finish_reason": choices[0].finish_reason or "",
            }
        except Exception as e:  # noqa: BLE001 - any transport or server error: back off and try again
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < ATTEMPTS:
                await asyncio.sleep(min(30, 2 ** attempt + random.random()))
    end = time.time()
    return {
        "id": row["id"], "text": "", "texts": [], "reasoning_chars": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
        "seconds": round(end - t0, 3), "started_at": t0, "finished_at": end, "attempts": ATTEMPTS, "error": last_error, "finish_reason": "error",
    }


def scored(task, row: dict, result: dict) -> dict:
    """Attach the parsed prediction and its score to a result."""
    pred = task.parse(result["text"]) if result["text"] else None
    s = task.score(pred, row["gold"])
    return {**result, "pred": pred, **{k: v for k, v in s.items() if k not in ("fields", "scores")},
            "fields": json.dumps(s.get("fields", {})), "scores": json.dumps(s.get("scores", {}))}


async def run_rows(
    task,
    rows: list[dict],
    effort: str = DEFAULT_EFFORT,
    concurrency: int = 8,
    max_tokens: int | None = None,
    on_progress: Callable[[list[dict]], Awaitable[None]] | None = None,
    every: int | None = None,
) -> list[dict]:
    """Every row through the endpoint, `concurrency` in flight, scored; results in input order.

    `on_progress(done_so_far)` is awaited every `every` completions (default: 5% of the shard) and once at the end.
    """
    oai = client()
    sem = asyncio.Semaphore(concurrency)
    max_tokens = max_tokens or getattr(task, "MAX_TOKENS", 512)
    every = every or max(1, len(rows) // 20)
    done: list[dict] = []
    lock = asyncio.Lock()

    async def worker(row):
        async with sem:
            r = scored(task, row, await one(oai, task, row, effort, max_tokens))
        async with lock:
            done.append(r)
            if on_progress and (len(done) % every == 0):
                await on_progress(list(done))
        return r

    results = await asyncio.gather(*(worker(r) for r in rows))
    if on_progress:
        await on_progress(list(done))
    return list(results)


def summarize(results: list[dict], wall_seconds: float | None = None) -> dict:
    """The numbers a report puts in its stat strip."""
    n = len(results)
    ok = [r for r in results if not r["error"]]
    wall = wall_seconds if wall_seconds is not None else (max(r["finished_at"] for r in results) - min(r["started_at"] for r in results) if results else 0)
    in_tok = sum(r["input_tokens"] for r in results)
    out_tok = sum(r["output_tokens"] for r in results)
    return {
        "rows": n, "errors": n - len(ok),
        "valid": sum(1 for r in results if r.get("valid")), "correct": sum(1 for r in results if r.get("correct")),
        "accuracy": round(100 * sum(1 for r in results if r.get("correct")) / max(n, 1), 1),
        "field_accuracy": round(100 * sum(r.get("field_accuracy", 0) for r in results) / max(n, 1), 1),
        "valid_rate": round(100 * sum(1 for r in results if r.get("valid")) / max(n, 1), 1),
        "wall_seconds": round(wall, 1), "rows_per_min": round(60 * n / max(wall, 1e-6), 1),
        "input_tokens": in_tok, "output_tokens": out_tok, "cached_tokens": sum(r["cached_tokens"] for r in results),
        "tokens_per_s": round(out_tok / max(wall, 1e-6), 1),
        "mean_seconds": round(sum(r["seconds"] for r in results) / max(n, 1), 2),
        "p50_seconds": round(sorted(r["seconds"] for r in results)[n // 2], 2) if n else 0,
        "p95_seconds": round(sorted(r["seconds"] for r in results)[min(n - 1, int(n * 0.95))], 2) if n else 0,
        "mean_output_tokens": round(out_tok / max(n, 1), 1), "mean_input_tokens": round(in_tok / max(n, 1), 1),
        "truncated": sum(1 for r in results if r.get("finish_reason") == "length"),
        "retried": sum(1 for r in results if r["attempts"] > 1),
        "cost_usd": round(cost(in_tok, out_tok), 4),
    }
