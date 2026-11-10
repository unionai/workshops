"""Batch inference on a warm pool of long-running GPU containers, and what happens when one of them dies.

    flyte run workflow.py batch                         # 200 clinical notes, chunks of 25, through the warm pool
    flyte run workflow.py batch                         # again: the pool is still warm, so no model load this time
    flyte run workflow.py batch --crash_chunk 3         # chunk 3 kills its replica on the first try; watch the run finish anyway

There is no model server. `generate` is a task in a reusable environment (`config.worker_env`): Flyte keeps REPLICAS
containers alive between calls, each loads the model once into GPU memory, and every call hands it a whole chunk of
rows as one vLLM batch. `batch` is an ordinary CPU task that splits the rows, calls `generate` once per chunk, and
scores what comes back. If a replica dies mid-chunk, the call is retried (retries=3) on a live replica, the pool
starts a replacement, and the report shows which replica did what.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import socket
import time

import flyte
import flyte.io
import flyte.report

import engine
import extract as task
from config import (
    DEFAULT_EFFORT, FAMILY, GPU, KEEP_SPECIAL_TOKENS, LLM_KWARGS, MAX_MODEL_LEN, MODEL, MODEL_ID, MODEL_URI, REPLICAS,
    chat_template_kwargs, final_answer,
    driver_env, worker_env,
)
from report_helpers import make_bar_chart, make_line_chart, stat, wrap_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# The worker: one engine per container, kept for the container's life
# ------------------------------------------------------------------

_ENGINE: dict = {}  # module globals survive between task calls in a reusable container
_LOCK = asyncio.Lock()
_SERVED = {"chunks": 0}


async def _download_top_level(uri: str) -> str:
    """Download the files at the top of a model directory and skip its subfolders. Some repos ship extra copies of the
    weights in subfolders (gpt-oss: `original/`, `metal/`), which vLLM never reads; that is half of gpt-oss-120b."""
    import tempfile

    import flyte.storage

    local = tempfile.mkdtemp(prefix="model-")
    files = [f async for f in flyte.io.Dir.from_existing_remote(uri).walk(recursive=False) if isinstance(f, flyte.io.File)]
    await asyncio.gather(*(flyte.storage.get(f.path, os.path.join(local, os.path.basename(f.path))) for f in files))
    return local


async def _engine(model: str, uri: str):
    """Load the model once per container. Later calls on this replica get the same engine back."""
    key = (model, uri)
    async with _LOCK:
        if key in _ENGINE:
            return _ENGINE[key], False
        t0 = time.time()
        path = await _download_top_level(uri) if uri else model
        t_dl = time.time()
        from vllm import LLM

        # LLM() blocks for the whole load; off the event loop so the replica's heartbeat keeps running.
        llm = await asyncio.to_thread(LLM, model=path, served_model_name=MODEL_ID, max_model_len=MAX_MODEL_LEN, **LLM_KWARGS)
        _ENGINE.clear()
        _ENGINE[key] = {"llm": llm, "download_seconds": round(t_dl - t0, 1), "load_seconds": round(time.time() - t0, 1),
                        "loaded_at": time.time(), "load_started": t0}
        log.info(f"engine ready on {socket.gethostname()} in {_ENGINE[key]['load_seconds']}s")
        return _ENGINE[key], True


@worker_env.task(retries=3)
async def generate(rows: list[dict], chunk: int, effort: str = DEFAULT_EFFORT, max_tokens: int = task.MAX_TOKENS,
                   model_uri: str = MODEL_URI, crash_marker: str = "") -> list[dict]:
    """One chunk through the warm engine as a single vLLM batch.

    `crash_marker` is an object-store path. If it does not exist yet, this call writes it and kills its own replica,
    as a revoked node or an OOM kill would; the retry finds the marker and runs normally. (An attempt counter would not
    do: a reusable container's FLYTE_ATTEMPT_NUMBER is set when the container starts, not per task.)
    """
    eng, fresh = await _engine(MODEL, model_uri)
    replica = socket.gethostname()
    crashed_before = False
    if crash_marker:
        import flyte.storage

        crashed_before = await flyte.storage.exists(crash_marker)
        if not crashed_before:
            await flyte.storage.put_stream(json.dumps({"replica": replica, "t": time.time()}).encode(), to_path=crash_marker)
            log.warning(f"chunk {chunk}: killing replica {replica} on purpose")
            logging.shutdown()
            os._exit(137)  # the container's process dies mid-call, as on an OOM kill or a lost node
    attempt = 1 if crashed_before else 0

    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, skip_special_tokens=not KEEP_SPECIAL_TOKENS)
    t0 = time.time()
    outs = await asyncio.to_thread(
        eng["llm"].chat, [task.messages(r) for r in rows], params, chat_template_kwargs=chat_template_kwargs(effort), use_tqdm=False
    )
    t1 = time.time()
    _SERVED["chunks"] += 1
    meta = {"replica": replica, "chunk": chunk, "attempt": attempt, "fresh_engine": fresh, "chunks_on_replica": _SERVED["chunks"],
            "load_seconds": eng["load_seconds"], "download_seconds": eng["download_seconds"],
            "load_started": eng["load_started"], "loaded_at": eng["loaded_at"]}
    results = []
    for row, o in zip(rows, outs):
        c = o.outputs[0]
        answer, reasoning_chars = final_answer(c.text)
        results.append({
            "id": row["id"], "text": answer, "texts": [answer], "reasoning_chars": reasoning_chars,
            "input_tokens": len(o.prompt_token_ids or []), "output_tokens": len(c.token_ids or []),
            "cached_tokens": int(getattr(o, "num_cached_tokens", 0) or 0),
            "seconds": round(t1 - t0, 3), "started_at": t0, "finished_at": t1, "attempts": attempt + 1, "error": "",
            "finish_reason": c.finish_reason or "", **meta,
        })
    log.info(f"chunk {chunk}: {len(rows)} rows in {t1 - t0:.1f}s on {replica} (engine {'loaded now' if fresh else 'warm'})")
    return results


# ------------------------------------------------------------------
# The driver: split, fan out, score, report
# ------------------------------------------------------------------


@driver_env.task(cache="auto")
async def prepare(n: int = 200, seed: int = 42) -> list[dict]:
    return task.load(n, seed)


def _short(host: str) -> str:
    return host.rsplit("-", 1)[-1][:8] if host else "?"


INK, MUTED, GRID = "#1a1a2e", "#6c757d", "#e9ecef"
C_CHUNK, C_RETRY, C_LOAD, C_KILL = "#5a7db5", "#e07a1f", "#b8c0cc", "#d62839"


def _lane_chart(results: list[dict], origin: float, t_end: float, crash: dict | None) -> str:
    """One lane per replica: its model load, every chunk it ran (retries in orange), and the moment a replica was killed."""
    lanes: dict[str, dict] = {}
    for r in sorted(results, key=lambda x: x["started_at"]):
        lane = lanes.setdefault(r["replica"], {"chunks": {}, "load": None})
        lane["chunks"].setdefault(r["chunk"], r)
        if r["fresh_engine"] and lane["load"] is None:
            lane["load"] = (max(r["load_started"], origin) - origin, r["loaded_at"] - origin)
    if crash and crash.get("replica") and crash["replica"] not in lanes:
        lanes[crash["replica"]] = {"chunks": {}, "load": None}
    if not lanes:
        return ""
    t_max = max(t_end, 1.0)
    width, ml, mr, mt, lane_h = 900, 96, 16, 30, 30
    height = mt + lane_h * len(lanes) + 46
    cw = width - ml - mr
    sx = lambda t: ml + max(0.0, min(t, t_max)) / t_max * cw  # noqa: E731
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
           f'<text x="{ml}" y="18" font-size="13" font-weight="600" fill="{INK}">Who did what, when (seconds since the batch started)</text>']
    for i in range(6):
        t = t_max * i / 5
        out.append(f'<line x1="{sx(t):.1f}" y1="{mt}" x2="{sx(t):.1f}" y2="{mt + lane_h * len(lanes)}" stroke="{GRID}"/>')
        out.append(f'<text x="{sx(t):.1f}" y="{mt + lane_h * len(lanes) + 14}" text-anchor="middle" font-size="11" fill="{MUTED}">{t:.0f}s</text>')
    for li, (rep, lane) in enumerate(lanes.items()):
        y = mt + li * lane_h + 5
        h = lane_h - 10
        out.append(f'<text x="{ml - 8}" y="{y + h / 2 + 4:.1f}" text-anchor="end" font-size="11" fill="{INK}" font-family="ui-monospace,monospace">{html.escape(_short(rep))}</text>')
        if lane["load"]:
            a, b = lane["load"]
            out.append(f'<rect x="{sx(a):.1f}" y="{y}" width="{max(sx(b) - sx(a), 2):.1f}" height="{h}" rx="3" fill="{C_LOAD}"><title>model load {b - a:.0f}s</title></rect>')
            if sx(b) - sx(a) > 60:
                out.append(f'<text x="{(sx(a) + sx(b)) / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle" font-size="10" fill="{INK}">model load {b - a:.0f}s</text>')
        for c, r in lane["chunks"].items():
            a, b = r["started_at"] - origin, r["finished_at"] - origin
            col = C_RETRY if r["attempt"] > 0 else C_CHUNK
            out.append(f'<rect x="{sx(a):.1f}" y="{y}" width="{max(sx(b) - sx(a), 1.5):.1f}" height="{h}" fill="{col}" stroke="#fff" stroke-width="0.5">'
                       f'<title>chunk {c}{" (retry)" if r["attempt"] > 0 else ""}: {b - a:.1f}s</title></rect>')
        if crash and crash.get("replica") == rep and crash.get("t"):
            x = sx(crash["t"] - origin)
            out.append(f'<line x1="{x:.1f}" y1="{y - 2}" x2="{x:.1f}" y2="{y + h + 2}" stroke="{C_KILL}" stroke-width="3"/>')
            left = x > width - mr - 50  # near the right edge, put the label on the left of the marker
            out.append(f'<text x="{x - 5 if left else x + 5:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="{"end" if left else "start"}" '
                       f'font-size="11" font-weight="600" fill="{C_KILL}">killed</text>')
    ly = mt + lane_h * len(lanes) + 34
    lx = ml
    for label, col in (("model load", C_LOAD), ("chunk", C_CHUNK), ("retried chunk", C_RETRY), ("replica killed", C_KILL)):
        out.append(f'<rect x="{lx}" y="{ly - 9}" width="12" height="10" rx="2" fill="{col}"/><text x="{lx + 17}" y="{ly}" font-size="11" fill="{INK}">{label}</text>')
        lx += 30 + 7 * len(label)
    out.append("</svg>")
    return "".join(out)


def _throughput(results: list[dict], origin: float, t_end: float) -> str:
    """Rows finished per minute over time: the dip when a replica dies and the recovery when one returns."""
    if len(results) < 2:
        return ""
    bucket = max(5.0, round(t_end / 40))
    n = int(t_end // bucket) + 1
    counts = [0] * n
    for r in results:
        counts[min(int((r["finished_at"] - origin) // bucket), n - 1)] += 1
    first = next((i for i, c in enumerate(counts) if c), 0)  # start the chart where work starts, after the model load
    data = [{"t": int(i * bucket), "rows per minute": int(round(c * 60 / bucket))} for i, c in enumerate(counts) if i >= first]
    return make_line_chart(data, "t", ["rows per minute"], title=f"Rows finished per minute ({bucket:.0f} s buckets)",
                           x_label="seconds since the batch started", width=900, height=260)


def _report(results: list[dict], s: dict, chunks_total: int, origin: float, final: bool, crash_chunk: int,
            crash: dict | None = None, rows_by_id: dict | None = None) -> str:
    done = {r["chunk"] for r in results}
    pct = 100 * len(done) / max(chunks_total, 1)
    t_end = max([r["finished_at"] - origin for r in results] + [time.time() - origin if not final else 0, 1.0])
    replicas = {}
    for r in results:
        rep = replicas.setdefault(r["replica"], {"chunks": set(), "rows": 0, "load": None})
        rep["chunks"].add(r["chunk"])
        rep["rows"] += 1
        if r["fresh_engine"] and r["loaded_at"] >= origin - 60:
            rep["load"] = r["load_seconds"]
    loads = [v["load"] for v in replicas.values() if v["load"]]
    retried = {r["chunk"]: r for r in results if r["attempt"] > 0}

    head = (f"<h2>Batch on a warm GPU pool · {html.escape(MODEL_ID)} · {REPLICAS} × {html.escape(GPU)}</h2>"
            f"<div style='background:{GRID};border-radius:5px;height:10px'><div style='width:{pct:.0f}%;background:#0f3460;height:10px;border-radius:5px'></div></div>"
            f"<p>{len(done)} / {chunks_total} chunks · {len(results)} rows{'' if final else ' · running'}"
            f"{f' · chunk {crash_chunk} kills its replica on the first try' if crash_chunk >= 0 else ''}</p>")
    stats = "<div class='stat-grid'>" + "".join([
        stat(f"{s.get('rows', 0):,}", "rows scored"),
        stat(s.get("rows_per_min", "–"), "rows / min"),
        stat(s.get("tokens_per_s", "–"), "output tokens / s"),
        stat(f"{s.get('field_accuracy', '–')}%", "mean field score"),
        stat(len(replicas), "replicas that did work"),
        stat(f"{len(loads)}" + (f" · {max(loads):.0f}s" if loads else ""), "model loads this run"),
        stat(len(retried), "chunks retried"),
    ]) + "</div>"

    retry_rows = ""
    for c, r in sorted(retried.items()):
        killed = crash.get("replica") if crash and str(c) == str(crash_chunk) else None
        lost = f"{r['finished_at'] - crash['t']:.0f}s" if killed and crash.get("t") else "–"
        retry_rows += (f"<tr><td>{c}</td><td><code>{html.escape(_short(killed or '?'))}</code></td>"
                       f"<td><code>{html.escape(_short(r['replica']))}</code></td><td>{'no, already warm' if not r['fresh_engine'] else 'yes'}</td>"
                       f"<td>{lost}</td></tr>")
    retries = (f"<h3>Retried chunks</h3><table><tr><th>chunk</th><th>replica that died</th><th>finished by</th>"
               f"<th>needed a model load?</th><th>killed → done</th></tr>{retry_rows}</table>") if retry_rows else ""

    rep_rows = "".join(
        f"<tr><td><code>{html.escape(_short(rep))}</code></td><td>{len(v['chunks'])}</td><td>{v['rows']:,}</td>"
        f"<td>{'yes, ' + format(v['load'], '.0f') + ' s' if v['load'] else 'no, already warm'}</td></tr>"
        for rep, v in sorted(replicas.items(), key=lambda kv: -kv[1]["rows"]))
    if crash and crash.get("replica") and crash["replica"] not in replicas:
        rep_rows += f"<tr><td><code>{html.escape(_short(crash['replica']))}</code></td><td>0</td><td>0</td><td>killed by chunk {crash_chunk}</td></tr>"
    table = ("<h3>Replicas</h3><table><tr><th>replica</th><th>chunks</th><th>rows</th><th>loaded the model in this run</th></tr>"
             f"{rep_rows}</table>")

    fields = ""
    if final and s.get("per_field"):
        cats = list(s["per_field"])
        fields = make_bar_chart(cats, {"mean score %": [s["per_field"][c] for c in cats]}, title="Score per field",
                                colors=[C_CHUNK], width=900, height=280, y_max_cap=100)
    samples = ""
    if final and rows_by_id:
        picks = [r for r in results if r.get("valid")][:3]
        samples = ("<h3>Sample rows</h3><table><tr><th>note</th><th>gold</th><th>model output</th></tr>"
                   + "".join("<tr>" + task.preview(rows_by_id[r["id"]], r) + "</tr>" for r in picks) + "</table>")
    note = ("<p style='color:#6c757d'>Every chunk is one call to <code>generate</code> on a warm replica that keeps the model in GPU "
            "memory between calls. When a replica dies, its chunk is retried on whichever replica is free first, and the "
            "pool starts a replacement; finished chunks are never redone.</p>")
    return wrap_report(head + stats + _lane_chart(results, origin, t_end, crash) + _throughput(results, origin, t_end)
                       + retries + table + fields + samples + note)


async def _read_crash(marker: str) -> dict | None:
    """The crash marker the killed replica wrote: which replica it was and when."""
    if not marker:
        return None
    try:
        import flyte.storage

        if not await flyte.storage.exists(marker):
            return None
        local = await flyte.storage.get(marker)
        return json.loads(open(local).read())
    except Exception:  # noqa: BLE001 - the report is better without the marker than not at all
        return None


@driver_env.task(report=True)
async def batch(n: int = 200, chunk_size: int = 25, effort: str = DEFAULT_EFFORT, max_tokens: int = task.MAX_TOKENS,
                model_uri: str = MODEL_URI, crash_chunk: int = -1, seed: int = 42) -> dict:
    """Split the rows into chunks, send each chunk to the warm pool, score everything, report replica by replica."""
    rows = await prepare(n, seed)
    by_id = {r["id"]: r for r in rows}
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    origin = time.time()
    results: list[dict] = []
    await flyte.report.replace.aio(_report([], {}, len(chunks), origin, False, crash_chunk), do_flush=True)

    marker = flyte.ctx().raw_data_path.get_random_remote_path(f"crash-chunk-{crash_chunk}") if crash_chunk >= 0 else ""

    last = {"t": 0.0}

    async def one(i: int, part: list[dict]):
        out = await generate(part, i, effort, max_tokens, model_uri, marker if i == crash_chunk else "")
        results.extend(engine.scored(task, by_id[r["id"]], r) for r in out)
        if time.time() - last["t"] > 5:  # at most one live refresh every 5 s, however many chunks finish
            last["t"] = time.time()
            crash = await _read_crash(marker) if any(r["attempt"] > 0 for r in results) else None
            await flyte.report.replace.aio(_report(results, engine.summarize(results, wall_seconds=time.time() - origin),
                                                   len(chunks), origin, False, crash_chunk, crash), do_flush=True)

    await asyncio.gather(*(one(i, part) for i, part in enumerate(chunks)))
    wall = max(r["finished_at"] for r in results) - origin
    s = engine.summarize(results, wall_seconds=wall)
    per_field: dict[str, list[float]] = {}
    for r in results:
        for f, v in json.loads(r["scores"]).items():
            per_field.setdefault(f, []).append(v)
    s["per_field"] = {f: round(100 * sum(v) / len(v), 1) for f, v in per_field.items()}
    s.update({
        "model": MODEL_ID, "family": FAMILY.name, "gpu": GPU, "replicas_configured": REPLICAS, "chunk_size": chunk_size,
        "effort": effort, "crash_chunk": crash_chunk,
        "replicas_used": sorted({r["replica"] for r in results}),
        "model_loads_this_run": len({(r["replica"], r["loaded_at"]) for r in results if r["fresh_engine"] and r["loaded_at"] >= origin}),
        "load_seconds": sorted({r["load_seconds"] for r in results if r["fresh_engine"]}),
        "chunks_retried": sorted({r["chunk"] for r in results if r["attempt"] > 0}),
        # Who did the work: chunks per replica, and for each retried chunk the replica that finished it and when.
        "chunks_per_replica": {rep: len({r["chunk"] for r in results if r["replica"] == rep}) for rep in sorted({r["replica"] for r in results})},
        "retries": {str(c): {"replica": r["replica"], "fresh_engine": r["fresh_engine"],
                             "finished_s": round(r["finished_at"] - origin, 1)}
                    for c, r in {r["chunk"]: r for r in results if r["attempt"] > 0}.items()},
        "chunk_finish_s_by_replica": {rep: [round(r["finished_at"] - origin, 1) for r in sorted(
            {r["chunk"]: r for r in results if r["replica"] == rep}.values(), key=lambda x: x["finished_at"])]
            for rep in sorted({r["replica"] for r in results})},
    })
    log.info(json.dumps(s))
    crash = await _read_crash(marker)
    if crash:
        s["crash"] = {"replica": crash.get("replica"), "at_s": round(crash.get("t", origin) - origin, 1)}
    await flyte.report.replace.aio(_report(results, s, len(chunks), origin, True, crash_chunk, crash, by_id), do_flush=True)
    return s


if __name__ == "__main__":
    # A run is bound to one cluster pool, and a child task on another pool's queue is refused ("action crosses cluster
    # pool boundary"). On a separate GPU pool the whole run goes to its queue, so launch it with the queue on the run
    # context (the engineer's advice: not `flyte run --queue`, which has sent inputs to the default pool's store).
    import argparse

    from config import QUEUE

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--chunk_size", type=int, default=25)
    p.add_argument("--crash_chunk", type=int, default=-1)
    p.add_argument("--max_tokens", type=int, default=task.MAX_TOKENS, help="reasoning models need room before the answer")
    p.add_argument("--config", default=None, help="Flyte config file (defaults to the usual lookup)")
    a = p.parse_args()
    flyte.init_from_config(a.config)
    run = flyte.with_runcontext(queue=QUEUE or None).run(batch, n=a.n, chunk_size=a.chunk_size, crash_chunk=a.crash_chunk, max_tokens=a.max_tokens)
    print(run.name, run.url)
