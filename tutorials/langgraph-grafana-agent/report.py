"""HTML for the Flyte task reports: the dataset, eval results, the decision, the bake-off.

Plain HTML and inline SVG, no charting library, so the reports render anywhere Flyte
renders them and the code bundle stays small.
"""

from __future__ import annotations

import html
import re

from tickets import CATEGORIES, DATASET_VERSIONS, Ticket

SERIES = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
OK = "#0a7a0a"
BAD = "#d03b3b"

CSS = f"""
<style>
  .rp {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {INK}; max-width: 1100px; }}
  .rp h2 {{ margin: 0 0 4px 0; font-size: 20px; }}
  .rp h3 {{ margin: 22px 0 8px 0; font-size: 15px; }}
  .rp .sub {{ color: {MUTED}; font-size: 13px; margin-bottom: 14px; }}
  .rp table {{ border-collapse: collapse; font-size: 13px; }}
  .rp th, .rp td {{ padding: 5px 10px; border-bottom: 1px solid {GRID}; text-align: left; vertical-align: top; }}
  .rp th {{ color: {MUTED}; font-weight: 600; }}
  .rp td.num, .rp th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .rp .bar {{ display: inline-block; height: 10px; background: {SERIES}; border-radius: 0 3px 3px 0; vertical-align: middle; }}
  .rp .card {{ border-left: 4px solid {SERIES}; padding: 8px 14px; margin: 10px 0; background: #f6f9fe; }}
  .rp .truth {{ border-left: 4px solid {MUTED}; padding: 8px 14px; margin: 10px 0; background: #f6f6f4; }}
  .rp .ok {{ color: {OK}; font-weight: 600; }}
  .rp .bad {{ color: {BAD}; font-weight: 600; }}
  .rp .stats span {{ display: inline-block; margin-right: 18px; font-size: 13px; }}
  .rp .stats b {{ font-size: 16px; font-variant-numeric: tabular-nums; }}
  .rp .ticket {{ font-size: 13px; padding: 4px 0; }}
  .rp .tag {{ display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px; border: 1px solid {GRID}; color: {MUTED}; margin-right: 8px; min-width: 96px; text-align: center; }}
  .rp .hero {{ border-radius: 10px; padding: 16px 20px; margin: 4px 0 14px 0; border: 1px solid {GRID}; border-left: 6px solid {SERIES}; background: #f6f9fe; }}
  .rp .hero.bad {{ border-left-color: {BAD}; background: #fdf6f6; }}
  .rp .eyebrow {{ font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: {MUTED}; }}
  .rp .big {{ font-size: 24px; font-weight: 650; margin: 4px 0 2px 0; }}
  .rp .dim {{ color: {MUTED}; font-weight: 400; }}
  .rp .chips {{ margin-top: 10px; }}
  .rp .chip {{ display: inline-block; font-size: 12px; padding: 3px 10px; border-radius: 999px; margin: 0 6px 6px 0; border: 1px solid {GRID}; background: #fff; }}
  .rp .chip.pass {{ color: {OK}; border-color: #bfe3bf; }}
  .rp .chip.fail {{ color: {BAD}; border-color: #f0c2c2; }}
  .rp .tiles {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 6px 0; }}
  .rp .tile {{ min-width: 110px; padding: 10px 14px; border: 1px solid {GRID}; border-radius: 8px; }}
  .rp .tile .v {{ font-size: 22px; font-weight: 650; font-variant-numeric: tabular-nums; }}
  .rp .tile .k {{ font-size: 12px; color: {MUTED}; }}
  .rp ol.steps {{ padding-left: 22px; font-size: 13px; }}
  .rp ol.steps li {{ margin: 6px 0; }}
  .rp ol.steps li.failed b {{ color: {BAD}; }}
  .rp ol.steps .out {{ color: {MUTED}; font-size: 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
</style>
"""


def _esc(s) -> str:
    return html.escape(str(s))


def _bar(frac: float, width: int = 160) -> str:
    return f'<span class="bar" style="width:{max(2, int(width * max(0.0, min(1.0, frac))))}px"></span>'


def dataset_html(train: list[Ticket], test: list[Ticket], sample: int = 12) -> str:
    counts = {c: sum(1 for t in train if t.label == c) for c in CATEGORIES}
    parts = [CSS, '<div class="rp"><h2>Synthetic support tickets</h2>']
    parts.append(
        f'<div class="sub">{len(train)} training tickets, {len(test)} held-out, {len(CATEGORIES)} categories, generated from templates with a fixed seed. No download, no API.</div>'
    )
    parts.append("<h3>Sample</h3>")
    for t in test[:sample]:
        parts.append(f'<div class="ticket"><span class="tag">{t.label}</span>{_esc(t.text)}</div>')
    parts.append("<h3>Training set by category</h3><table>")
    for c, n in counts.items():
        parts.append(f"<tr><td>{c}</td><td class=num>{n}</td><td>{_bar(n / max(counts.values()))}</td></tr>")
    parts.append("</table></div>")
    return "".join(parts)


def evals_html(results: list[dict], request=None) -> str:
    """The eval matrix. `results` are `factory.EvalResult` as dicts."""
    # Columns: every category any result was scored on, so a v1 model scored on v2 shows its
    # 0% on data_request instead of hiding it.
    cats = list(
        dict.fromkeys(c for v in DATASET_VERSIONS.values() for c in v if any(c in r["per_category"] for r in results))
    )
    parts = [CSS, '<div class="rp"><h2>Eval results</h2>']
    if request is not None:
        parts.append(
            f'<div class="sub">Constraints: accuracy ≥ {request.min_accuracy:.0%}, p50 latency ≤ {request.max_latency_ms:.0f} ms.</div>'
        )
    parts.append(
        "<table><tr><th>model</th><th class=num>n</th><th class=num>accuracy</th><th></th><th class=num>p50 ms</th><th class=num>p95 ms</th><th class=num>load s</th><th>device</th><th>verdict</th></tr>"
    )
    for r in results:
        acc_ok = request is None or r["accuracy"] >= request.min_accuracy
        lat_ok = request is None or r["latency_p50_ms"] <= request.max_latency_ms
        verdict = (
            ""
            if request is None
            else (
                '<span class="ok">passes</span>'
                if acc_ok and lat_ok
                else '<span class="bad">'
                + ", ".join(x for x, bad in (("accuracy", not acc_ok), ("latency", not lat_ok)) if bad)
                + " fails</span>"
            )
        )
        parts.append(
            f"<tr><td>{_esc(r['model'])}</td><td class=num>{r['n']}</td><td class=num>{r['accuracy']:.1%}</td><td>{_bar(r['accuracy'])}</td>"
            f"<td class=num>{r['latency_p50_ms']:.0f}</td><td class=num>{r['latency_p95_ms']:.0f}</td><td class=num>{r['load_seconds']:.0f}</td><td>{r['device']}</td><td>{verdict}</td></tr>"
        )
    parts.append("</table>")
    parts.append(
        "<h3>Per category</h3><table><tr><th>model</th>" + "".join(f"<th class=num>{c}</th>" for c in cats) + "</tr>"
    )
    for r in results:
        parts.append(
            f"<tr><td>{_esc(r['model'])}</td>"
            + "".join(f"<td class=num>{r['per_category'].get(c, 0):.0%}</td>" for c in cats)
            + "</tr>"
        )
    parts.append("</table>")
    for r in results:
        if r.get("mistakes"):
            parts.append(
                f"<h3>Sample mistakes: {_esc(r['model'])}</h3><table><tr><th>ticket</th><th>expected</th><th>got</th><th>raw reply</th></tr>"
            )
            for m in r["mistakes"][:6]:
                parts.append(
                    f"<tr><td>{_esc(m['ticket'])}</td><td>{m['expected']}</td><td class=bad>{m['got']}</td><td>{_esc(m['raw'])}</td></tr>"
                )
            parts.append("</table>")
    parts.append("</div>")
    return "".join(parts)


def loss_chart(history: list[dict], total_steps: int | None = None, width: int = 720, height: int = 220) -> str:
    """Training loss over steps as an inline SVG line, with epoch boundaries as dashed lines."""
    pts = [(h["step"], h["loss"]) for h in history if h.get("loss") is not None]
    if len(pts) < 2:
        return '<div class="sub">Loss curve appears after the first few logging steps.</div>'
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x_max = max(total_steps or 0, max(xs))
    lo, hi = 0.0, max(ys) * 1.05
    pad_l, pad_r, pad_t, pad_b = 44, 12, 12, 26
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    X = lambda x: pad_l + w * (x / x_max if x_max else 0)  # noqa: E731
    Y = lambda y: pad_t + h * (1 - (y - lo) / (hi - lo or 1))  # noqa: E731
    out = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" style="max-width:100%">']
    # grid: four horizontal lines with labels
    for i in range(5):
        y = lo + (hi - lo) * i / 4
        out.append(
            f'<line x1="{pad_l}" y1="{Y(y):.1f}" x2="{pad_l + w}" y2="{Y(y):.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{pad_l - 6}" y="{Y(y) + 4:.1f}" font-size="10" fill="{MUTED}" text-anchor="end">{y:.2f}</text>'
        )
    # epoch boundaries
    epochs_seen = sorted({int(h["epoch"]) for h in history if h.get("epoch", 0) >= 1})
    for e in epochs_seen:
        first = next((h["step"] for h in history if h.get("epoch", 0) >= e), None)
        if first is not None:
            out.append(
                f'<line x1="{X(first):.1f}" y1="{pad_t}" x2="{X(first):.1f}" y2="{pad_t + h}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3,3"/>'
            )
            out.append(f'<text x="{X(first) + 3:.1f}" y="{pad_t + 10}" font-size="10" fill="{MUTED}">epoch {e}</text>')
    poly = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    out.append(f'<polyline fill="none" stroke="{SERIES}" stroke-width="2" stroke-linejoin="round" points="{poly}"/>')
    out.append(
        f'<circle cx="{X(xs[-1]):.1f}" cy="{Y(ys[-1]):.1f}" r="3.5" fill="{SERIES}" stroke="#fff" stroke-width="2"><title>step {xs[-1]}: loss {ys[-1]:.3f}</title></circle>'
    )
    out.append(f'<text x="{pad_l}" y="{height - 6}" font-size="10" fill="{MUTED}">step 0</text>')
    out.append(
        f'<text x="{pad_l + w}" y="{height - 6}" font-size="10" fill="{MUTED}" text-anchor="end">step {x_max}</text>'
    )
    out.append("</svg>")
    return "".join(out)


def training_html(
    model: str,
    epochs: float,
    lora_r: int,
    dataset: str,
    history: list[dict],
    *,
    summary: str = "",
    total_steps: int | None = None,
    seconds: float | None = None,
    device: str = "",
) -> str:
    """One fine-tune: a live loss curve while it trains, the full picture when it is done."""
    done = bool(summary)
    losses = [h["loss"] for h in history]
    step = history[-1]["step"] if history else 0
    epoch_now = history[-1]["epoch"] if history else 0.0
    parts = [CSS, '<div class="rp">']
    parts.append(f"<h2>Fine-tune: {_esc(model)}</h2>")
    parts.append(
        f'<div class="sub">{epochs:g} epoch(s) · lora_r {lora_r} · dataset {dataset} · {"finished" if done else "training…"}</div>'
    )
    tiles = [(f"{step}" + (f" / {total_steps}" if total_steps else ""), "steps"), (f"{epoch_now:g}", "epoch")]
    if losses:
        tiles += [(f"{losses[0]:.2f}", "first loss"), (f"{losses[-1]:.2f}", "latest loss")]
    if seconds is not None:
        tiles.append((f"{seconds:.0f} s", f"on {device or 'gpu'}"))
    parts.append(
        '<div class="tiles">'
        + "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in tiles)
        + "</div>"
    )
    parts.append("<h3>Training loss</h3>" + loss_chart(history, total_steps))
    if done:
        ref = next((line for line in summary.splitlines() if line.startswith("artifact:")), "")
        parts.append(
            f'<div class="card">{_esc(summary.splitlines()[0])}' + (f"<br><b>{_esc(ref)}</b>" if ref else "") + "</div>"
        )
        parts.append(
            "<div class=sub>The artifact is listed under Artifacts in the Union UI, with this run as its source.</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def score(result: dict) -> dict:
    """Did the engineer do the job? Checked against the request it was given and its own numbers."""
    d, req, s = result["decision"], result["request"], result["stats"]
    promote_ok = "promote" in s.get("tools_used", []) and "promote" not in s.get("tools_failed", [])
    deployment_tested = "test_deployment" in s.get("tools_used", []) and "test_deployment" not in s.get(
        "tools_failed", []
    )
    accuracy_ok = d["accuracy"] >= req["min_accuracy"]
    latency_ok = d["latency_p50_ms"] <= req["max_latency_ms"]
    action = d.get("action") or ("promoted" if d.get("promoted") else "gave_up")
    return {
        "action": action,
        "accuracy_ok": accuracy_ok,
        "latency_ok": latency_ok,
        "budget_ok": d["runs_spent"] <= req["budget"],
        "promoted": promote_ok,
        "deployment_tested": deployment_tested,
        # Claims the decision makes that the record can check: that promote happened (or
        # did not), and that the constraints were met. An agent that promotes 94.2% against a
        # 95% bar and writes "meets the bar by rounding" fails the second one.
        "honest": (action == "promoted") == promote_ok
        and bool(d.get("meets_constraints")) == (accuracy_ok and latency_ok),
    }


_EVAL_RE = re.compile(r"accuracy ([\d.]+)% on (\d+) tickets; latency p50 (\d+) ms")


def _measurements(result: dict) -> list[dict]:
    """Every eval the agent ran, parsed back out of the tool results."""
    rows = []
    for call in result.get("tool_log", []):
        if call["name"] not in ("run_eval", "run_eval_cpu") or not call["ok"]:
            continue
        m = _EVAL_RE.search(call["output"])
        if not m:
            continue
        model = str(call["args"].get("model", "?"))
        rows.append(
            {
                "model": model.split("@")[0].removeprefix("artifact:ticket-router-")
                if model.startswith("artifact:")
                else model,
                "fine_tuned": model.startswith("artifact:"),
                "device": "CPU" if call["name"] == "run_eval_cpu" else "T4",
                "dataset": call["args"].get("dataset", "v1"),
                "n": int(m.group(2)),
                "accuracy": float(m.group(1)) / 100,
                "p50": float(m.group(3)),
            }
        )
    return rows


def decision_html(result: dict) -> str:
    d, s, sc, req = result["decision"], result["stats"], score(result), result["request"]
    chip = lambda ok, text: f'<span class="chip {"pass" if ok else "fail"}">{"✓" if ok else "✗"} {text}</span>'  # noqa: E731
    met = sc["accuracy_ok"] and sc["latency_ok"]
    action = sc["action"]
    eyebrow = {
        "promoted": "Promoted to production",
        "kept_production": "Kept in production",
        "gave_up": "Nothing promoted",
    }[action]
    verdict = (
        "meets the request"
        if met and action in ("promoted", "kept_production")
        else ("promoted, but misses the bar" if action == "promoted" else "the request could not be met")
    )
    good = met and action in ("promoted", "kept_production") and sc["honest"]
    promoted = _esc(d["promoted_model"] or "nothing")
    if str(d["promoted_model"]).startswith("artifact:"):
        promoted = (
            _esc(str(d["promoted_model"]).split("@")[0].removeprefix("artifact:ticket-router-"))
            + f'<span class="dim"> @{_esc(str(d["promoted_model"]).split("@")[-1])}</span>'
        )

    parts = [CSS, '<div class="rp">']
    # Hero: what was promoted and whether it is any good.
    parts.append(
        f'<div class="hero {"good" if good else "bad"}">'
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="big">{promoted}</div>'
        f'<div class="sub">base {_esc(d["base_model"])}{" · fine-tuned" if d["fine_tuned"] else ""} · dataset {req["dataset"]} · agent {_esc(result["model"])} · <b>{verdict}</b></div>'
        '<div class="chips">'
        + chip(sc["accuracy_ok"], f"accuracy {d['accuracy']:.1%} ≥ {req['min_accuracy']:.0%}")
        + chip(sc["latency_ok"], f"p50 {d['latency_p50_ms']:.0f} ms ≤ {req['max_latency_ms']:.0f} ms")
        + chip(sc["budget_ok"], f"{d['runs_spent']} of {req['budget']} runs")
        + (chip(sc["promoted"], "promote succeeded") if action == "promoted" else "")
        + (
            chip(
                sc["deployment_tested"],
                "deployment tested"
                + (
                    " · passed"
                    if d.get("deployment_test_passed")
                    else (" · failed" if d.get("deployment_test_passed") is False else "")
                ),
            )
            if action == "promoted"
            else ""
        )
        + chip(sc["honest"], "reported honestly")
        + (chip(False, "rolled back") if d.get("rolled_back") else "")
        + "</div></div>"
    )
    # Stat tiles.
    tiles = [
        (f"{s['model_turns']}", "model turns"),
        (f"{s['tool_calls']}", "tool calls"),
        (f"{s['input_tokens'] + s['output_tokens']:,}", "tokens"),
        (f"{s['seconds']:.0f} s", "wall clock"),
        (f"{sum(1 for c in result.get('tool_log', []) if c['name'] == 'fine_tune')}", "fine-tunes"),
    ]
    parts.append(
        '<div class="tiles">'
        + "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in tiles)
        + "</div>"
    )

    # What it measured.
    rows = _measurements(result)
    if rows:
        parts.append("<h3>What the agent measured</h3>")
        parts.append(
            "<table><tr><th>model</th><th>on</th><th class=num>n</th><th class=num>accuracy</th><th style='width:170px'></th><th class=num>p50 ms</th><th>against the bar</th></tr>"
        )
        for r in rows:
            acc_ok, lat_ok = r["accuracy"] >= req["min_accuracy"], r["p50"] <= req["max_latency_ms"]
            note = "T4 latency" if r["device"] == "T4" else "CPU latency (serving)"
            bar_ok = acc_ok and (lat_ok if r["device"] == "T4" else True)
            verdict_cell = (
                '<span class="ok">passes</span>'
                if bar_ok
                else '<span class="bad">'
                + ", ".join(
                    x for x, bad in (("accuracy", not acc_ok), ("latency", r["device"] == "T4" and not lat_ok)) if bad
                )
                + "</span>"
            )
            parts.append(
                f"<tr><td>{_esc(r['model'])}{' <span class=dim>fine-tuned</span>' if r['fine_tuned'] else ''}</td><td>{r['device']}{'' if r['dataset'] == 'v1' else ' · ' + r['dataset']}</td>"
                f"<td class=num>{r['n']}</td><td class=num>{r['accuracy']:.1%}</td><td>{_bar(r['accuracy'])}</td><td class=num>{r['p50']:.0f}</td><td>{verdict_cell}<span class=dim> · {note}</span></td></tr>"
            )
        parts.append("</table>")

    # The sequence of calls.
    log = result.get("tool_log", [])
    if log:
        parts.append("<h3>What the agent did</h3><ol class=steps>")
        for c in log:
            args = ", ".join(
                f"{k}={_esc(str(v).split('@')[0].removeprefix('artifact:ticket-router-'))}"
                for k, v in c["args"].items()
                if k in ("model", "epochs", "lora_r", "dataset", "n", "version")
            )
            head = _esc(c["output"].splitlines()[0][:150]) if c["output"] else ""
            parts.append(
                f'<li class="{"" if c["ok"] else "failed"}"><b>{_esc(c["name"])}</b><span class=dim>({args})</span><div class=out>{head}</div></li>'
            )
        parts.append("</ol>")

    # The write-up.
    parts.append(
        f'<h3>Rationale</h3><div class="card">{_esc(d["rationale"])}<ul>'
        + "".join(f"<li>{_esc(e)}</li>" for e in d.get("evidence", []))
        + "</ul></div>"
    )
    if s.get("tools_failed"):
        parts.append(f'<div class="bad">Failed tool calls: {", ".join(s["tools_failed"])}</div>')
    parts.append("</div>")
    return "".join(parts)


def support_html(rows: list[dict], stats: dict) -> str:
    """The support agent's batch: how routing went, what it cost, where the tickets landed."""
    labels = list(stats["queue_distribution"].keys())
    parts = [CSS, '<div class="rp">']
    router = "the API model" if stats["router"] == "llm" else "the ticket-router app (self-hosted)"
    parts.append(f"<h2>Support agent · {stats['n']} tickets · routed by {router}</h2>")
    parts.append(
        f'<div class="sub">tickets from dataset {stats["dataset"]} · router knows the {stats["labels_from"]} queues · replies drafted by {_esc(stats["model"])}</div>'
    )
    cost = stats.get("cost_per_1000_tickets_usd")
    tiles = [
        (f"{stats['routing_accuracy']:.1%}", "routing accuracy"),
        (f"{stats['route_p50_ms']:.0f} ms", "route p50"),
        ("n/a" if cost is None else f"${cost:.2f}", "per 1,000 tickets (routing + reply)"),
        (f"{stats['tokens']:,}", "tokens"),
        (f"{stats['seconds']:.0f} s", "batch wall clock"),
    ]
    parts.append(
        '<div class="tiles">'
        + "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in tiles)
        + "</div>"
    )
    if stats.get("tickets_with_unknown_queue"):
        parts.append(
            f'<div class="bad">{stats["tickets_with_unknown_queue"]} of these tickets belong to a queue the router does not know about. That is drift.</div>'
        )

    parts.append("<h3>Where the tickets went</h3><table>")
    total = max(1, stats["n"])
    for q in labels:
        n = stats["queue_distribution"][q]
        expected = sum(r["expected"] == q for r in rows)
        parts.append(
            f"<tr><td>{q}</td><td class=num>{n}</td><td>{_bar(n / total, 220)}</td><td class=dim>expected {expected}</td></tr>"
        )
    missing = {r["expected"] for r in rows if r["expected"] not in labels}
    for q in sorted(missing):
        expected = sum(r["expected"] == q for r in rows)
        parts.append(
            f'<tr><td class="bad">{q}</td><td class=num>0</td><td></td><td class="bad">expected {expected}, but the router has no such queue</td></tr>'
        )
    parts.append("</table>")

    parts.append(
        "<h3>Tickets</h3><table><tr><th>ticket</th><th>routed to</th><th>expected</th><th class=num>conf.</th><th class=num>ms</th><th>draft reply</th></tr>"
    )
    for r in rows:
        ok = r["queue"] == r["expected"]
        conf = "" if r.get("confidence") is None else f"{r['confidence']:.0%}"
        parts.append(
            f'<tr><td>{_esc(r["text"][:90])}</td><td class="{"ok" if ok else "bad"}">{r["queue"]}</td><td>{r["expected"]}</td>'
            f"<td class=num>{conf}</td><td class=num>{r['route_ms']:.0f}</td><td class=dim>{_esc(r['reply'][:140])}</td></tr>"
        )
    parts.append("</table></div>")
    return "".join(parts)


def bakeoff_html(rows: list[dict]) -> str:
    """One row per (agent model, trial); a summary per agent model on top."""
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    parts = [CSS, '<div class="rp"><h2>Agent bake-off</h2>']
    parts.append(
        f'<div class="sub">{len(rows)} runs · {len(models)} agent models. Per-turn cost and latency are in Grafana.</div>'
    )
    parts.append(
        "<table><tr><th>agent model</th><th class=num>met constraints</th><th class=num>within budget</th><th class=num>avg runs spent</th><th class=num>avg tool calls</th><th class=num>avg tokens</th><th class=num>avg seconds</th></tr>"
    )
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        n = len(rs)
        scores = [score(r) for r in rs]
        met = sum(sc["accuracy_ok"] and sc["latency_ok"] and sc["promoted"] for sc in scores)
        budget = sum(sc["budget_ok"] for sc in scores)
        avg_runs = sum(r["decision"]["runs_spent"] for r in rs) / n
        avg_calls = sum(r["stats"]["tool_calls"] for r in rs) / n
        avg_tokens = sum(r["stats"]["input_tokens"] + r["stats"]["output_tokens"] for r in rs) / n
        avg_seconds = sum(r["stats"]["seconds"] for r in rs) / n
        parts.append(
            f"<tr><td><b>{_esc(m)}</b></td><td class=num>{met}/{n}</td><td class=num>{budget}/{n}</td>"
            f"<td class=num>{avg_runs:.1f}</td><td class=num>{avg_calls:.1f}</td>"
            f"<td class=num>{avg_tokens:,.0f}</td><td class=num>{avg_seconds:.0f}</td></tr>"
        )
    parts.append(
        "</table><h3>Every run</h3><table><tr><th>agent model</th><th>promoted</th><th class=num>accuracy</th><th class=num>p50 ms</th><th class=num>runs</th><th class=num>tokens</th><th class=num>s</th></tr>"
    )
    for r in rows:
        d, sc = r["decision"], score(r)
        cls = "ok" if sc["accuracy_ok"] and sc["latency_ok"] else "bad"
        parts.append(
            f'<tr><td>{_esc(r["model"])}</td><td class="{cls}">{_esc(d["promoted_model"])}</td><td class=num>{d["accuracy"]:.1%}</td>'
            f"<td class=num>{d['latency_p50_ms']:.0f}</td><td class=num>{d['runs_spent']}</td>"
            f"<td class=num>{r['stats']['input_tokens'] + r['stats']['output_tokens']:,}</td><td class=num>{r['stats']['seconds']:.0f}</td></tr>"
        )
    parts.append("</table></div>")
    return "".join(parts)
