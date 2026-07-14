"""Turns the blocks the render tools produced into a Flyte HTML report.

Nothing conceptually interesting here — it is the presentation layer. The one
thing worth noting is that we always show the generated program alongside the
results, because in code mode the program *is* the reasoning: if an answer looks
wrong, the code is where you find out why.
"""

from __future__ import annotations

import html
import json

_CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
       margin: 0; padding: 24px; line-height: 1.5; }
h1 { font-size: 1.4rem; margin: 0 0 4px; }
.question { color: #6b7280; margin-bottom: 24px; }
.summary { padding: 14px 16px; border-left: 3px solid #6366f1;
           background: rgba(99,102,241,.08); border-radius: 4px; margin-bottom: 24px; }
.metrics { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 24px; }
.metric { flex: 1 1 160px; padding: 14px 16px; border: 1px solid rgba(128,128,128,.25);
          border-radius: 8px; }
.metric-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .04em;
                color: #6b7280; }
.metric-value { font-size: 1.6rem; font-weight: 600; margin: 2px 0; }
.metric-note { font-size: .8rem; color: #6b7280; }
.chart { margin-bottom: 24px; max-width: 720px; }
.table { margin-bottom: 24px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid rgba(128,128,128,.2); }
th { font-weight: 600; color: #6b7280; }
details { margin-top: 32px; }
summary { cursor: pointer; color: #6b7280; font-size: .9rem; }
pre { overflow-x: auto; padding: 14px; border-radius: 6px;
      background: rgba(128,128,128,.1); font-size: .82rem; }
"""

CHART_JS_CDN = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>'


def chart_html(spec: dict, canvas_id: str) -> str:
    """A self-contained chart: a canvas plus the script that draws it.

    Self-contained on purpose. The Flyte report renders this as a normal page, but
    the chat app in step 5 injects it into an existing page and then re-executes
    any <script> tags it finds — so a chart that depends on a separate init pass
    elsewhere on the page would render in one place and silently not the other.
    """
    return (
        f'<div class="chart" style="position:relative;height:340px;max-width:720px">'
        f'<canvas id="{html.escape(canvas_id)}"></canvas></div>'
        f"<script>new Chart(document.getElementById('{html.escape(canvas_id)}'),"
        f"{json.dumps(spec)});</script>"
    )


def render(question: str, code: str | list[str], blocks: list[str], summary: str) -> str:
    """Assemble the report: summary, metric cards, charts and tables, then the code.

    `code` may be a single program or every program the model wrote. Show all of
    them: in code mode the program *is* the reasoning, so if an answer looks wrong,
    this is where you find out why.
    """
    programs = [code] if isinstance(code, str) else list(code)
    if len(programs) > 1:
        listing = "".join(
            f"<p style='color:#6b7280;font-size:.8rem;margin:12px 0 4px'>Program {i}"
            f" of {len(programs)}</p><pre><code>{html.escape(p)}</code></pre>"
            for i, p in enumerate(programs, 1)
        )
    else:
        listing = f"<pre><code>{html.escape(programs[0] if programs else '')}</code></pre>"

    metrics = [b for b in blocks if b.startswith('<div class="metric"')]
    rest = [b for b in blocks if not b.startswith('<div class="metric"')]

    metric_row = f'<div class="metrics">{"".join(metrics)}</div>' if metrics else ""

    # Chart.js first: the per-chart scripts below call `new Chart` as they are
    # parsed, so loading the library at the end of the page would be too late.
    return f"""{CHART_JS_CDN}
<style>{_CSS}</style>
<h1>NYC Taxi analysis</h1>
<div class="question">{html.escape(question)}</div>
<div class="summary">{html.escape(summary)}</div>
{metric_row}
{"".join(rest)}
<details>
  <summary>{"The programs the model wrote" if len(programs) > 1 else "The program the model wrote"}</summary>
  {listing}
</details>
"""
