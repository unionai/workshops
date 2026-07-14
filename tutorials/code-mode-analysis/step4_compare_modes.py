"""Step 4 — why bother? Run the same question both ways and count.

Everything so far has taken it on faith that writing code beats calling tools one
at a time. This step stops asserting it and measures it.

The two agents are identical — same model, same tools, same instructions, same
question. One flag differs:

    code_mode=False   the model emits one JSON tool call, reads the result,
                      thinks, emits the next one. Every intermediate result —
                      every row of every query — travels back through the context
                      window, and every step costs a round-trip to the model.

    code_mode=True    the model writes one program. The loop over months runs in
                      the sandbox. The intermediate rows never reach the model at
                      all; only the final summary does.

Ask a question that touches several months and the gap is not subtle. Measured on
"Which borough tipped best in each quarter of 2024?" with Claude Opus 4.8:

                    sequential   code mode
    model turns              9           2
    tool calls              22           0
    total tokens        91,185      10,320
    wall clock            158s         20s

The turn count is the number to watch: sequential grows with the work, code mode
doesn't. Every one of those 22 tool results also passed back through the context
window, which is where the tokens went.

Both agents get the same tools — but sequential has a harder time using them. It
commonly fumbles `create_table`, because handing a list of row dicts to a JSON tool
call is awkward, while in code mode a list of dicts is just a list of dicts. That
friction is itself a cost of sequential tool calling. Both answers are printed at
the bottom of the report, side by side.

Run it:

    uv run flyte run step4_compare_modes.py compare \\
        --question "Which borough tipped best in each quarter of 2024?"
"""

from __future__ import annotations

import tools
from analysis import build_agent  # same tools, same instructions, same durable query
from config import env
from llm import Usage


async def _run_one(question: str, code_mode: bool) -> tuple[Usage, str, str]:
    tools.new_report()
    # Headroom for sequential. Code mode won't use it.
    agent, usage = build_agent(code_mode=code_mode, max_turns=30)
    result = await agent.run.aio(question)
    return usage, result.summary or result.error, result.code


@env.task(report=True)
async def compare(question: str = "Which borough tipped best in each quarter of 2024?") -> dict:
    """Answer the same question twice — once per mode — and chart the cost."""
    import flyte.report

    sequential, seq_summary, _ = await _run_one(question, code_mode=False)
    coded, code_summary, code_src = await _run_one(question, code_mode=True)

    await flyte.report.replace.aio(
        _render(question, sequential, coded, seq_summary, code_summary, code_src)
    )
    await flyte.report.flush.aio()

    return {
        "sequential_turns": sequential.turns,
        "code_mode_turns": coded.turns,
        "sequential_tokens": sequential.total_tokens,
        "code_mode_tokens": coded.total_tokens,
        "token_reduction_pct": round(
            (1 - coded.total_tokens / sequential.total_tokens) * 100, 1
        )
        if sequential.total_tokens
        else 0.0,
    }


def _render(question, seq: Usage, code: Usage, seq_summary, code_summary, code_src) -> str:
    import html as _h

    from report import CHART_JS_CDN, chart_html

    def saving(a: int, b: int) -> str:
        if not a:
            return "—"
        return f"{(1 - b / a) * 100:.0f}% fewer"

    def chart(title: str, key: str) -> str:
        spec = {
            "type": "bar",
            "data": {
                "labels": ["Sequential tool calls", "Code mode"],
                "datasets": [
                    {
                        "label": title,
                        "data": [getattr(seq, key), getattr(code, key)],
                        "backgroundColor": ["#f87171", "#34d399"],
                    }
                ],
            },
            "options": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {"legend": {"display": False},
                            "title": {"display": True, "text": title}},
            },
        }
        return chart_html(spec, canvas_id=f"cmp-{key}")

    rows = [
        ("Model turns", seq.turns, code.turns, saving(seq.turns, code.turns)),
        ("Tool calls", seq.tool_calls, code.tool_calls, "—"),
        ("Input tokens", f"{seq.input_tokens:,}", f"{code.input_tokens:,}",
         saving(seq.input_tokens, code.input_tokens)),
        ("Output tokens", f"{seq.output_tokens:,}", f"{code.output_tokens:,}",
         saving(seq.output_tokens, code.output_tokens)),
        ("Total tokens", f"{seq.total_tokens:,}", f"{code.total_tokens:,}",
         saving(seq.total_tokens, code.total_tokens)),
        ("Seconds in the model", f"{seq.seconds:.1f}", f"{code.seconds:.1f}",
         saving(int(seq.seconds * 100), int(code.seconds * 100))),
    ]
    body = "".join(
        f"<tr><td>{_h.escape(str(a))}</td><td>{_h.escape(str(b))}</td>"
        f"<td>{_h.escape(str(c))}</td><td><b>{_h.escape(str(d))}</b></td></tr>"
        for a, b, c, d in rows
    )

    return f"""{CHART_JS_CDN}
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; padding: 24px; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; max-width: 720px; margin: 20px 0; }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid rgba(128,128,128,.2); }}
th {{ color: #6b7280; font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }}
pre {{ background: rgba(128,128,128,.1); padding: 14px; border-radius: 6px; overflow-x: auto;
      font-size: .82rem; }}
.charts {{ display: flex; gap: 24px; flex-wrap: wrap; }}
.answers {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0; }}
.answer {{ flex: 1 1 280px; padding: 12px 14px; border: 1px solid rgba(128,128,128,.25);
          border-radius: 8px; font-size: .9rem; }}
</style>
<h1>Sequential tool calling vs code mode</h1>
<p style="color:#6b7280">{_h.escape(question)}</p>

<div class="charts">{chart("Model turns", "turns")}{chart("Total tokens", "total_tokens")}</div>

<table>
  <thead><tr><th></th><th>Sequential</th><th>Code mode</th><th>Difference</th></tr></thead>
  <tbody>{body}</tbody>
</table>

<p style="color:#6b7280;font-size:.9rem">Both agents had the same tools and the same
question. The only difference is whether the model emitted one JSON tool call at a
time, or wrote a program that called the tools itself.</p>

<div class="answers">
  <div class="answer"><b>Sequential answer</b><br>{_h.escape(str(seq_summary))}</div>
  <div class="answer"><b>Code mode answer</b><br>{_h.escape(str(code_summary))}</div>
</div>

<details><summary style="cursor:pointer;color:#6b7280">The program code mode wrote</summary>
<pre><code>{_h.escape(code_src or "")}</code></pre></details>
"""


if __name__ == "__main__":
    import flyte

    flyte.init_from_config(image_builder="remote")
    run = flyte.run(compare, question="Which borough tipped best in each quarter of 2024?")
    print(f"View at: {run.url}")
    run.wait()
    print(run.outputs())
