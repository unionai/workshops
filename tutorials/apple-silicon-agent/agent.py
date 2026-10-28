"""A LangGraph agent whose brain is a model on your Mac's GPU and whose tools are Flyte tasks.

    flyte run agent.py ask --question "Which artist has the most albums?"
    flyte run agent.py evaluate                      # 16 questions with known answers, scored
    AGENT_MODEL=anthropic:claude-haiku-4-5 flyte run agent.py evaluate    # swap the brain

The agent is a plain ReAct loop: the model picks a tool, the tool runs, the result goes back to the
model, until it answers. What Flyte adds is that every tool call is a child action (durable,
retryable, inspectable in the run graph) and that `evaluate` fans the questions out concurrently.
The model never touches the agent's container: it is served by `serve_model.py`, natively on the
Mac for `ACCELERATOR=apple`, and the agent talks to it over the OpenAI API.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time

import flyte
import flyte.report

import tools
from config import AGENT_MODEL, AGENT_SECRETS, MODEL_NAME, PROPAGATED, llm_base_url
from report_helpers import make_bar_chart, wrap_report

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

agent_env = flyte.TaskEnvironment(
    name="apple-silicon-agent",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "langgraph>=0.4", "langchain-openai>=0.3", "langchain-anthropic>=0.3", "python-dotenv"
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    env_vars=PROPAGATED,
    secrets=AGENT_SECRETS,
    depends_on=[tools.tools_env],
)

SYSTEM = """You are a data analyst with read-only access to the Chinook music store database (SQLite).
Answer the user's question by querying the database with the run_sql tool. The schema:

{schema}

Rules:
- Use only the tables and columns listed above. Join on the *Id columns (Album.ArtistId = Artist.ArtistId, and so on).
- Always call run_sql at least once before answering; never guess a value.
- Keep queries small: aggregate in SQL and use ORDER BY ... LIMIT for top-N questions.
- Report names, not ids: when the answer is an artist, customer, employee or genre, join to its Name columns.
- If run_sql returns an error, fix the query and call it again.
- When you have the result, answer in one short sentence that states the value(s) plainly, then stop calling tools.
"""

MAX_STEPS = 12


# ── The brain ────────────────────────────────────────────────────────────────


def chat_model(spec: str = AGENT_MODEL):
    """A LangChain chat model for an AGENT_MODEL spec: `local`, `anthropic:<model>` or `openai:<model>`."""
    provider, _, name = spec.partition(":")
    if provider == "local":
        from langchain_openai import ChatOpenAI

        base_url = llm_base_url()
        log.info(f"local model {MODEL_NAME} at {base_url}")
        return ChatOpenAI(model=MODEL_NAME, base_url=base_url, api_key="not-used", temperature=0, max_tokens=512, timeout=600, max_retries=1)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, max_tokens=1024, temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=0)
    raise ValueError(f"unknown AGENT_MODEL {spec!r}: use local, anthropic:<model> or openai:<model>")


def model_label(spec: str = AGENT_MODEL) -> str:
    return MODEL_NAME if spec == "local" else spec.split(":", 1)[-1]


# ── The tools, as the model sees them ────────────────────────────────────────
# Each wrapper calls a Flyte task, so the call becomes a child action of the agent's run.


def langchain_tools():
    from langchain_core.tools import tool

    @tool
    async def list_tables() -> str:
        """List the tables in the database."""
        return json.dumps(await tools.list_tables())

    @tool
    async def describe_table(table: str) -> str:
        """The columns of one table, with types, and three sample rows."""
        return await tools.describe_table(table)

    @tool
    async def run_sql(query: str) -> str:
        """Run one read-only SQL query (SQLite dialect). Returns up to 50 rows as JSON, or an error message."""
        return await tools.run_sql(query)

    return [list_tables, describe_table, run_sql]


# ── The loop ─────────────────────────────────────────────────────────────────


async def run_agent(question: str, spec: str = AGENT_MODEL) -> dict:
    """Ask the agent one question. Returns the answer plus the trace of tool calls."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langgraph.prebuilt import create_react_agent

    graph = create_react_agent(chat_model(spec), langchain_tools())
    start = time.perf_counter()
    # The schema is a tool call the model did not have to make: a small model does far better when
    # it is told the tables up front and only has to write the query.
    system = SYSTEM.format(schema=await tools.schema())
    try:
        state = await graph.ainvoke(
            {"messages": [SystemMessage(system), HumanMessage(question)]},
            config={"recursion_limit": 2 * MAX_STEPS + 1},
        )
        messages, error = state["messages"], None
    except Exception as e:  # recursion limit, model errors: keep what we have, report it
        messages, error = [], f"{type(e).__name__}: {e}"
    seconds = time.perf_counter() - start

    trace, answer, tokens = [], "", {"input": 0, "output": 0}
    for m in messages:
        if isinstance(m, AIMessage):
            usage = m.usage_metadata or {}
            tokens["input"] += usage.get("input_tokens", 0)
            tokens["output"] += usage.get("output_tokens", 0)
            for call in m.tool_calls:
                trace.append({"tool": call["name"], "args": call["args"], "result": None})
            if not m.tool_calls and (m.content or "").strip():
                answer = m.content if isinstance(m.content, str) else str(m.content)
        elif isinstance(m, ToolMessage):
            for t in reversed(trace):
                if t["result"] is None:
                    t["result"] = m.content if isinstance(m.content, str) else str(m.content)
                    break
    return {
        "question": question,
        "answer": answer.strip(),
        "model": model_label(spec),
        "tool_calls": trace,
        "steps": len(trace),
        "seconds": round(seconds, 1),
        "tokens": tokens,
        "error": error,
    }


def _trace_html(result: dict) -> str:
    rows = "".join(
        f"<tr><td>{i + 1}</td><td><code>{t['tool']}</code></td><td><code>{json.dumps(t['args'])[:300]}</code></td>"
        f"<td><code>{(t['result'] or '')[:400]}</code></td></tr>"
        for i, t in enumerate(result["tool_calls"])
    )
    return (
        f"<div class='card'><b>Q:</b> {result['question']}<br><b>A:</b> {result['answer'] or '(no answer)'}"
        + (f"<br><span style='color:#b00'>{result['error']}</span>" if result["error"] else "")
        + f"<br><small>{result['model']} &middot; {result['steps']} tool calls &middot; {result['seconds']}s &middot; "
        f"{result['tokens']['input']} in / {result['tokens']['output']} out tokens</small></div>"
        "<table><tr><th>#</th><th>Tool</th><th>Arguments</th><th>Result</th></tr>" + rows + "</table>"
    )


@agent_env.task(report=True)
async def ask(question: str = "Which artist has the most albums?") -> dict:
    """Ask the agent one question. The report shows every tool call and what came back."""
    result = await run_agent(question)
    log.info(f"Q: {question}\nA: {result['answer']}  ({result['steps']} tool calls, {result['seconds']}s)")
    await flyte.report.replace.aio(wrap_report(f"<h2>Agent on {result['model']}</h2>" + _trace_html(result)), do_flush=True)
    return result


# ── Evaluation ───────────────────────────────────────────────────────────────

# Gold answers computed directly against Chinook v1.4.5.
QUESTIONS: list[tuple[str, str]] = [
    ("How many customers are there?", "59"),
    ("How many employees are there?", "8"),
    ("How many genres are in the database?", "25"),
    ("Which country has the most customers?", "USA"),
    ("Which artist has the most albums?", "Iron Maiden"),
    ("How many tracks are in the Rock genre?", "1297"),
    ("What is the total of all invoices, in dollars?", "2328.6"),
    ("What is the average invoice total?", "5.65"),
    ("Which employee is the support rep for the most customers?", "Jane Peacock"),
    ("What is the name of the longest track?", "Occupation / Precipice"),
    ("How many playlists contain more than 100 tracks?", "5"),
    ("Which customer has spent the most in total?", "Helena Holý"),
    ("Which genre has the highest total sales revenue?", "Rock"),
    ("How many tracks are longer than ten minutes?", "260"),
    ("How many customers are from Brazil?", "5"),
    ("How many albums does Led Zeppelin have?", "14"),
]


def is_correct(answer: str, gold: str) -> bool:
    """Numbers: any number in the answer within 0.01 of the gold (commas ignored). Text: case-insensitive substring."""
    try:
        target = float(gold)
    except ValueError:
        return gold.lower() in answer.lower()
    for found in re.findall(r"-?\d[\d,]*\.?\d*", answer):
        try:
            if abs(float(found.replace(",", "")) - target) < 0.01:
                return True
        except ValueError:
            continue
    return False


@agent_env.task(report=True)
async def evaluate(num_questions: int = 16, concurrency: int = 4) -> dict:
    """Ask every question, `concurrency` at a time, and score the answers."""
    questions = QUESTIONS[:num_questions]
    sem = asyncio.Semaphore(concurrency)

    async def one(q: str) -> dict:
        async with sem:
            return await ask(q)

    start = time.perf_counter()
    results = await asyncio.gather(*(one(q) for q, _ in questions))
    wall = time.perf_counter() - start
    for r, (_, gold) in zip(results, questions):
        r["gold"], r["correct"] = gold, is_correct(r["answer"], gold)
        log.info(f"{'PASS' if r['correct'] else 'FAIL'} {r['question']} -> {r['answer'][:80]!r} (gold {gold})")

    correct = sum(r["correct"] for r in results)
    summary = {
        "model": results[0]["model"] if results else model_label(),
        "questions": len(results),
        "correct": correct,
        "accuracy": round(correct / len(results), 3) if results else 0.0,
        "mean_tool_calls": round(sum(r["steps"] for r in results) / len(results), 1) if results else 0,
        "mean_seconds": round(sum(r["seconds"] for r in results) / len(results), 1) if results else 0,
        "wall_seconds": round(wall, 1),
        "errors": sum(1 for r in results if r["error"]),
    }
    log.info(json.dumps(summary))

    rows = "".join(
        f"<tr><td>{'✅' if r['correct'] else '❌'}</td><td>{r['question']}</td><td>{r['gold']}</td>"
        f"<td>{(r['answer'] or r['error'] or '')[:200]}</td><td>{r['steps']}</td><td>{r['seconds']}s</td></tr>"
        for r in results
    )
    body = (
        f"<h2>SQL agent on {summary['model']}</h2><div class='stats'>"
        f"<div class='stat'><div class='value'>{summary['accuracy']:.0%}</div><div class='label'>correct ({correct}/{len(results)})</div></div>"
        f"<div class='stat'><div class='value'>{summary['mean_tool_calls']}</div><div class='label'>tool calls per question</div></div>"
        f"<div class='stat'><div class='value'>{summary['mean_seconds']}s</div><div class='label'>per question</div></div>"
        f"<div class='stat'><div class='value'>{summary['wall_seconds']}s</div><div class='label'>wall clock, {concurrency} concurrent</div></div>"
        "</div><div class='chart-container'>"
        + make_bar_chart([r["question"][:28] for r in results], {"tool calls": [r["steps"] for r in results]}, title="Tool calls per question", colors=["#5a7db5"])
        + "</div><table><tr><th></th><th>Question</th><th>Gold</th><th>Answer</th><th>Calls</th><th>Time</th></tr>" + rows + "</table>"
    )
    await flyte.report.replace.aio(wrap_report(body), do_flush=True)
    return summary


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(evaluate)
    print(run.url)
    run.wait()
    print(run.outputs())
