"""
Reflection agent workflow — iterative self-improvement through critique.

The agent generates a response, critiques it, and refines until the quality
threshold is met or max iterations are reached.

Usage:
    # Local with TUI
    flyte run --local --tui agent_reflection/workflow.py reflection_agent --request "Write a Python function to find all prime numbers up to N"

    # Remote (on Flyte cluster)
    flyte run agent_reflection/workflow.py reflection_agent --request "Write a Python function to find all prime numbers up to N"
"""

import base64
import json
import logging
import markdown
import flyte
import flyte.report
from config import base_env, OPENAI_API_KEY
from agent_reflection.graph import build_reflection_graph

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("agent_reflection.graph").setLevel(logging.INFO)

env = base_env


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


@env.task(report=True)
async def reflection_agent(request: str, quality_threshold: int = 8, max_iterations: int = 3) -> str:
    """Run the reflection agent on a request."""
    log.info(f"Request: {request}")
    log.info(f"Quality threshold: {quality_threshold}/10, Max iterations: {max_iterations}")

    graph = build_reflection_graph(
        openai_api_key=OPENAI_API_KEY,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations,
    )

    # Show graph visualization
    png_bytes = graph.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    await flyte.report.log.aio(
        f"<h2>Reflection Agent Graph</h2>"
        f'<img src="data:image/png;base64,{img_b64}" alt="Reflection agent graph" />'
        f"<h2>Request</h2><p>{request}</p>"
        f"<p>Generating...</p>"
    )
    await flyte.report.flush.aio()

    result = await graph.ainvoke({"request": request})

    answer = result["response"]
    score = result["score"]
    iterations = result["iteration"]
    log.info(f"Done. Final score: {score}/10 after {iterations} iteration(s)")

    # Build iteration history for report
    history_html = "<h2>Iteration History</h2><ol>"
    for entry in result.get("history", []):
        history_html += f"<li>{entry}</li>"
    history_html += "</ol>"

    await flyte.report.replace.aio(
        f"<h2>Reflection Agent</h2>"
        f"<p><b>Request:</b> {request}</p>"
        f"<p><b>Final Score:</b> {score}/10 after {iterations} iteration(s)</p>"
        f"<h2>Final Response</h2>{md_to_html(answer)}"
        f"<hr/>{history_html}"
        f'<hr/><h2>Agent Graph</h2><img src="data:image/png;base64,{img_b64}" />'
    )
    await flyte.report.flush.aio()

    return json.dumps({"response": answer, "score": score, "iterations": iterations})


