"""
ReAct agent workflow — reason, act, observe loop with math and string tools.

The agent breaks down problems step-by-step, calling tools as needed,
until it arrives at a final answer.

Usage:
    # Local with TUI
    flyte run --local --tui agent_react/workflow.py react_agent --request "How many words are in 'the quick brown fox' multiplied by 5?"

    # Remote (on Flyte cluster)
    flyte run agent_react/workflow.py react_agent --request "How many words are in 'the quick brown fox' multiplied by 5?"
"""

import base64
import logging
import markdown
import flyte
import flyte.report
from langchain_core.messages import HumanMessage
from config import base_env, OPENAI_API_KEY
from agent_react.graph import build_react_graph

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("agent_react.graph").setLevel(logging.INFO)
logging.getLogger("tools.math").setLevel(logging.INFO)
logging.getLogger("tools.string").setLevel(logging.INFO)

env = base_env


def md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


@env.task(report=True)
async def react_agent(request: str, max_steps: int = 10) -> str:
    """Run the ReAct agent on a request."""
    log.info(f"Task: {request}")

    graph = build_react_graph(openai_api_key=OPENAI_API_KEY, max_steps=max_steps)

    # Show graph visualization in report
    png_bytes = graph.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    await flyte.report.log.aio(
        f"<h2>ReAct Agent Graph</h2>"
        f'<img src="data:image/png;base64,{img_b64}" alt="ReAct agent graph" />'
        f"<h2>Task</h2><p>{request}</p>"
        f"<p>Running...</p>"
    )
    await flyte.report.flush.aio()

    result = await graph.ainvoke({"messages": [HumanMessage(content=request)]})
    answer = result["messages"][-1].content
    log.info(f"Answer: {answer}")

    # Build step-by-step trace for report
    steps_html = "<h2>Reasoning Trace</h2><ol>"
    for msg in result["messages"][1:]:  # skip the initial HumanMessage
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps_html += f"<li><b>Tool call:</b> {tc['name']}({tc['args']})</li>"
        elif msg.type == "tool":
            steps_html += f"<li><b>Result:</b> <code>{str(msg.content)[:200]}</code></li>"
        elif msg.content:
            steps_html += f"<li><b>Answer:</b> {msg.content[:300]}</li>"
    steps_html += "</ol>"

    await flyte.report.replace.aio(
        f"<h2>ReAct Agent</h2>"
        f"<p><b>Task:</b> {request}</p>"
        f"<h2>Final Answer</h2>{md_to_html(answer)}"
        f"<hr/>{steps_html}"
        f'<hr/><h2>Agent Graph</h2><img src="data:image/png;base64,{img_b64}" />'
    )
    await flyte.report.flush.aio()

    return answer


