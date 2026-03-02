"""LangGraph research agent — demonstrates Flyte caching, tracing, and reports."""

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import flyte
import flyte.report

load_dotenv()

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "langchain-core", "langchain-openai", "langgraph", "ddgs", "python-dotenv",
)

env = flyte.TaskEnvironment(
    name="research_agent",
    image=image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
)


@tool
@flyte.trace
async def search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    ddgs = DDGS()
    results = ddgs.text(query, max_results=3)
    return "\n\n".join(f"{r['title']}: {r['body']}" for r in results)


@tool
@flyte.trace
async def calculate(expression: str) -> str:
    """Evaluate a math expression with numbers and operators only. Example: '68000000 * 0.1' not 'population * 0.1'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}. Use only numbers and operators."


tools = [search, calculate]


async def run_agent(request: str) -> tuple[str, list]:
    """Core agent logic — returns (answer, messages). Can be called from any context."""
    print(f"Processing: {request}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    react_agent = create_react_agent(llm, tools)
    result = await react_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]},
        config={"recursion_limit": 15},
    )
    return result["messages"][-1].content, result["messages"]


def build_report_html(request: str, messages: list) -> str:
    """Build an HTML reasoning trace from agent messages."""
    html_parts = [f"<h2>Agent Trace</h2><p><b>Request:</b> {request}</p><hr>"]
    for msg in messages:
        role = msg.type if hasattr(msg, "type") else "unknown"
        content = str(msg.content) if msg.content else ""
        if role == "human":
            html_parts.append(f"<p><b>User:</b> {content}</p>")
        elif role == "ai":
            html_parts.append(f"<p><b>Agent:</b> {content}</p>")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    html_parts.append(
                        f"<p style='margin-left:20px; color:#666;'>"
                        f"Tool call: <code>{tc['name']}({tc['args']})</code></p>"
                    )
        elif role == "tool":
            html_parts.append(
                f"<p style='margin-left:20px; color:#080;'>"
                f"Tool result: <code>{content[:500]}</code></p>"
            )
    return "\n".join(html_parts)


@env.task(cache="auto", retries=2, report=True)
async def agent(request: str) -> str:
    """Research agent — retries on API failure, cached, traced, with HTML report."""
    answer, messages = await run_agent(request)

    await flyte.report.replace.aio(build_report_html(request, messages))
    await flyte.report.flush.aio()

    return answer


# Local:  flyte run --local --tui agent_research.py agent --request "What is the population of France and what is 10% of it?"
# Remote: flyte run agent_research.py agent --request "What is the population of France and what is 10% of it?"