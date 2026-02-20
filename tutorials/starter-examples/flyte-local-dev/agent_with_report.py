from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import flyte
import flyte.report

load_dotenv()

env = flyte.TaskEnvironment(name="agent_report")


@tool
@flyte.trace
async def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
@flyte.trace
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


tools = [add, multiply]


@env.task(report=True)
async def agent(request: str) -> str:
    """ReAct agent that logs its reasoning to a Flyte report."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    react_agent = create_react_agent(llm, tools)
    result = await react_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]}
    )

    # Build an HTML trace of the conversation
    messages = result["messages"]
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
                f"Tool result: <code>{content}</code></p>"
            )

    await flyte.report.replace.aio("\n".join(html_parts))
    await flyte.report.flush.aio()

    return result["messages"][-1].content
