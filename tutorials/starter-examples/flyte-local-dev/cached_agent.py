from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import flyte

load_dotenv()

env = flyte.TaskEnvironment(name="cached_agent")


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


@env.task(cache="auto")
async def agent(request: str) -> str:
    """Cached ReAct agent — same question returns instantly."""
    print(f"Running agent for: {request}")
    llm = ChatOpenAI(model="gpt-4o-mini")
    react_agent = create_react_agent(llm, tools)
    result = await react_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content
