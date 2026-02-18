from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

import flyte

load_dotenv()

env = flyte.TaskEnvironment(
    name="langgraph_env",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=1, memory="1Gi"),
)

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

tools = [add, multiply,]

@env.task
async def agent(request: str) -> str:
    """Run a LangGraph ReAct agent with math tools."""
    llm = ChatOpenAI(model="gpt-4o-mini")
    react_agent = create_react_agent(llm, tools)

    result = await react_agent.ainvoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content

#flyte run langgraph_react_agent.py agent --request "What is 12 * 7 plus 3?"