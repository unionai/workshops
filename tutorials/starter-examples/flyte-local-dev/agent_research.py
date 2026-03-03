"""Multi-agent research — demonstrates parallel agents with asyncio.gather in Flyte."""

import asyncio
import json

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
    """
    Evaluate a math expression with numbers and operators only.
    Example: '68000000 * 0.1' not 'population * 0.1'.
     """
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


# -- Parallel research tasks --------------------------------------------------

@env.task(cache="auto", retries=2)
async def search_topic(topic: str) -> str:
    """Research a single sub-topic using the ReAct agent."""
    answer, _ = await run_agent(topic)
    return json.dumps({"topic": topic, "answer": answer})


@env.task(cache="auto", retries=2, report=True)
async def research(request: str) -> str:
    """Multi-agent research: plan sub-topics, search in parallel, synthesize."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    # 1. Plan — split the request into sub-questions
    plan_response = await llm.ainvoke(
        f"Break this research question into exactly 3 focused sub-questions. "
        f"Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {request}"
    )
    try:
        topics = json.loads(plan_response.content)
    except json.JSONDecodeError:
        topics = [request]

    # 2. Fan out — research each sub-topic in parallel
    result_jsons = await asyncio.gather(*[search_topic(t) for t in topics])
    results = [json.loads(r) for r in result_jsons]

    # 3. Synthesize — combine findings into a final report
    sections = "\n\n".join(f"### {r['topic']}\n{r['answer']}" for r in results)
    synthesis = await llm.ainvoke(
        f"You researched this question: {request}\n\n"
        f"Here are findings from parallel research agents:\n\n{sections}\n\n"
        f"Write a concise final report that synthesizes all findings. "
        f"Highlight key connections and end with takeaways."
    )

    # 4. Report
    html = f"<h2>Research Report</h2><p><b>Query:</b> {request}</p><hr>"
    for r in results:
        html += f"<h3>{r['topic']}</h3><p>{r['answer']}</p>"
    html += f"<hr><h3>Synthesis</h3><p>{synthesis.content}</p>"
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return synthesis.content


# Local:  flyte run --local --tui agent_research.py research --request "Compare the tech industries of Japan, South Korea, and Germany"
# Remote: flyte run agent_research.py research --request "Compare the tech industries of Japan, South Korea, and Germany"