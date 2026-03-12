"""
Multi-agent research workflow — LangGraph for agent logic, Flyte for orchestration.

Architecture:
    1. Plan (Flyte task): LLM splits query into sub-topics
    2. Research (parallel Flyte tasks): Each runs a LangGraph ReAct agent with web search
    3. Synthesize (Flyte task): LLM combines all findings into a final report

The planner and synthesizer are single-shot LLM calls — no graph needed.
The research agent is where LangGraph shines: a ReAct loop with tool calling.
Flyte handles parallel fan-out so each researcher runs on its own compute.

Usage:
    # Local
    flyte run --local --tui workflow.py research_workflow --query "Compare quantum computing approaches"

    # Remote (on Flyte cluster)
    flyte run workflow.py research_workflow --query "Compare quantum computing approaches"
"""

import json
import asyncio
import base64
import logging
import markdown

import flyte
import flyte.report
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import base_env, OPENAI_API_KEY, TAVILY_API_KEY
from graph import build_research_graph

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("graph").setLevel(logging.INFO)
logging.getLogger("tools.search").setLevel(logging.INFO)

env = base_env
MODEL = "gpt-4.1-nano"


def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Task 1: Plan — split query into sub-topics (single LLM call)
# ------------------------------------------------------------------

@env.task(report=True, cache="auto")
async def plan_research(query: str, num_topics: int = 3) -> list[str]:
    """Use LLM to break a broad query into focused sub-topics."""
    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
    response = llm.invoke(
        f"Break this research question into exactly {num_topics} focused sub-topics. "
        f"Return ONLY a JSON array of strings, nothing else.\n\n"
        f"Question: {query}"
    )
    try:
        topics = json.loads(response.content)
    except json.JSONDecodeError:
        topics = [query]

    topics = topics[:num_topics]
    log.info(f"[Plan] {len(topics)} sub-topics: {topics}")

    html = f"<h2>Research Plan</h2><p><b>Query:</b> {query}</p><h3>Sub-topics:</h3><ol>"
    for t in topics:
        html += f"<li>{t}</li>"
    html += "</ol>"
    await flyte.report.replace.aio(html)
    await flyte.report.flush.aio()

    return topics


# ------------------------------------------------------------------
# Task 2: Research — one topic per task, runs in parallel
# ------------------------------------------------------------------
# This is where LangGraph shines — the ReAct agent loop with tool
# calling needs a graph. Each topic gets its own Flyte task so it
# can scale to 100+ parallel researchers on separate compute.

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2) -> str:
    """Run the LangGraph ReAct research agent on a single sub-topic."""
    log.info(f"[Research] Starting: {topic}")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running searches...</p>")
    await flyte.report.flush.aio()

    graph = build_research_graph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        max_searches=max_searches,
        model=MODEL,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=f"Research this topic: {topic}")]})
    report = result["messages"][-1].content
    log.info(f"[Research] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return json.dumps({"topic": topic, "report": report})


# ------------------------------------------------------------------
# Task 3: Synthesize — combine all reports (single LLM call)
# ------------------------------------------------------------------

@env.task(report=True)
async def synthesize_reports(query: str, reports_json: str) -> str:
    """Combine sub-topic reports into a final comprehensive report."""
    reports = json.loads(reports_json)
    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)

    await flyte.report.replace.aio(f"<h2>Synthesizing {len(reports)} reports...</h2>")
    await flyte.report.flush.aio()

    sections = "\n\n---\n\n".join(
        f"## {r['topic']}\n\n{r['report']}" for r in reports
    )
    response = llm.invoke(
        f"You have research reports on sub-topics of this question:\n\n"
        f"{query}\n\n"
        f"Sub-topic reports:\n\n{sections}\n\n"
        f"Write a comprehensive final report that synthesizes all findings. "
        f"Organize by theme, highlight connections between sub-topics, "
        f"and end with key takeaways."
    )
    log.info(f"[Synthesize] Combined {len(reports)} reports")

    await flyte.report.replace.aio(f"<h2>Final Report</h2>{md_to_html(response.content)}")
    for r in reports:
        tab = flyte.report.get_tab(r["topic"][:30])
        tab.log(f"<h2>{r['topic']}</h2>{md_to_html(r['report'])}")
    await flyte.report.flush.aio()

    return json.dumps({"query": query, "report": response.content, "sub_reports": reports})


# ------------------------------------------------------------------
# Orchestrator: plan → fan-out research → synthesize
# ------------------------------------------------------------------

@env.task(report=True)
async def research_workflow(query: str, num_topics: int = 3, max_searches: int = 2) -> str:
    """
    Full research workflow:
    1. Plan sub-topics (single LLM call)
    2. Research each in parallel (separate Flyte tasks, each running a LangGraph ReAct agent)
    3. Synthesize into final report (single LLM call)
    """
    log.info(f"Starting research workflow: {query}")

    # Visualize the research agent graph in a report tab
    graph_tab = flyte.report.get_tab("Agent Graph")
    research_graph = build_research_graph(OPENAI_API_KEY, TAVILY_API_KEY, max_searches, model=MODEL)
    png_bytes = research_graph.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    graph_tab.log(
        f"<h2>Research Agent (ReAct)</h2>"
        f'<img src="data:image/png;base64,{img_b64}" alt="ReAct research agent" />'
    )
    await flyte.report.flush.aio()

    # 1. Plan
    topics = await plan_research(query, num_topics)

    # 2. Research in parallel — each topic gets its own Flyte task
    report_jsons = await asyncio.gather(*[
        research_topic(topic, max_searches) for topic in topics
    ])
    reports = [json.loads(r) for r in report_jsons]

    # 3. Synthesize
    result = await synthesize_reports(query, json.dumps(reports))

    log.info("Workflow complete")
    return result
