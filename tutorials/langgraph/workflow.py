"""
Multi-agent research workflow — fans out parallel researchers, then synthesizes.

Architecture:
    1. Planner (Flyte task): LLM splits query into sub-topics
    2. Researchers (parallel Flyte tasks): Each runs a LangGraph research graph
    3. Synthesizer (Flyte task): LLM combines all findings into final report

Each task shows up separately in the Flyte UI for tracking and expansion.

Usage:
    python -m workflow --local --query "Compare quantum computing approaches: superconducting vs trapped ion vs photonic"
"""

import json
import asyncio
import logging
import flyte
from langchain_openai import ChatOpenAI
from config import base_env, OPENAI_API_KEY, TAVILY_API_KEY
from graph import build_research_graph

logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
log = logging.getLogger(__name__)

env = base_env
MODEL = "gpt-4.1-nano"


# ------------------------------------------------------------------
# Task 1: Split query into sub-topics
# ------------------------------------------------------------------

@env.task
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
    log.info(f"Planned {len(topics)} sub-topics: {topics}")
    return topics[:num_topics]


# ------------------------------------------------------------------
# Task 2: Research a single sub-topic (runs in parallel)
# ------------------------------------------------------------------

@env.task
async def research_topic(topic: str, max_searches: int = 2) -> dict:
    """Run the LangGraph research graph on a single sub-topic."""
    from langchain_core.messages import HumanMessage
    log.info(f"Researching: {topic}")
    graph = build_research_graph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        max_searches=max_searches,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=f"Research this topic: {topic}")]})
    report = result["messages"][-1].content
    log.info(f"Done: {topic}")
    return {"topic": topic, "report": report}


# ------------------------------------------------------------------
# Task 3: Synthesize all sub-topic reports into final report
# ------------------------------------------------------------------

@env.task
async def synthesize_reports(query: str, reports: list[dict]) -> dict:
    """Combine sub-topic reports into a final comprehensive report."""
    llm = ChatOpenAI(model=MODEL, api_key=OPENAI_API_KEY)
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
    log.info(f"Final report synthesized from {len(reports)} sub-topics")
    return {"query": query, "report": response.content, "sub_reports": reports}


# ------------------------------------------------------------------
# Orchestrator: plan → fan-out research → synthesize
# ------------------------------------------------------------------

@env.task
async def research_workflow(query: str, num_topics: int = 3, max_searches: int = 2) -> dict:
    """
    Full research workflow:
    1. Plan sub-topics
    2. Research each in parallel (separate Flyte tasks)
    3. Synthesize into final report
    """
    log.info(f"Starting research workflow: {query}")

    # Step 1: Plan
    topics = await plan_research(query, num_topics)

    # Step 2: Research in parallel
    reports = await asyncio.gather(*[
        research_topic(topic, max_searches) for topic in topics
    ])

    # Step 3: Synthesize
    result = await synthesize_reports(query, list(reports))

    log.info("Workflow complete")
    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent research workflow")
    parser.add_argument("--local", action="store_true", help="Run locally with flyte.init()")
    parser.add_argument("--query", type=str, required=True, help="Research question")
    parser.add_argument("--num-topics", type=int, default=3, help="Number of sub-topics to research")
    parser.add_argument("--max-searches", type=int, default=2, help="Max searches per sub-topic")
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"Query: {args.query}")
    execution = flyte.run(
        research_workflow,
        query=args.query,
        num_topics=args.num_topics,
        max_searches=args.max_searches,
    )
    log.info(f"Execution: {execution.name} | URL: {execution.url}")