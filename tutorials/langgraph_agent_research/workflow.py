"""
Research pipeline workflow — LangGraph controls the pipeline, Flyte provides the compute.

Each pipeline step is a separate Flyte task visible in the UI:
- plan_topics: break query into sub-topics
- research_topic: ReAct agent researches one sub-topic (parallel via Send)
- synthesize: combine sub-topic reports into a unified synthesis
- quality_check: evaluate quality and identify gaps

LangGraph manages the flow: plan → research → synthesize → quality check → loop

Usage:
    # Local
    flyte run --local --tui workflow.py research_pipeline --query "Compare quantum computing approaches"

    # Remote
    flyte run workflow.py research_pipeline --query "Compare quantum computing approaches"
"""

import json
import os
import base64
import logging
import markdown

import flyte
import flyte.report
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import base_env
from models import TopicReport, QualityResult, PipelineResult
from graph import build_pipeline_graph, build_research_subgraph

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
# Flyte tasks — each step is visible in the UI while running
# ------------------------------------------------------------------

@env.task(report=True)
async def plan_topics(query: str, num_topics: int = 3) -> list[str]:
    """Break a research query into focused sub-topics."""
    log.info(f"Planning {num_topics} sub-topics for: {query}")

    await flyte.report.replace.aio(
        f"<h2>Planning</h2><p>Breaking query into {num_topics} sub-topics...</p>"
    )
    await flyte.report.flush.aio()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=MODEL, api_key=openai_api_key)

    response = llm.invoke(
        f"Break this research question into exactly {num_topics} focused sub-topics. "
        f"Return ONLY a JSON array of strings, nothing else.\n\nQuestion: {query}"
    )
    try:
        topics = json.loads(response.content)
    except json.JSONDecodeError:
        topics = [query]

    topics = topics[:num_topics]
    log.info(f"Sub-topics: {topics}")

    topic_html = "".join(f"<li>{t}</li>" for t in topics)
    await flyte.report.replace.aio(
        f"<h2>Planning</h2><p>Sub-topics:</p><ul>{topic_html}</ul>"
    )
    await flyte.report.flush.aio()

    return topics


@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2) -> TopicReport:
    """Run the ReAct research agent on a single sub-topic."""
    log.info(f"[Research Task] Starting: {topic}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running searches...</p>")
    await flyte.report.flush.aio()

    graph = build_research_subgraph(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        max_searches=max_searches,
        model=MODEL,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=f"Research this topic: {topic}")]})
    report = result["messages"][-1].content
    log.info(f"[Research Task] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return TopicReport(topic=topic, report=report)


@env.task(report=True)
async def synthesize(query: str, results: list[TopicReport]) -> str:
    """Combine sub-topic research reports into a unified synthesis."""
    log.info(f"Synthesizing {len(results)} report(s)...")

    await flyte.report.replace.aio(
        f"<h2>Synthesis</h2><p>Combining {len(results)} reports...</p>"
    )
    await flyte.report.flush.aio()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=MODEL, api_key=openai_api_key)

    sections = "\n\n---\n\n".join(
        f"## {r.topic}\n\n{r.report}" for r in results
    )

    response = llm.invoke(
        f"You have research reports on sub-topics of this question:\n\n{query}\n\n"
        f"Sub-topic reports:\n\n{sections}\n\n"
        f"Write a comprehensive report that synthesizes all findings. "
        f"Organize by theme, highlight connections between sub-topics, "
        f"and end with key takeaways."
    )
    synthesis = response.content
    log.info(f"Synthesis complete: {len(synthesis)} chars")

    await flyte.report.replace.aio(f"<h2>Synthesis</h2>{md_to_html(synthesis)}")
    await flyte.report.flush.aio()

    return synthesis


@env.task(report=True)
async def quality_check(query: str, synthesis: str) -> QualityResult:
    """Evaluate report quality and identify gaps."""
    log.info("Evaluating quality...")

    await flyte.report.replace.aio(
        "<h2>Quality Check</h2><p>Evaluating report quality...</p>"
    )
    await flyte.report.flush.aio()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model=MODEL, api_key=openai_api_key)

    response = llm.invoke(
        f'Evaluate this research report for the question: {query}\n\n'
        f'Report:\n{synthesis}\n\n'
        f'Rate the report quality from 1-10 and identify any gaps or missing perspectives. '
        f'Return JSON: {{"score": <int>, "gaps": [<string>, ...]}}\n'
        f'If the report is comprehensive (score >= 8) or there are no significant gaps, '
        f'return an empty gaps list.'
    )

    try:
        evaluation = json.loads(response.content)
        score = evaluation.get("score", 8)
        gaps = evaluation.get("gaps", [])
    except json.JSONDecodeError:
        score = 8
        gaps = []

    result = QualityResult(score=score, gaps=gaps)
    log.info(f"Score: {result.score}/10, Gaps: {len(result.gaps)}")

    gap_html = "".join(f"<li>{g}</li>" for g in result.gaps) if result.gaps else "<li>None</li>"
    await flyte.report.replace.aio(
        f"<h2>Quality Check</h2>"
        f"<p><b>Score:</b> {result.score}/10</p>"
        f"<p><b>Gaps:</b></p><ul>{gap_html}</ul>"
    )
    await flyte.report.flush.aio()

    return result


# ------------------------------------------------------------------
# Orchestrator: runs the LangGraph pipeline, backed by Flyte tasks
# ------------------------------------------------------------------

@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 2,
    max_iterations: int = 2,
) -> PipelineResult:
    """
    Research pipeline workflow:
    1. LangGraph plans sub-topics via plan_topics Flyte task
    2. LangGraph fans out research via Send → each dispatches to research_topic Flyte task
    3. LangGraph synthesizes results via synthesize Flyte task
    4. LangGraph evaluates quality via quality_check Flyte task
    5. If gaps found, loops back to step 2
    """
    log.info(f"Starting research pipeline: {query}")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Build the pipeline graph, passing all Flyte tasks as compute backends
    pipeline = build_pipeline_graph(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        plan_task=plan_topics,
        research_task=research_topic,
        synthesize_task=synthesize,
        quality_check_task=quality_check,
        model=MODEL,
    )

    # Visualize the graphs in report tabs
    graph_tab = flyte.report.get_tab("Agent Graphs")

    png_bytes = pipeline.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    graph_tab.log(f"""\
<h2>Research Pipeline</h2>\
<img src="data:image/png;base64,{img_b64}" alt="Research pipeline" />""")

    subgraph = build_research_subgraph(openai_api_key, tavily_api_key, max_searches, model=MODEL)
    sub_png = subgraph.get_graph().draw_mermaid_png()
    sub_b64 = base64.b64encode(sub_png).decode()
    graph_tab.log(f"""\
<h2>Research Agent (ReAct)</h2>\
<img src="data:image/png;base64,{sub_b64}" alt="ReAct research agent" />""")
    await flyte.report.flush.aio()

    # Run the pipeline — LangGraph controls the flow, Flyte tasks run the compute
    result = await pipeline.ainvoke({
        "query": query,
        "num_topics": num_topics,
        "max_searches": max_searches,
        "max_iterations": max_iterations,
        "iteration": 0,
        "topics": [],
        "research_results": [],
        "synthesis": "",
        "score": 0,
        "gaps": [],
        "final_report": "",
    })

    # Build the final report
    final_report = result["final_report"]
    sub_reports = [TopicReport(**r) for r in result["research_results"]]
    score = result.get("score", 0)
    iteration = result.get("iteration", 1) - 1

    await flyte.report.replace.aio(f"""\
<h2>Research Report</h2>\
<p><b>Query:</b> {query}</p>\
<p><b>Quality:</b> {score}/10 after {iteration} iteration(s)</p>\
<hr/>{md_to_html(final_report)}""")
    await flyte.report.flush.aio()

    log.info(f"Research pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return PipelineResult(
        query=query,
        report=final_report,
        sub_reports=sub_reports,
        score=score,
        iterations=iteration,
    )
