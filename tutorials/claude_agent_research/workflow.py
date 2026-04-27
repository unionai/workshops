"""
Research pipeline — Claude agent with tool use, Flyte provides compute.

Same architecture as the LangGraph version, but replaces LangGraph + OpenAI with
Claude's native tool-use loop:

  plan (Claude) → research (fan-out to Flyte tasks, each running a Claude agent)
                    → synthesize (Claude) → quality check (Claude) → loop or finalize

Usage:
    # Local
    flyte run --local --tui workflow.py research_pipeline --query "Compare quantum computing approaches"

    # Remote
    flyte run workflow.py research_pipeline --query "Compare quantum computing approaches"
"""

import asyncio
import json
import logging
import os

import markdown

import flyte
import flyte.report
from config import base_env
from agent import run_research_agent, plan_topics, synthesize_reports, evaluate_quality

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("agent").setLevel(logging.INFO)

env = base_env
MODEL = "claude-sonnet-4-6"


def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Flyte task: research a single topic using Claude agent with tools
# ------------------------------------------------------------------

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 3) -> str:
    """Run the Claude research agent on a single sub-topic."""
    log.info(f"📄 [Research Task] Starting: {topic}")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running Claude agent...</p>")
    await flyte.report.flush.aio()

    report = await run_research_agent(topic, max_searches=max_searches, model=MODEL)
    log.info(f"✅ [Research Task] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return json.dumps({"topic": topic, "report": report})


# ------------------------------------------------------------------
# Orchestrator: plans, fans out research, synthesizes, evaluates
# ------------------------------------------------------------------

@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 3,
    max_iterations: int = 2,
) -> str:
    """
    Research pipeline:
    1. Claude plans sub-topics
    2. Fan out to Flyte tasks — each runs a Claude agent with web search
    3. Claude synthesizes results
    4. Claude evaluates quality — if gaps, loop back to step 2
    5. Repeat until quality is good or max iterations reached
    """
    log.info(f"🚀 Starting research pipeline: {query}")

    await flyte.report.replace.aio(
        f"<h2>🔬 Research Pipeline</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p>Planning sub-topics...</p>"
    )
    await flyte.report.flush.aio()

    # -- Step 1: Plan sub-topics --
    log.info(f"📋 Planning {num_topics} sub-topics...")
    topics = await plan_topics(query, num_topics=num_topics, model=MODEL)
    log.info(f"📋 Sub-topics: {topics}")

    all_results = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log.info(f"\n{'='*60}")
        log.info(f"🔄 Iteration {iteration}/{max_iterations}")
        log.info(f"{'='*60}")

        await flyte.report.replace.aio(
            f"<h2>🔬 Research Pipeline</h2>"
            f"<p><b>Query:</b> {query}</p>"
            f"<p>Iteration {iteration}/{max_iterations} — researching {len(topics)} topic(s)...</p>"
        )
        await flyte.report.flush.aio()

        # -- Step 2: Fan out research to Flyte tasks --
        log.info(f"🔍 Researching {len(topics)} topic(s) in parallel...")
        research_coros = [
            research_topic.override(short_name=f"research-{i}")(topic, max_searches)
            for i, topic in enumerate(topics)
        ]
        results_json = await asyncio.gather(*research_coros)

        new_results = [json.loads(r) for r in results_json]
        all_results.extend(new_results)
        for r in new_results:
            log.info(f"  📄 {r['topic']}: {len(r['report'])} chars")

        # -- Step 3: Synthesize --
        log.info(f"📝 Synthesizing {len(all_results)} report(s)...")
        synthesis = await synthesize_reports(query, all_results, model=MODEL)

        # -- Step 4: Quality check --
        log.info(f"🔎 Evaluating quality...")
        score, gaps = await evaluate_quality(query, synthesis, model=MODEL)
        log.info(f"📊 Score: {score}/10, Gaps: {len(gaps)}")

        if not gaps or iteration >= max_iterations:
            if gaps:
                log.info(f"⏹️  Max iterations reached, finishing with score {score}/10")
            else:
                log.info(f"✅ Quality sufficient ({score}/10), finalizing")
            break

        # -- Loop: research the gaps --
        log.info(f"🔄 Gaps found, researching further: {gaps}")
        topics = gaps

    # -- Final report --
    await flyte.report.replace.aio(
        f"<h2>📊 Research Report</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p><b>Quality:</b> {score}/10 after {iteration} iteration(s)</p>"
        f"<hr/>{md_to_html(synthesis)}"
    )
    await flyte.report.flush.aio()

    log.info(f"\n🏁 Pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return json.dumps({
        "query": query,
        "report": synthesis,
        "sub_reports": all_results,
        "score": score,
        "iterations": iteration,
    })


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------

if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(
        research_pipeline,
        query="Compare quantum computing approaches: superconducting vs trapped ion",
        num_topics=3,
        max_searches=3,
        max_iterations=2,
    )
    print(f"View at: {run.url}")
    run.wait()
    print(f"Result: {run.outputs()}")
