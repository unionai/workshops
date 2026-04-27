"""
Research pipeline — Claude agent with tool use, orchestrated by Flyte.

  plan (Claude) → research (parallel Flyte tasks, each a Claude ReAct agent)
               → synthesize (Flyte task) → quality check (Flyte task)
               → loop or finalize

Usage:
    # Local
    flyte run --local --tui workflow.py research_pipeline \
        --query "Compare quantum computing approaches"

    # Remote
    flyte run workflow.py research_pipeline \
        --query "Compare quantum computing approaches"
"""

import asyncio
import json
import logging

import markdown

import flyte
import flyte.report
from config import base_env
from agent import (
    run_research_agent,
    plan_topics,
    synthesize_reports,
    evaluate_quality,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("agent").setLevel(logging.INFO)

env = base_env
MODEL = "claude-haiku-4-5"

def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Flyte tasks — each step is visible in the UI while running
# ------------------------------------------------------------------

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 3) -> str:
    """Run the Claude research agent on a single sub-topic."""
    log.info(f"📄 Researching: {topic}")

    await flyte.report.replace.aio(
        f"<h2>Researching: {topic}</h2><p>Running Claude agent...</p>"
    )
    await flyte.report.flush.aio()

    report = await run_research_agent(topic, max_searches=max_searches, model=MODEL)
    log.info(f"✅ Done: {topic}")

    await flyte.report.replace.aio(
        f"<h2>{topic}</h2>{md_to_html(report)}"
    )
    await flyte.report.flush.aio()

    return json.dumps({"topic": topic, "report": report})


@env.task(report=True)
async def synthesize(query: str, results_json: str) -> str:
    """Combine sub-topic research reports into a unified synthesis."""
    results = json.loads(results_json)
    log.info(f"📝 Synthesizing {len(results)} report(s)...")

    await flyte.report.replace.aio(
        f"<h2>Synthesizing</h2><p>Combining {len(results)} reports...</p>"
    )
    await flyte.report.flush.aio()

    synthesis = await synthesize_reports(query, results, model=MODEL)
    log.info(f"📝 Synthesis complete: {len(synthesis)} chars")

    await flyte.report.replace.aio(
        f"<h2>Synthesis</h2>{md_to_html(synthesis)}"
    )
    await flyte.report.flush.aio()

    return synthesis


@env.task(report=True)
async def quality_check(query: str, synthesis: str) -> str:
    """Evaluate report quality and identify gaps."""
    log.info("🔎 Evaluating quality...")

    await flyte.report.replace.aio(
        "<h2>Quality Check</h2><p>Evaluating report...</p>"
    )
    await flyte.report.flush.aio()

    score, gaps = await evaluate_quality(query, synthesis, model=MODEL)
    log.info(f"📊 Score: {score}/10, Gaps: {len(gaps)}")

    gap_html = "".join(f"<li>{g}</li>" for g in gaps) if gaps else "<li>None</li>"
    await flyte.report.replace.aio(
        f"<h2>Quality Check</h2>"
        f"<p><b>Score:</b> {score}/10</p>"
        f"<p><b>Gaps:</b></p><ul>{gap_html}</ul>"
    )
    await flyte.report.flush.aio()

    return json.dumps({"score": score, "gaps": gaps})


# ------------------------------------------------------------------
# Orchestrator
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
    1. Plan sub-topics (Claude)
    2. Fan out research to parallel Flyte tasks (each runs a Claude ReAct agent)
    3. Synthesize results (Flyte task)
    4. Quality check (Flyte task) — if gaps found, loop back to step 2
    5. Finalize when quality >= 8 or max iterations reached
    """
    log.info(f"🚀 Starting research pipeline: {query}")

    await flyte.report.replace.aio(
        f"<h2>Research Pipeline</h2>"
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
            f"<h2>Research Pipeline</h2>"
            f"<p><b>Query:</b> {query}</p>"
            f"<p>Iteration {iteration}/{max_iterations} "
            f"— researching {len(topics)} topic(s)...</p>"
        )
        await flyte.report.flush.aio()

        # -- Step 2: Fan out research to parallel Flyte tasks --
        log.info(f"🔍 Researching {len(topics)} topic(s) in parallel...")
        research_coros = [
            research_topic.override(short_name=f"research-{i}")(
                topic, max_searches
            )
            for i, topic in enumerate(topics)
        ]
        results_json = await asyncio.gather(*research_coros)

        new_results = [json.loads(r) for r in results_json]
        all_results.extend(new_results)
        for r in new_results:
            log.info(f"  📄 {r['topic']}: {len(r['report'])} chars")

        # -- Step 3: Synthesize --
        synthesis = await synthesize.override(
            short_name=f"synthesize-{iteration}"
        )(query, json.dumps(all_results))

        # -- Step 4: Quality check --
        eval_json = await quality_check.override(
            short_name=f"quality-{iteration}"
        )(query, synthesis)
        evaluation = json.loads(eval_json)
        score, gaps = evaluation["score"], evaluation["gaps"]
        log.info(f"📊 Score: {score}/10, Gaps: {len(gaps)}")

        if not gaps or score >= 8 or iteration >= max_iterations:
            if iteration >= max_iterations and gaps:
                log.info(f"⏹️  Max iterations reached, finishing with score {score}/10")
            else:
                log.info(f"✅ Quality sufficient ({score}/10), finalizing")
            break

        # -- Loop: research the gaps (cap at num_topics to avoid explosion) --
        topics = gaps[:num_topics]
        log.info(
            f"🔄 Gaps found, researching {len(topics)} of {len(gaps)}: {topics}"
        )

    # -- Final report --
    await flyte.report.replace.aio(
        f"<h2>Research Report</h2>"
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
