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
import logging

import markdown

import flyte
import flyte.report
from config import base_env
from models import TopicReport, QualityResult, PipelineResult
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

REPORT_CSS = """\
<style>
  .report { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; color: #1a1a2e; max-width: 800px; margin: 0 auto; padding: 20px; }
  .report h1, .report h2, .report h3 { color: #16213e; margin-top: 1.4em; }
  .report h2 { border-bottom: 2px solid #e2e8f0; padding-bottom: 0.3em; }
  .report p { margin: 0.8em 0; }
  .report ul, .report ol { padding-left: 1.5em; }
  .report li { margin: 0.4em 0; }
  .report a { color: #2563eb; text-decoration: none; }
  .report a:hover { text-decoration: underline; }
  .report code { background: #f1f5f9; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }
  .report pre { background: #f1f5f9; padding: 1em; border-radius: 8px; overflow-x: auto; }
  .report hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5em 0; }
  .report table { border-collapse: collapse; width: 100%; margin: 1em 0; }
  .report th, .report td { border: 1px solid #e2e8f0; padding: 0.6em 1em; text-align: left; }
  .report th { background: #f8fafc; font-weight: 600; }
  .badge { display: inline-block; padding: 0.2em 0.7em; border-radius: 12px;
           font-size: 0.85em; font-weight: 600; }
  .badge-blue { background: #dbeafe; color: #1e40af; }
  .badge-green { background: #dcfce7; color: #166534; }
  .badge-amber { background: #fef3c7; color: #92400e; }
  .status { padding: 12px 16px; border-radius: 8px; margin: 1em 0; }
  .status-running { background: #eff6ff; border-left: 4px solid #3b82f6; }
  .status-done { background: #f0fdf4; border-left: 4px solid #22c55e; }
</style>
"""


def styled(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Flyte tasks — each step is visible in the UI while running
# ------------------------------------------------------------------

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 3) -> TopicReport:
    """Run the Claude research agent on a single sub-topic."""
    log.info(f"📄 Researching: {topic}")

    await flyte.report.replace.aio(styled(
        f'<h2>{topic}</h2>'
        f'<div class="status status-running">Searching the web with Claude...</div>'
    ))
    await flyte.report.flush.aio()

    report = await run_research_agent(topic, max_searches=max_searches, model=MODEL)
    log.info(f"✅ Done: {topic}")

    await flyte.report.replace.aio(styled(
        f'<h2>{topic}</h2>{md_to_html(report)}'
    ))
    await flyte.report.flush.aio()

    return TopicReport(topic=topic, report=report)


@env.task(report=True)
async def synthesize(query: str, results: list[TopicReport]) -> str:
    """Combine sub-topic research reports into a unified synthesis."""
    log.info(f"📝 Synthesizing {len(results)} report(s)...")

    await flyte.report.replace.aio(styled(
        f'<h2>Synthesis</h2>'
        f'<div class="status status-running">Combining {len(results)} reports...</div>'
    ))
    await flyte.report.flush.aio()

    synthesis = await synthesize_reports(query, results, model=MODEL)
    log.info(f"📝 Synthesis complete: {len(synthesis)} chars")

    await flyte.report.replace.aio(styled(
        f'<h2>Synthesis</h2>{md_to_html(synthesis)}'
    ))
    await flyte.report.flush.aio()

    return synthesis


@env.task(report=True)
async def quality_check(query: str, synthesis: str) -> QualityResult:
    """Evaluate report quality and identify gaps."""
    log.info("🔎 Evaluating quality...")

    await flyte.report.replace.aio(styled(
        '<h2>Quality Check</h2>'
        '<div class="status status-running">Evaluating report quality...</div>'
    ))
    await flyte.report.flush.aio()

    result = await evaluate_quality(query, synthesis, model=MODEL)
    log.info(f"📊 Score: {result.score}/10, Gaps: {len(result.gaps)}")

    badge_class = "badge-green" if result.score >= 8 else "badge-amber"
    gap_html = "".join(f"<li>{g}</li>" for g in result.gaps) if result.gaps else "<li>None</li>"
    await flyte.report.replace.aio(styled(
        f'<h2>Quality Check</h2>'
        f'<p><span class="badge {badge_class}">{result.score}/10</span></p>'
        f'<p><b>Gaps:</b></p><ul>{gap_html}</ul>'
    ))
    await flyte.report.flush.aio()

    return result


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 3,
    max_iterations: int = 2,
) -> PipelineResult:
    """
    Research pipeline:
    1. Plan sub-topics (Claude)
    2. Fan out research to parallel Flyte tasks (each runs a Claude ReAct agent)
    3. Synthesize results (Flyte task)
    4. Quality check (Flyte task) — if gaps found, loop back to step 2
    5. Finalize when quality >= 8 or max iterations reached
    """
    log.info(f"🚀 Starting research pipeline: {query}")

    await flyte.report.replace.aio(styled(
        f'<h2>Research Pipeline</h2>'
        f'<p><b>Query:</b> {query}</p>'
        f'<div class="status status-running">Planning sub-topics...</div>'
    ))
    await flyte.report.flush.aio()

    # -- Step 1: Plan sub-topics --
    log.info(f"📋 Planning {num_topics} sub-topics...")
    topics = await plan_topics(query, num_topics=num_topics, model=MODEL)
    log.info(f"📋 Sub-topics: {topics}")

    all_results: list[TopicReport] = []
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log.info(f"\n{'='*60}")
        log.info(f"🔄 Iteration {iteration}/{max_iterations}")
        log.info(f"{'='*60}")

        await flyte.report.replace.aio(styled(
            f'<h2>Research Pipeline</h2>'
            f'<p><b>Query:</b> {query}</p>'
            f'<div class="status status-running">'
            f'Iteration {iteration}/{max_iterations} '
            f'— researching {len(topics)} topic(s)...</div>'
        ))
        await flyte.report.flush.aio()

        # -- Step 2: Fan out research to parallel Flyte tasks --
        log.info(f"🔍 Researching {len(topics)} topic(s) in parallel...")
        research_coros = [
            research_topic.override(short_name=f"research-{i}")(
                topic, max_searches
            )
            for i, topic in enumerate(topics)
        ]
        new_results = await asyncio.gather(*research_coros)

        all_results.extend(new_results)
        for r in new_results:
            log.info(f"  📄 {r.topic}: {len(r.report)} chars")

        # -- Step 3: Synthesize --
        synthesis = await synthesize.override(
            short_name=f"synthesize-{iteration}"
        )(query, all_results)

        # -- Step 4: Quality check --
        evaluation = await quality_check.override(
            short_name=f"quality-{iteration}"
        )(query, synthesis)
        score, gaps = evaluation.score, evaluation.gaps
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
    badge_class = "badge-green" if score >= 8 else "badge-amber"
    await flyte.report.replace.aio(styled(
        f'<h2>Research Report</h2>'
        f'<p><b>Query:</b> {query}</p>'
        f'<p><span class="badge {badge_class}">{score}/10</span> '
        f'after {iteration} iteration(s)</p>'
        f'<hr/>{md_to_html(synthesis)}'
    ))
    await flyte.report.flush.aio()

    log.info(f"\n🏁 Pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return PipelineResult(
        query=query,
        report=synthesis,
        sub_reports=all_results,
        score=score,
        iterations=iteration,
    )
