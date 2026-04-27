"""
Claude research agent — ReAct loop with web search.

Uses Claude's native tool-use API to implement a ReAct (Reason → Act → Observe)
agent loop. No agent framework needed — Claude's tool-use protocol is ReAct:
  1. Send prompt to Claude
  2. Claude returns tool_use blocks → execute the tools
  3. Send tool results back as tool_result → Claude incorporates them
  4. Repeat until Claude responds with text (stop_reason == "end_turn")

See: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview
"""

import json
import logging
import os

import anthropic
import flyte
from tavily import TavilyClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Anthropic tool-use format)
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web for current information on a topic. "
        "Returns titles, snippets, and URLs from top results."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            }
        },
        "required": ["query"],
    },
}

TOOLS = [SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

@flyte.trace
def execute_search(query: str) -> str:
    """Run a Tavily web search and format results."""
    log.info(f"  🔍 Searching: {query}")
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    results = tavily.search(query=query, max_results=3)

    formatted = ""
    for r in results.get("results", []):
        formatted += f"- {r['title']}: {r['content'][:300]}\n  {r['url']}\n\n"
    return formatted or "No results found."


def execute_tool(name: str, input_data: dict) -> str:
    """Dispatch a tool call to its implementation."""
    if name == "web_search":
        return execute_search(input_data["query"])
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_research_agent(
    topic: str,
    max_searches: int = 3,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Run a Claude research agent with web search on a single topic.

    The ReAct loop:
      Reason → Claude decides what to search
      Act    → execute_search via Tavily
      Observe → results sent back to Claude
      Repeat → until Claude writes the final summary
    """
    client = anthropic.AsyncAnthropic()

    system_prompt = (
        f"You are a research agent. Your job is to thoroughly research a topic "
        f"by searching the web. Use the web_search tool up to {max_searches} times "
        f"to gather information from different angles. After gathering enough "
        f"information, write a clear research summary with key findings and sources."
    )

    messages = [{"role": "user", "content": f"Research this topic: {topic}"}]
    search_count = 0

    for iteration in range(max_searches + 2):  # +2 for safety margin
        log.info(f"  🤖 Agent iteration {iteration + 1}")

        # Remove tools once search limit is hit — forces Claude to summarize
        available_tools = TOOLS if search_count < max_searches else []

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=available_tools,
            messages=messages,
        )

        # If Claude is done (no tool calls), extract the final text
        if response.stop_reason == "end_turn":
            text_blocks = [b.text for b in response.content if b.type == "text"]
            log.info(f"  ✅ Agent done after {iteration + 1} iteration(s)")
            return "\n".join(text_blocks)

        # Process tool calls
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                search_count += 1
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        messages.append({"role": "user", "content": tool_results})

    # Fallback: extract whatever text we have
    text_blocks = [b.text for b in response.content if b.type == "text"]
    return "\n".join(text_blocks) or "Research incomplete — max iterations reached."


# ---------------------------------------------------------------------------
# Pipeline helpers (planning, synthesis, quality check)
# ---------------------------------------------------------------------------

async def call_claude(
    prompt: str,
    system: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Simple Claude call without tools — for planning, synthesis, etc."""
    client = anthropic.AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        system=system or "You are a helpful research assistant.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def plan_topics(
    query: str, num_topics: int = 3, model: str = "claude-sonnet-4-6",
) -> list[str]:
    """Use Claude to break a research query into focused sub-topics."""
    response = await call_claude(
        prompt=(
            f"Break this research question into exactly {num_topics} focused sub-topics. "
            f"Return ONLY a JSON array of strings, nothing else.\n\n"
            f"Question: {query}"
        ),
        model=model,
    )
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        topics = json.loads(text)
        return topics[:num_topics]
    except json.JSONDecodeError:
        return [query]


async def synthesize_reports(
    query: str,
    results: list[dict],
    model: str = "claude-sonnet-4-6",
) -> str:
    """Combine sub-topic research reports into a unified synthesis."""
    sections = "\n\n---\n\n".join(
        f"## {r['topic']}\n\n{r['report']}" for r in results
    )
    return await call_claude(
        prompt=(
            f"You have research reports on sub-topics of this question:\n\n{query}\n\n"
            f"Sub-topic reports:\n\n{sections}\n\n"
            f"Write a comprehensive report that synthesizes all findings. "
            f"Organize by theme, highlight connections between sub-topics, "
            f"and end with key takeaways."
        ),
        model=model,
    )


async def evaluate_quality(
    query: str,
    synthesis: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[int, list[str]]:
    """Evaluate report quality and identify gaps. Returns (score, gaps)."""
    response = await call_claude(
        prompt=(
            f"Evaluate this research report for the question: {query}\n\n"
            f"Report:\n{synthesis}\n\n"
            f"Rate the report quality from 1-10 and identify any gaps or missing perspectives. "
            f'Return JSON: {{"score": <int>, "gaps": [<string>, ...]}}\n'
            f"If the report is comprehensive (score >= 8) or there are no significant gaps, "
            f"return an empty gaps list."
        ),
        model=model,
    )
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        evaluation = json.loads(text)
        return evaluation.get("score", 8), evaluation.get("gaps", [])
    except json.JSONDecodeError:
        return 8, []
