"""
Simple Web Search Agent

This is a basic agent that searches the web using DuckDuckGo.
Perfect for tutorials and simple use cases.

For advanced features like reflexion and quality checking,
see web_search_reflexion_agent.py
"""

import json
from openai import AsyncOpenAI
from dataclasses import dataclass

from utils.decorators import agent, agent_tools
from utils.plan_executor import execute_tool_plan, parse_plan_from_response
from config import base_env, OPENAI_API_KEY

# Import tools to register them
import tools.web_search_tools

# ----------------------------------
# Agent Configuration
# ----------------------------------
WEB_SEARCH_AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 800,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WebSearchAgentResult:
    """Result from web search agent execution"""
    final_result: str      # Raw search results
    steps: str             # JSON string of steps taken
    summary: str = ""      # Brief summary of findings
    error: str = ""        # Empty if no error


# ----------------------------------
# Simple Web Search Agent
# ----------------------------------

env = base_env

@env.task
@agent("web_search")
async def web_search_agent(task: str) -> WebSearchAgentResult:
    """
    Simple web search agent - searches the web and returns results.

    This agent:
    1. Takes a search query
    2. Uses DuckDuckGo to search
    3. Returns the results

    Args:
        task (str): The search query or task description

    Returns:
        WebSearchAgentResult: Search results and summary
    """
    print(f"[Web Search Agent] Searching for: {task}")

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Get available tools
    toolset = agent_tools["web_search"]
    tool_list = "\n".join([f"{name}: {fn.__doc__.strip()}" for name, fn in toolset.items()])

    system_msg = f"""You are a web search agent. You can search the web using DuckDuckGo.

Available Tools:
{tool_list}

CRITICAL: Respond with ONLY a valid JSON array, nothing else.
Return a JSON array of tool calls in this exact format:
[
  {{"tool": "duck_duck_go", "args": ["search query here", 5, "us-en", "moderate", null], "reasoning": "Why you're doing this search"}}
]

RULES:
1. Start with [ and end with ]
2. No markdown code blocks (no ```)
3. No extra text before or after the JSON
4. Always include a "reasoning" field
5. Keep it simple - usually one search is enough
"""

    # Ask LLM to create a search plan
    response = await client.chat.completions.create(
        model=WEB_SEARCH_AGENT_CONFIG["model"],
        temperature=WEB_SEARCH_AGENT_CONFIG["temperature"],
        max_tokens=WEB_SEARCH_AGENT_CONFIG["max_tokens"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Search for Python async tutorials"},
            {"role": "assistant", "content": '[{"tool": "duck_duck_go", "args": ["Python async tutorial", 5, "us-en", "moderate", null], "reasoning": "Searching for Python async tutorials"}]'},
            {"role": "user", "content": task}
        ]
    )

    # Parse and execute the plan
    raw_plan = response.choices[0].message.content
    plan = parse_plan_from_response(raw_plan)
    result = await execute_tool_plan(plan, agent="web_search")

    print(f"[Web Search Agent] Search complete")

    # Get the search results
    full_result = str(result.get("final_result", ""))

    # Create a simple summary using the LLM
    if full_result:
        summary_prompt = f"""Summarize these search results in 2-3 sentences:

{full_result[:2000]}

Keep it brief and informative."""

        summary_response = await client.chat.completions.create(
            model=WEB_SEARCH_AGENT_CONFIG["model"],
            temperature=0.3,
            max_tokens=200,
            messages=[{"role": "user", "content": summary_prompt}]
        )

        summary = summary_response.choices[0].message.content
    else:
        summary = "No results found"

    print(f"[Web Search Agent] Summary: {summary[:100]}...")

    return WebSearchAgentResult(
        final_result=full_result,
        steps=json.dumps(result.get("steps", [])),
        summary=summary,
        error=result.get("error", "")
    )