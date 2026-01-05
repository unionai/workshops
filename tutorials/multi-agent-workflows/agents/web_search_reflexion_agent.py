"""
This module defines the web_search_agent, which can search the web and fetch content from pages.
"""

import json
import flyte
from openai import AsyncOpenAI

# Import tools to register them
import tools.web_search_tools

from utils.decorators import agent, agent_tools
from utils.plan_executor import execute_tool_plan, parse_plan_from_response
from utils.summarizer import smart_summarize
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY

# ----------------------------------
# Agent-Specific Configuration
# ----------------------------------
WEB_SEARCH_AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.3,
    "max_tokens": 1000,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WebSearchAgentResult:
    """Result from web search agent execution"""
    final_result: str
    steps: str  # JSON string of steps taken
    summary: str = ""  # Concise summary for passing to other agents
    error: str = ""  # Empty if no error


# ----------------------------------
# Web Search Agent Task Environment
# ----------------------------------
env = base_env
# Future: If you need agent-specific dependencies, create separate environments:
# env = flyte.TaskEnvironment(
#     name="code_agent_env",
#     image=base_env.image.with_pip_packages(["numpy", "pandas"]),
#     secrets=base_env.secrets,
#     resources=flyte.Resources(cpu=2, mem="4Gi")
# )


@env.task
@agent("web_search")
async def web_search_agent(task: str) -> WebSearchAgentResult:
    """
    Web search agent that can search using Tavily (premium, curated results)
    and DuckDuckGo (broader coverage), and fetch webpage content.

    Strategy: Use BOTH Tavily and DuckDuckGo for comprehensive research.
    - Tavily: Better for research-quality, curated results with higher relevance
    - DuckDuckGo: Broader coverage, good for finding diverse perspectives

    Args:
        task (str): The web search task to perform.

    Returns:
        WebSearchAgentResult: The result of the search and the steps taken.
    """
    print(f"[Web Search Agent] Processing: {task}")

    # Initialize client inside task for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    toolset = agent_tools["web_search"]
    tool_list = "\n".join([f"{name}: {fn.__doc__.strip()}" for name, fn in toolset.items()])
    system_msg = f"""
You are a premium web search agent with access to multiple search engines.

Tools:
{tool_list}

SEARCH STRATEGY:
- Use BOTH tavily_search AND duck_duck_go for comprehensive coverage
- Tavily provides higher-quality, curated results (use first for best sources)
- DuckDuckGo provides broader coverage (use to fill gaps or find diverse perspectives)
- Combining both gives the most comprehensive research results

CRITICAL: You must respond with ONLY a valid JSON array, nothing else. No markdown, no explanations.
Return a JSON array of tool calls in this exact format:
[
  {{"tool": "tavily_search", "args": ["Python async frameworks", 5, false, false, "basic"], "reasoning": "Getting high-quality curated results from Tavily"}},
  {{"tool": "duck_duck_go", "args": ["Python async frameworks comparison", 5, "us-en", "moderate", null], "reasoning": "Getting broader coverage from DuckDuckGo to supplement Tavily results"}}
]

RULES:
1. Start your response with [ and end with ]
2. No markdown code blocks (no ```)
3. No extra text before or after the JSON
4. Always include a "reasoning" field for each step
5. For comprehensive research, use BOTH tavily_search and duck_duck_go
6. When using fetch_webpage, use the "href" or "url" from search results
"""

    # Call LLM to create plan using agent-specific config
    response = await client.chat.completions.create(
        model=WEB_SEARCH_AGENT_CONFIG["model"],
        temperature=WEB_SEARCH_AGENT_CONFIG["temperature"],
        max_tokens=WEB_SEARCH_AGENT_CONFIG["max_tokens"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Search for Python async tutorials"},
            {"role": "assistant", "content": '[{"tool": "tavily_search", "args": ["Python async tutorial", 5, false, false, "basic"], "reasoning": "Getting curated high-quality tutorials from Tavily"}, {"tool": "duck_duck_go", "args": ["Python async tutorial", 5, "us-en", "moderate", null], "reasoning": "Getting additional diverse perspectives from DuckDuckGo"}]'},
            {"role": "user", "content": task}
        ]
    )

    # Parse and execute the initial plan
    raw_plan = response.choices[0].message.content
    plan = parse_plan_from_response(raw_plan)
    result = await execute_tool_plan(plan, agent="web_search")

    print(f"[Web Search Agent] Initial search complete")

    full_result = str(result.get("final_result", ""))

    # ----------------------------------
    # REFLEXION: Evaluate search quality
    # ----------------------------------
    print("\n🤔 [Reflexion] Evaluating search quality...")

    evaluation_prompt = f"""You are evaluating the quality of web search results for this task:

Task: {task}

Search Results:
{full_result[:3000]}... (truncated if long)

Evaluate the search results on these criteria:
1. **Relevance**: Do results directly address the task?
2. **Depth**: Is there enough detailed information?
3. **Coverage**: Are different perspectives/sources covered?
4. **Quality**: Are sources credible and informative?

Respond in JSON format:
{{
  "quality_score": 8.5,
  "relevance_score": 9.0,
  "depth_score": 8.0,
  "coverage_score": 8.5,
  "quality_score": 9.0,
  "sufficient": true,
  "gaps": ["any information gaps found"],
  "reasoning": "Brief explanation of the assessment",
  "suggested_searches": ["additional search query 1", "additional search query 2"]
}}

If quality_score >= 8.0, set sufficient=true. Otherwise suggest 1-2 additional targeted searches.
"""

    eval_response = await client.chat.completions.create(
        model=WEB_SEARCH_AGENT_CONFIG["model"],
        temperature=0.3,
        messages=[{"role": "user", "content": evaluation_prompt}]
    )

    raw_eval = eval_response.choices[0].message.content

    # Parse evaluation
    try:
        eval_data = json.loads(raw_eval)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_eval, re.DOTALL)
        if json_match:
            eval_data = json.loads(json_match.group(1))
        else:
            # Fallback: assume quality is sufficient
            eval_data = {"quality_score": 8.0, "sufficient": True, "gaps": [], "reasoning": "Could not parse evaluation"}

    quality_score = eval_data.get("quality_score", 8.0)
    is_sufficient = eval_data.get("sufficient", True)
    gaps = eval_data.get("gaps", [])
    reasoning = eval_data.get("reasoning", "")
    suggested_searches = eval_data.get("suggested_searches", [])

    print(f"   📊 Quality Score: {quality_score}/10")
    print(f"   ✓ Sufficient: {is_sufficient}")
    print(f"   💡 Reasoning: {reasoning[:100]}...")

    if gaps:
        print(f"   ⚠️  Gaps identified: {', '.join(gaps[:2])}")

    # ----------------------------------
    # CONDITIONAL FOLLOW-UP SEARCHES
    # ----------------------------------
    if not is_sufficient and suggested_searches:
        print(f"\n🔍 [Follow-up] Quality below threshold - performing {len(suggested_searches)} additional searches...")

        for i, search_query in enumerate(suggested_searches[:2], 1):  # Limit to 2 follow-up searches
            print(f"   [{i}] Searching: {search_query[:60]}...")

            # Create follow-up search plan (prefer Tavily for quality)
            followup_plan_prompt = f"""Create a search plan for this follow-up query: {search_query}

Use tavily_search for high-quality results. Respond with ONLY a JSON array:
[
  {{"tool": "tavily_search", "args": ["{search_query}", 5, false, false, "basic"], "reasoning": "Targeted follow-up search to fill gaps"}}
]
"""

            followup_response = await client.chat.completions.create(
                model=WEB_SEARCH_AGENT_CONFIG["model"],
                temperature=0.3,
                messages=[{"role": "user", "content": followup_plan_prompt}]
            )

            followup_raw_plan = followup_response.choices[0].message.content
            followup_plan = parse_plan_from_response(followup_raw_plan)
            followup_result = await execute_tool_plan(followup_plan, agent="web_search")

            # Append follow-up results
            followup_data = str(followup_result.get("final_result", ""))
            full_result += f"\n\n--- Follow-up Search {i}: {search_query} ---\n{followup_data}"

            print(f"   ✅ [{i}] Found {len(followup_data)} chars of additional data")

        print(f"✅ [Follow-up] Additional searches complete - results enhanced!")

    # Create intelligent summary using LLM if content is long
    summary = await smart_summarize(full_result, context="web_search")

    print(f"\n[Web Search Agent] Final summary: {summary[:100]}...")
    print(f"[Web Search Agent] Total result length: {len(full_result)} chars")

    return WebSearchAgentResult(
        final_result=full_result,
        steps=json.dumps(result.get("steps", [])),
        summary=summary,
        error=result.get("error", "")
    )