"""
This module defines the math_agent, which is responsible for solving arithmetic, powers, and multi-step problems.
"""

import json
import flyte
from openai import AsyncOpenAI

# Import tools to register them
import tools.math_tools

from utils.decorators import agent, agent_tools
from utils.plan_executor import execute_tool_plan, parse_plan_from_response
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY

# ----------------------------------
# Agent-Specific Configuration
# ----------------------------------
MATH_AGENT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 500,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class MathAgentResult:
    """Result from math agent execution"""
    final_result: str
    steps: str  # JSON string of steps taken
    error: str = ""  # Empty if no error


# ----------------------------------
# Math Agent Task Environment
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
@agent("math")
async def math_agent(task: str) -> MathAgentResult:
    """
    Math agent that processes user prompts to solve arithmetic and multi-step problems.

    Args:
        task (str): The math task to solve.

    Returns:
        MathAgentResult: The result of the computation and the steps taken.
    """
    print(f"[Math Agent] Processing: {task}")

    # Initialize client inside task for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Build system message with available tools
    toolset = agent_tools["math"]
    tool_list = "\n".join([f"{name}: {fn.__doc__.strip()}" for name, fn in toolset.items()])
    system_msg = f"""
You are a math agent that can solve arithmetic, powers, and multi-step problems.

Tools:
{tool_list}

CRITICAL: You must respond with ONLY a valid JSON array, nothing else. No markdown, no explanations.
Return a JSON array of tool calls in this exact format:
[
  {{"tool": "add", "args": [2, 3], "reasoning": "Adding 2 and 3 to compute the sum."}},
  {{"tool": "multiply", "args": ["previous", 5], "reasoning": "Multiplying previous result by 5."}}
]

RULES:
1. Start your response with [ and end with ]
2. No markdown code blocks (no ```)
3. No extra text before or after the JSON
4. Always include a "reasoning" field for each step
5. Use "previous" in args to reference the previous step result
"""

    # Call LLM to create plan using agent-specific config
    response = await client.chat.completions.create(
        model=MATH_AGENT_CONFIG["model"],
        temperature=MATH_AGENT_CONFIG["temperature"],
        max_tokens=MATH_AGENT_CONFIG["max_tokens"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "Add 2 and 3"},
            {"role": "assistant", "content": '[{"tool": "add", "args": [2, 3], "reasoning": "Adding 2 and 3"}]'},
            {"role": "user", "content": task}
        ]
    )

    # Parse and execute the plan
    raw_plan = response.choices[0].message.content
    plan = parse_plan_from_response(raw_plan)
    result = await execute_tool_plan(plan, agent="math")

    print(f"[Math Agent] Result: {result}")

    return MathAgentResult(
        final_result=str(result.get("final_result", "")),
        steps=json.dumps(result.get("steps", [])),
        error=result.get("error", "")
    )