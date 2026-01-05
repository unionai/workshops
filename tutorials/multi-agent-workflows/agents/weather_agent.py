"""
This module defines the weather_agent, which can get weather information for locations.
"""

import json
import flyte
from openai import AsyncOpenAI

# Import tools to register them
import tools.weather_tools

from utils.decorators import agent, agent_tools
from utils.plan_executor import execute_tool_plan, parse_plan_from_response
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY

# ----------------------------------
# Agent-Specific Configuration
# ----------------------------------
WEATHER_AGENT_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.0,
    "max_tokens": 300,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WeatherAgentResult:
    """Result from weather agent execution"""
    final_result: str
    steps: str  # JSON string of steps taken
    error: str = ""  # Empty if no error


# ----------------------------------
# Weather Agent Task Environment
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
@agent("weather")
async def weather_agent(task: str) -> WeatherAgentResult:
    """
    Weather agent that can get current weather information for locations.

    Args:
        task (str): The weather task to perform.

    Returns:
        WeatherAgentResult: The result of the weather query and the steps taken.
    """
    print(f"[Weather Agent] Processing: {task}")

    # Initialize client inside task for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    toolset = agent_tools["weather"]
    tool_list = "\n".join([f"{name}: {fn.__doc__.strip()}" for name, fn in toolset.items()])
    system_msg = f"""
You are a weather information agent. You can get current weather information for any location.

Tools:
{tool_list}

CRITICAL: You must respond with ONLY a valid JSON array, nothing else. No markdown, no explanations.
Return a JSON array of tool calls in this exact format:
[
  {{"tool": "get_weather", "args": ["London"], "reasoning": "Getting current weather for London"}}
]

RULES:
1. Start your response with [ and end with ]
2. No markdown code blocks (no ```)
3. No extra text before or after the JSON
4. Always include a "reasoning" field for each step
5. Use the location name as the argument (e.g., "London", "New York", "Tokyo")
"""

    # Call LLM to create plan using agent-specific config
    response = await client.chat.completions.create(
        model=WEATHER_AGENT_CONFIG["model"],
        temperature=WEATHER_AGENT_CONFIG["temperature"],
        max_tokens=WEATHER_AGENT_CONFIG["max_tokens"],
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "What's the weather in London?"},
            {"role": "assistant", "content": '[{"tool": "get_weather", "args": ["London"], "reasoning": "Getting current weather for London"}]'},
            {"role": "user", "content": task}
        ]
    )

    # Parse and execute the plan
    raw_plan = response.choices[0].message.content
    plan = parse_plan_from_response(raw_plan)
    result = await execute_tool_plan(plan, agent="weather")

    print(f"[Weather Agent] Result: {result}")

    return WeatherAgentResult(
        final_result=str(result.get("final_result", "")),
        steps=json.dumps(result.get("steps", [])),
        error=result.get("error", "")
    )
