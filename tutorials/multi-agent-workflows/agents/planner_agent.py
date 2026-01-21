"""
This module defines the planner_agent, which routes requests to appropriate specialist agents.
"""

import json
import flyte
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import List

from config import OPENAI_API_KEY
from utils.decorators import agent, agent_registry
from config import base_env

import agents.math_agent
import agents.string_agent
import agents.web_search_agent
import agents.code_agent
import agents.weather_agent

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class AgentStep:
    """Single step in the execution plan"""
    agent: str
    task: str
    dependencies: List[int] = None  # List of step indices this step depends on (0-indexed)

    def __post_init__(self):
        # Default to empty list if None
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class PlannerDecision:
    """Decision from planner agent - can contain multiple steps"""
    steps: List[AgentStep]


# ----------------------------------
# Planner Agent Task Environment
# ----------------------------------
env = base_env
# Future: If you need agent-specific dependencies, create separate environments:
# env = flyte.TaskEnvironment(
#     name="code_agent_env",
#     image=base_env.image.with_pip_packages(["numpy", "pandas"]),
#     secrets=base_env.secrets,
#     resources=flyte.Resources(cpu=2, mem="4Gi")
# )


@agent("planner")
@env.task
async def planner_agent(user_request: str) -> PlannerDecision:
    """
    Planner agent that analyzes requests and creates execution plans.

    Args:
        user_request (str): The user's request to analyze and route.

    Returns:
        PlannerDecision: Plan with one or more agent steps.
    """
    print(f"[Planner Agent] Processing request: {user_request}")

    # Initialize client inside task for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    available_agents = [a for a in agent_registry if a != "planner"]
    agent_list = "\n".join([f"- {a}" for a in available_agents])

    system_msg = f"""
You are a routing agent.
Available agents:
{agent_list}

Analyze the user's request and decide which agent(s) to use.

DEPENDENCIES: Use 'dependencies' to specify which steps must complete before this step.
- dependencies is a list of step indices (0-indexed)
- Empty list [] means the step can run immediately (no dependencies)
- Steps with no dependencies can run in PARALLEL
- Steps with dependencies will wait for those steps to complete and receive their results

IMPORTANT: When a step depends on previous steps, the results will be automatically
provided in the prompt. Write clear task descriptions that reference what to do with those results.

Examples:

INDEPENDENT TASKS (can run in parallel):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 factorial", "dependencies": []}},
  {{"agent": "string", "task": "Count words in 'Hello World'", "dependencies": []}},
  {{"agent": "web_search", "task": "Search for Python tutorials", "dependencies": []}}
]}}

DEPENDENT TASKS (step 1 uses result from step 0):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 times 3", "dependencies": []}},
  {{"agent": "math", "task": "Add 10 to the result from step 0", "dependencies": [0]}}
]}}

CROSS-AGENT DEPENDENCIES (web search then math):
{{"steps": [
  {{"agent": "web_search", "task": "Search for USA GDP in 2023", "dependencies": []}},
  {{"agent": "math", "task": "Calculate 10% of the GDP value from step 0", "dependencies": [0]}}
]}}

MULTIPLE DEPENDENCIES (step 2 waits for both 0 and 1):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 factorial", "dependencies": []}},
  {{"agent": "string", "task": "Count letters in 'test'", "dependencies": []}},
  {{"agent": "code", "task": "Multiply the factorial from step 0 by the letter count from step 1", "dependencies": [0, 1]}}
]}}

IMPORTANT: Always include 'dependencies' field for each step, even if empty [].
"""

    res = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_request}
        ]
    )

    result = json.loads(res.choices[0].message.content)
    print(f"[Planner Agent] Raw result: {result}")

    # Convert to dataclass
    steps = [
        AgentStep(
            agent=step["agent"],
            task=step["task"],
            dependencies=step.get("dependencies", [])
        )
        for step in result["steps"]
    ]

    print(f"[Planner Agent] Plan has {len(steps)} step(s)")
    for i, step in enumerate(steps):
        deps_str = f" → depends on steps {step.dependencies}" if step.dependencies else " → independent (can run in parallel)"
        print(f"[Planner Agent]   Step {i}: [{step.agent}] {step.task}{deps_str}")

    return PlannerDecision(steps=steps)