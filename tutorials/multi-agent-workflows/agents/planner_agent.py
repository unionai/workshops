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


@env.task
@agent("planner")
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

    memory_log = []  # No memory persistence for now
    context = "\n".join([f"- {q} → {r}" for q, r in memory_log[-5:]]) or "No history."
    available_agents = [a for a in agent_registry if a != "planner"]
    agent_list = "\n".join([f"- {a}" for a in available_agents])

    system_msg = f"""
You are a routing agent.
Available agents:
{agent_list}

Recent memory:
{context}

Analyze the user's request and decide which agent(s) to use.

DEPENDENCIES: Use 'dependencies' to specify which steps must complete before this step.
- dependencies is a list of step indices (0-indexed)
- Empty list [] means the step can run immediately (no dependencies)
- Steps with no dependencies can run in PARALLEL
- Steps with dependencies will wait for those steps to complete

Examples:

INDEPENDENT TASKS (can run in parallel):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 factorial", "dependencies": []}},
  {{"agent": "string", "task": "Count words in 'Hello World'", "dependencies": []}},
  {{"agent": "web_search", "task": "Search for Python tutorials", "dependencies": []}}
]}}

DEPENDENT TASKS (step 1 must complete before step 2):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 times 3", "dependencies": []}},
  {{"agent": "math", "task": "Add 10 to the previous result", "dependencies": [0]}}
]}}

MIXED (steps 0 and 1 run in parallel, step 2 waits for both):
{{"steps": [
  {{"agent": "math", "task": "Calculate 5 factorial", "dependencies": []}},
  {{"agent": "string", "task": "Count letters in 'test'", "dependencies": []}},
  {{"agent": "code", "task": "Combine the results", "dependencies": [0, 1]}}
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

    # Handle old format (single step without "steps" wrapper)
    if "agent" in result and "task" in result:
        print(f"[Planner Agent] Converting old format to new format")
        result = {"steps": [{"agent": result["agent"], "task": result["task"]}]}

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
    for i, step in enumerate(steps, 1):
        deps_str = f" (depends on: {step.dependencies})" if step.dependencies else " (no dependencies)"
        print(f"[Planner Agent]   Step {i}: {step.agent} - {step.task}{deps_str}")

    return PlannerDecision(steps=steps)