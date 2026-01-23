"""
Dynamic workflow example using the planner agent for intelligent task routing.
This workflow demonstrates how the planner can dynamically choose which agent to use for different tasks.

Each agent (planner, math, string) is now a standalone Flyte task with its own TaskEnvironment,
allowing independent scaling, resource allocation, and container configuration.

python -m workflows.planner --request "Calculate 15 times 7"
python -m workflows.planner --request "Search for the latest news about AI agents
"""

import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import flyte
import asyncio

# Add workflows directory to Python path for imports
workflows_dir = Path(__file__).parent
sys.path.insert(0, str(workflows_dir))

# Import agents (they are now Flyte tasks with their own environments)
from agents.planner_agent import planner_agent, PlannerDecision, AgentStep
from agents.math_agent import math_agent, MathAgentResult
from agents.string_agent import string_agent, StringAgentResult
from agents.web_search_agent import web_search_agent, WebSearchAgentResult
from agents.code_agent import code_agent, CodeAgentResult
from agents.weather_agent import weather_agent, WeatherAgentResult
from config import base_env
from utils.logger import Logger
from utils.decorators import agent_registry

# Initialize logger for orchestrator
logger = Logger(path="agent_trace_log.jsonl", verbose=False)

# ----------------------------------
# Helper Functions
# ----------------------------------

def build_task_with_context(
    task: str,
    dependencies: List[int],
    completed_results: Dict
) -> str:
    """
    Build a task prompt with context from dependent steps.

    This is how results flow between agents: when a step depends on previous steps,
    we prepend their results to the task prompt so the agent has the context it needs.

    Args:
        task: The original task description from the planner
        dependencies: List of step indices this task depends on
        completed_results: Dictionary of completed step results

    Returns:
        If no dependencies: returns task unchanged
        If has dependencies: returns task with context section prepended

    Example output:
        ============================================================
        RESULTS FROM PREVIOUS STEPS:
        ============================================================
          - Step 0 (web_search agent): France's GDP is €2.6 trillion
        ============================================================

        YOUR TASK:
        Calculate 5% of the GDP from step 0
    """
    if not dependencies:
        return task

    # Build context section with results from dependent steps
    context_lines = [
        f"  - Step {dep_idx} ({completed_results[dep_idx].agent} agent): {completed_results[dep_idx].result_summary}"
        for dep_idx in dependencies
    ]

    # Format with clear visual separators
    context_header = "=" * 60 + "\nRESULTS FROM PREVIOUS STEPS:\n" + "=" * 60
    context_footer = "=" * 60

    return (
        f"{context_header}\n"
        f"{chr(10).join(context_lines)}\n"
        f"{context_footer}\n\n"
        f"YOUR TASK:\n{task}"
    )

# ----------------------------------
# Data Models for Orchestrator
# ----------------------------------

@dataclass
class AgentExecution:
    """Single agent execution with its result"""
    agent: str
    task: str
    result_summary: str  # Concise summary for passing to dependent steps
    result_full: str     # Complete result for final output and debugging
    error: str = ""


@dataclass
class TaskResult:
    """Final result from dynamic task execution"""
    planner_decision_summary: str
    agent_executions: List[AgentExecution]
    final_result: str  # Combined final result


# ----------------------------------
# Orchestrator Task Environment
# ----------------------------------
env = base_env
# env = flyte.TaskEnvironment(
#     name="orchestrator_env",
#     image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
#     secrets=[
#         flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
#     ],
# )


# ----------------------------------
# Main Orchestration Task
# ----------------------------------

@env.task
async def planner_agent_workflow(user_request: str) -> TaskResult:
    """
    Planner-based multi-agent workflow with dynamic routing and parallel execution.

    This workflow uses a planner agent to analyze the request and create an execution
    plan with dependencies, then orchestrates specialist agents accordingly. Steps
    with no dependencies run in parallel (fanout pattern).

    Args:
        user_request: The user's request to fulfill

    Returns:
        TaskResult: Combined result from all agent executions
    """
    print(f"[Orchestrator] User request: {user_request}")

    # Step 1: Call planner task to create execution plan
    print("[Orchestrator] Step 1: Calling planner agent...")
    planner_decision = await planner_agent(user_request)
    print(f"[Orchestrator] Planner created plan with {len(planner_decision.steps)} step(s)")

    # Step 2: Execute agent tasks with dependency-aware parallelism
    # Store completed results indexed by step number
    completed_results: Dict[int, AgentExecution] = {}

    # Track which steps are ready to execute (no pending dependencies)
    pending_steps = list(enumerate(planner_decision.steps))

    while pending_steps:
        # Find all steps that can execute now (dependencies satisfied)
        ready_steps = []
        remaining_steps = []

        for step_idx, step in pending_steps:
            # Check if all dependencies are completed
            deps_satisfied = all(dep_idx in completed_results for dep_idx in step.dependencies)

            if deps_satisfied:
                ready_steps.append((step_idx, step))
            else:
                remaining_steps.append((step_idx, step))

        if not ready_steps:
            # This shouldn't happen with valid dependency graphs, but handle it gracefully
            print("[Orchestrator] ERROR: No steps ready to execute, but pending steps remain (circular dependency?)")
            break

        print(f"[Orchestrator] Executing {len(ready_steps)} step(s) in parallel...")

        # Execute all ready steps in parallel
        async def execute_step(step_idx: int, step: AgentStep) -> tuple:
            """Execute a single agent step"""
            print(f"[Orchestrator]   Step {step_idx}: Calling {step.agent} agent...")
            print(f"[Orchestrator]     Task: {step.task}")

            # Build task with context from dependencies (if any)
            # This is how results flow between agents - previous results get prepended to the prompt
            task = build_task_with_context(step.task, step.dependencies, completed_results)

            if step.dependencies:
                print(f"[Orchestrator]     Context from steps {step.dependencies} added to task")

            # Route to appropriate agent task using agent registry
            # The registry now contains Flyte-wrapped versions thanks to decorator order
            agent_func = agent_registry.get(step.agent)
            if not agent_func:
                # Unknown agent - the planner hallucinated or requested invalid agent
                print(f"[Orchestrator] WARNING: Unknown agent '{step.agent}'")
                result_full = ""
                result_summary = ""
                error = f"Unknown agent: {step.agent}"
            else:
                # Call the agent and extract results
                agent_result = await agent_func(task)
                result_full = agent_result.final_result
                # Use summary field if available, otherwise use final_result
                result_summary = getattr(agent_result, 'summary', agent_result.final_result)
                error = agent_result.error

            print(f"[Orchestrator]   Step {step_idx} completed: {result_summary[:100]}...")

            # Log to trace file
            await logger.log(
                step_idx=step_idx,
                agent=step.agent,
                input_task=task,
                output_full=result_full,
                output_summary=result_summary,
                output_full_length=len(result_full),
                output_summary_length=len(result_summary),
                error=error,
                dependencies=step.dependencies
            )

            return step_idx, AgentExecution(
                agent=step.agent,
                task=step.task,
                result_summary=result_summary,
                result_full=result_full,
                error=error
            )

        # Execute all ready steps concurrently
        results = await asyncio.gather(*[execute_step(idx, step) for idx, step in ready_steps])

        # Store completed results
        for step_idx, execution in results:
            completed_results[step_idx] = execution

        # Update pending steps
        pending_steps = remaining_steps

    # Convert to list in original order
    agent_executions = [completed_results[i] for i in range(len(planner_decision.steps))]

    # Collect final results
    final_results = []
    for execution in agent_executions:
        if execution.error:
            final_results.append(f"{execution.agent}: ERROR - {execution.error}")
        elif execution.result_summary:
            final_results.append(f"{execution.agent}: {execution.result_summary}")

    # Combine all results
    combined_result = " | ".join(final_results) if final_results else "No results"
    print(f"[Orchestrator] All agents completed. Combined result: {combined_result}")

    # Create summary of planner decision
    planner_summary = f"{len(planner_decision.steps)} step(s): " + ", ".join(
        [f"{s.agent}" for s in planner_decision.steps]
    )

    return TaskResult(
        planner_decision_summary=planner_summary,
        agent_executions=agent_executions,
        final_result=combined_result
    )


# ----------------------------------
# Local Execution Helper
# ----------------------------------

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run Flyte dynamic workflow with intelligent agent routing",
        epilog="Example: python workflows/flyte_dynamic.py --local --request 'Calculate 5 factorial'"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run workflow locally using flyte.init() instead of remote execution"
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Calculate 5 factorial",
        help="The task request to execute (see README.md for examples)"
    )
    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        print("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        print("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    print(f"\n=== Planner Agent Workflow ===")
    print(f"Request: {args.request}\n")

    execution = flyte.run(
        planner_agent_workflow,
        user_request=args.request
    )

    print(f"\n{'='*60}")
    print(f"Execution: {execution.name}")
    print(f"URL: {execution.url}")
    print("Click the link above to view execution details in the Flyte UI")
    print(f"{'='*60}\n")
    print("\nSee README.md for more example queries!")