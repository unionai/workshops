"""
Dynamic workflow example using the planner agent for intelligent task routing.
This workflow demonstrates how the planner can dynamically choose which agent to use for different tasks.

Each agent (planner, math, string) is now a standalone Flyte task with its own TaskEnvironment,
allowing independent scaling, resource allocation, and container configuration.

python -m workflows.planner --request "Calculate 15 times 7"
python -m workflows.planner --request "Search for the latest news about AI agents
"""

from typing import List, Dict
from dataclasses import dataclass
import flyte
import asyncio

from agents import import_all_agents
from agents.planner_agent import planner_agent, AgentStep
import_all_agents()

from config import base_env
from utils.logger import Logger, setup_logging
from utils.decorators import agent_registry
from utils.plan_executor import set_logger

trace_logger = Logger(path="planner_trace_log.jsonl", verbose=False)
set_logger(trace_logger)
log = setup_logging(__name__)

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
    result_summary: str
    result_full: str
    error: str = ""


@dataclass
class TaskResult:
    """Final result from dynamic task execution"""
    planner_decision_summary: str
    agent_executions: List[AgentExecution]
    final_result: str


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
    log.info(f"[Orchestrator] User request: {user_request}")

    # ----------------------------------
    # PHASE 1: Planning - Generate Execution Plan
    # ----------------------------------
    # Planner agent analyzes request and creates DAG (directed acyclic graph) of agent steps
    log.info("[Orchestrator] Step 1: Calling planner agent...")
    planner_decision = await planner_agent(user_request)
    log.info(f"[Orchestrator] Planner created plan with {len(planner_decision.steps)} step(s)")

    # ----------------------------------
    # PHASE 2: Execution - Dependency-Aware Parallel Orchestration
    # ----------------------------------
    # Execute steps in waves: all independent steps run in parallel, dependent steps wait

    # Store completed results indexed by step number (enables dependency lookup)
    completed_results: Dict[int, AgentExecution] = {}

    # Track which steps still need execution
    pending_steps = list(enumerate(planner_decision.steps))

    # ----------------------------------
    # Main Orchestration Loop
    # ----------------------------------
    # Continue until all steps complete or circular dependency detected
    while pending_steps:
        # ----------------------------------
        # Identify Ready Steps (Dependencies Satisfied)
        # ----------------------------------
        # Partition pending steps into: ready to execute vs waiting on dependencies
        ready_steps = []
        remaining_steps = []

        for step_idx, step in pending_steps:
            # Check if all dependencies are completed
            deps_satisfied = all(dep_idx in completed_results for dep_idx in step.dependencies)

            if deps_satisfied:
                ready_steps.append((step_idx, step))
            else:
                remaining_steps.append((step_idx, step))

        # Circular dependency detection
        if not ready_steps:
            log.error("[Orchestrator] ERROR: No steps ready to execute, but pending steps remain (circular dependency?)")
            break

        log.info(f"[Orchestrator] Executing {len(ready_steps)} step(s) in parallel...")

        # ----------------------------------
        # Parallel Step Execution
        # ----------------------------------
        # Define nested async function to execute a single step (enables parallel gather)
        async def execute_step(step_idx: int, step: AgentStep) -> tuple:
            """Execute a single agent step with context injection"""
            log.info(f"[Orchestrator]   Step {step_idx}: Calling {step.agent} agent...")
            log.info(f"[Orchestrator]     Task: {step.task}")

            # ----------------------------------
            # Context Injection - How Results Flow Between Agents
            # ----------------------------------
            # If this step depends on previous steps, inject their results into the task prompt
            # Example: "Step 0 result: 120\nStep 1 result: Factorial...\n\nYOUR TASK: Add these"
            task = build_task_with_context(step.task, step.dependencies, completed_results)

            if step.dependencies:
                log.info(f"[Orchestrator]     Context from steps {step.dependencies} added to task")

            # ----------------------------------
            # Dynamic Agent Routing
            # ----------------------------------
            # Look up agent function from registry (populated at import time via decorators)
            agent_func = agent_registry.get(step.agent)
            if not agent_func:
                # Unknown agent - the planner hallucinated or requested invalid agent
                log.warning(f"[Orchestrator] WARNING: Unknown agent '{step.agent}'")
                result_full = ""
                result_summary = ""
                error = f"Unknown agent: {step.agent}"
            else:
                # Execute the agent with the (potentially context-enriched) task
                agent_result = await agent_func(task)
                result_full = agent_result.final_result
                # Use summary field if available (e.g., web_search), otherwise use final_result
                result_summary = getattr(agent_result, 'summary', agent_result.final_result)
                error = agent_result.error

            log.info(f"[Orchestrator]   Step {step_idx} completed: {result_summary[:300]}...")

            # Persist execution details to log file for debugging and analysis
            await trace_logger.log(
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

            # Return tuple: (step_idx, execution_result) for results collection
            return step_idx, AgentExecution(
                agent=step.agent,
                task=step.task,
                result_summary=result_summary,
                result_full=result_full,
                error=error
            )

        # ----------------------------------
        # Execute All Ready Steps Concurrently
        # ----------------------------------
        # asyncio.gather runs all ready steps in parallel (fanout pattern)
        # This is where the speedup comes from: independent steps don't wait for each other
        results = await asyncio.gather(*[execute_step(idx, step) for idx, step in ready_steps])

        # ----------------------------------
        # Store Completed Results
        # ----------------------------------
        # Index by step number so dependent steps can look up results
        for step_idx, execution in results:
            completed_results[step_idx] = execution

        # Update pending steps list (remove completed, keep waiting)
        pending_steps = remaining_steps

    # ----------------------------------
    # PHASE 3: Results Collection and Synthesis
    # ----------------------------------

    # ----------------------------------
    # Reconstruct Execution Order
    # ----------------------------------
    # Convert dict back to list in original plan order for result presentation
    agent_executions = [completed_results[i] for i in range(len(planner_decision.steps))]

    # ----------------------------------
    # Aggregate Results from All Agents
    # ----------------------------------
    # Collect summaries from each step, handling errors gracefully
    final_results = []
    for execution in agent_executions:
        if execution.error:
            final_results.append(f"{execution.agent}: ERROR - {execution.error}")
        elif execution.result_summary:
            final_results.append(f"{execution.agent}: {execution.result_summary}")

    # Combine all agent outputs into single result string
    combined_result = " | ".join(final_results) if final_results else "No results"
    log.info(f"[Orchestrator] All agents completed. Combined result: {combined_result}")

    # Create human-readable summary of what was executed
    planner_summary = f"{len(planner_decision.steps)} step(s): " + ", ".join(
        [f"{s.agent}" for s in planner_decision.steps]
    )

    # ----------------------------------
    # Return Complete Execution Trace
    # ----------------------------------
    # Package plan summary, individual executions, and final result
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
        help="Run workflow locally instead of remote execution"
    )
    parser.add_argument(
        "--request",
        type=str,
        default="Calculate 5 factorial",
        help="The task request to execute (see README.md for examples)"
    )
    parser.add_argument(
        "--get-output",
        type=str,
        nargs="?",
        const="latest",
        metavar="RUN_NAME",
        help="Fetch output from a run. Use without value for latest, or provide run name."
    )
    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        print("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        print("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    # ----------------------------------
    # Fetch Latest Output Mode
    # ----------------------------------
    if args.get_output:
        from flyte.remote import Run

        if args.get_output == "latest":
            print("\n=== Fetching Latest Run Output ===\n")
            # Get latest run
            runs = list(Run.listall(
                sort_by=("created_at", "desc"),
                limit=1
            ))
            if not runs:
                print("No runs found")
                exit(1)
            run_name = runs[0].name
            print(f"Latest run: {run_name}")
        else:
            run_name = args.get_output
            print(f"\n=== Fetching Output for Run: {run_name} ===\n")

        run = Run.get(run_name)
        outputs = run.outputs()
        print(f"Result: {outputs.named_outputs}")
        print(f"{'='*60}\n")
    else:
        # ----------------------------------
        # Run New Execution Mode
        # ----------------------------------
        print(f"\n=== Planner Agent Workflow ===")
        print(f"Request: {args.request}\n")

        execution = flyte.run(
            planner_agent_workflow,
            user_request=args.request
        )

        print(f"\n{'='*60}")
        print(f"Execution: {execution.name}")
        print(f"URL: {execution.url}")
        print(f"{'='*60}\n")


        # flyte run --local --tui workflows/planner.py planner_agent_workflow --user_request "Compare the tech industries of Japan, South Korea, and Germany"                                     
