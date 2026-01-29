"""
Sequential workflow - predefined agent pipeline execution.

This workflow implements a static sequential pattern where agents are called
in a predetermined order with results passed from one to the next.

Unlike dynamic patterns (planner, ReAct, reflection), this uses no LLM planning -
just executes agents in the exact sequence you define. This is:
- Simpler and more predictable
- Faster (no planning overhead)
- Cheaper (fewer LLM calls)
- Perfect for known workflows like write → review → edit

Usage:
    python -m workflows.sequential --local --pipeline write-review-edit --input "Write a poem about AI"
"""

from typing import List, Dict
from dataclasses import dataclass
import flyte

# Auto-import all agent modules to trigger @agent and @tool decorator registration
from agents import import_all_agents
import_all_agents()
from config import base_env
from utils.logger import Logger, setup_logging
from utils.decorators import agent_registry

# Initialize trace logger for structured JSONL output
trace_logger = Logger(path="sequential_trace_log.jsonl", verbose=False)
# Initialize standard logger for console output
log = setup_logging(__name__)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class PipelineStep:
    """Single step in the pipeline"""
    step_number: int
    agent: str
    task_template: str  # Can use {input} or {previous} placeholders
    actual_task: str    # Task after placeholder substitution
    result: str
    error: str = ""

@dataclass
class SequentialResult:
    """Final result from sequential workflow"""
    pipeline_name: str
    initial_input: str
    steps: List[PipelineStep]
    final_result: str
    total_steps: int
    success: bool

# ----------------------------------
# Pipeline Definitions
# ----------------------------------

PIPELINES = {
    "write-review-edit": [
        {"agent": "code", "task": "Write Python code for: {input}"},
        {"agent": "string", "task": "Review this code and count how many lines: {previous}"},
        {"agent": "code", "task": "Add docstrings to this code: {previous_0}"}
    ],
    "search-summarize": [
        {"agent": "web_search", "task": "Search for: {input}"},
        {"agent": "string", "task": "Count words in this search result: {previous}"}
    ],
    "calculate-explain": [
        {"agent": "math", "task": "{input}"},
        {"agent": "string", "task": "Analyze this number and describe it: {previous}"}
    ],
    "weather-analyze": [
        {"agent": "weather", "task": "Get weather for: {input}"},
        {"agent": "string", "task": "Count the words in this weather report: {previous}"}
    ],
    "multi-calc": [
        {"agent": "math", "task": "Calculate: {input}"},
        {"agent": "math", "task": "Multiply the previous result by 2: {previous}"},
        {"agent": "math", "task": "Add 10 to the previous result: {previous}"}
    ]
}

# ----------------------------------
# Helper Functions
# ----------------------------------

def substitute_placeholders(
    task_template: str,
    user_input: str,
    previous_results: List[str]
) -> str:
    """
    Substitute placeholders in task template.

    Placeholders:
        {input} - Original user input
        {previous} - Result from immediately previous step
        {previous_0}, {previous_1}, etc. - Result from specific step index

    Args:
        task_template: Template with placeholders
        user_input: Original user input
        previous_results: List of results from all previous steps

    Returns:
        Task string with placeholders replaced
    """
    task = task_template

    # Substitute {input}
    task = task.replace("{input}", user_input)

    # Substitute {previous} - most recent result
    if previous_results and "{previous}" in task:
        task = task.replace("{previous}", str(previous_results[-1]))

    # Substitute {previous_N} - specific step result
    for i, result in enumerate(previous_results):
        placeholder = f"{{previous_{i}}}"
        if placeholder in task:
            task = task.replace(placeholder, str(result))

    return task

# ----------------------------------
# Sequential Orchestrator
# ----------------------------------

env = base_env

@env.task
async def sequential_workflow(
    user_input: str,
    pipeline_name: str = "write-review-edit"
) -> SequentialResult:
    """
    Sequential workflow that executes agents in a predefined order.

    Args:
        user_input: The user's input/request
        pipeline_name: Name of the pipeline to execute (see PIPELINES dict)

    Returns:
        SequentialResult: Complete execution trace and final result
    """
    log.info("=" * 80)
    log.info(f"SEQUENTIAL WORKFLOW - Pipeline: {pipeline_name}")
    log.info(f"Input: {user_input}")
    log.info("=" * 80)

    # ----------------------------------
    # Pipeline Selection and Validation
    # ----------------------------------
    # Look up predefined pipeline from PIPELINES dictionary
    if pipeline_name not in PIPELINES:
        available = list(PIPELINES.keys())
        raise ValueError(
            f"Unknown pipeline '{pipeline_name}'. "
            f"Available pipelines: {available}"
        )

    # Get pipeline definition: list of {agent, task} step dictionaries
    pipeline_definition = PIPELINES[pipeline_name]
    log.info(f"\n[Sequential] Pipeline has {len(pipeline_definition)} step(s)")

    # Display execution plan (fixed sequence - no dynamic planning)
    for i, step_def in enumerate(pipeline_definition):
        log.info(f"  Step {i}: {step_def['agent']} - {step_def['task'][:60]}...")

    # ----------------------------------
    # Initialization for Sequential Execution
    # ----------------------------------
    # Track each step's execution and accumulate results
    steps = []                  # Full execution history
    previous_results = []       # Results available for placeholder substitution
    success = True              # Track if pipeline completed without errors

    # ----------------------------------
    # Main Sequential Execution Loop
    # ----------------------------------
    # Execute steps in exact order defined in pipeline (NO parallelism, NO planning)
    for step_num, step_def in enumerate(pipeline_definition):
        log.info(f"\n{'='*80}")
        log.info(f"STEP {step_num}")
        log.info(f"{'='*80}")

        agent_name = step_def["agent"]
        task_template = step_def["task"]  # May contain {input}, {previous}, {previous_N} placeholders

        # ----------------------------------
        # Placeholder Substitution - How Results Flow
        # ----------------------------------
        # Replace placeholders with actual values from user input and previous step results
        # Example: "{previous}" → "120", "{previous_0}" → "5", "{input}" → "calculate factorial"
        actual_task = substitute_placeholders(task_template, user_input, previous_results)

        log.info(f"\n[Sequential] Agent: {agent_name}")
        log.info(f"[Sequential] Task: {actual_task[:200]}...")

        # ----------------------------------
        # Dynamic Agent Routing
        # ----------------------------------
        # Look up agent from registry (populated at import time via decorators)
        agent_func = agent_registry.get(agent_name)
        if not agent_func:
            # Agent not found - pipeline definition error
            error_msg = f"Unknown agent: {agent_name}"
            log.error(f"[Sequential] {error_msg}")

            # Record failed step and stop pipeline execution
            steps.append(PipelineStep(
                step_number=step_num,
                agent=agent_name,
                task_template=task_template,
                actual_task=actual_task,
                result="",
                error=error_msg
            ))

            success = False
            break  # Stop pipeline - cannot continue without this agent

        # ----------------------------------
        # Execute Agent with Error Handling
        # ----------------------------------
        try:
            # Call the agent with the (placeholder-substituted) task
            result = await agent_func(actual_task)
            # Use summary field if available (e.g., web_search), otherwise final_result
            output = getattr(result, 'summary', result.final_result)
            error_msg = ""

            log.info(f"[Sequential] Result: {str(output)[:200]}...")

        except Exception as e:
            # Agent execution failed - capture error and stop pipeline
            output = ""
            error_msg = str(e)
            success = False
            log.error(f"[Sequential] {error_msg}")

        # ----------------------------------
        # Record Step Execution
        # ----------------------------------
        # Store both template (for traceability) and actual task (after substitution)
        step_record = PipelineStep(
            step_number=step_num,
            agent=agent_name,
            task_template=task_template,      # Original template with placeholders
            actual_task=actual_task,          # After placeholder substitution
            result=str(output),
            error=error_msg
        )
        steps.append(step_record)

        # Add result to previous_results so next step can reference it via {previous}
        previous_results.append(output)

        # Persist to log file for debugging and analysis
        await trace_logger.log(
            step=step_num,
            agent=agent_name,
            task_template=task_template,
            actual_task=actual_task,
            result=str(output)[:500],  # Truncate for logging
            error=error_msg
        )

        # ----------------------------------
        # Early Exit on Failure
        # ----------------------------------
        # Stop pipeline execution if any step fails (fail-fast behavior)
        if error_msg:
            break

    # ----------------------------------
    # Extract Final Result
    # ----------------------------------
    # Last step's output is the final result (unless pipeline failed early)
    final_result = previous_results[-1] if previous_results else "No result"

    log.info(f"\n{'='*80}")
    log.info(f"WORKFLOW COMPLETE")
    log.info(f"Steps executed: {len(steps)}, Success: {success}")
    log.info(f"Final result: {str(final_result)[:200]}...")
    log.info(f"{'='*80}")

    # ----------------------------------
    # Return Complete Execution Trace
    # ----------------------------------
    # Package pipeline name, all step executions, and final result
    return SequentialResult(
        pipeline_name=pipeline_name,          # Which pipeline was executed
        initial_input=user_input,             # Original user input
        steps=steps,                          # Full step-by-step execution trace
        final_result=str(final_result),       # Result from last step
        total_steps=len(steps),
        success=success                       # True if all steps completed, False if any failed
    )


# ----------------------------------
# CLI Entry Point
# ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sequential workflow with predefined agent pipelines",
        epilog=f"Available pipelines: {', '.join(PIPELINES.keys())}\n\n"
               "Example: python -m workflows.sequential --local --pipeline calculate-explain --input 'Calculate 5 factorial'"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run workflow locally using flyte.init() instead of remote execution"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Your input/request"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="write-review-edit",
        choices=list(PIPELINES.keys()),
        help=f"Pipeline to execute (default: write-review-edit)"
    )

    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        log.info("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        log.info("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"\n=== Sequential Multi-Agent Workflow ===")
    log.info(f"Pipeline: {args.pipeline}")
    log.info(f"Input: {args.input}\n")

    # Execute the workflow
    execution = flyte.run(
        sequential_workflow,
        user_input=args.input,
        pipeline_name=args.pipeline
    )

    log.info(f"\n{'='*80}")
    log.info(f"Execution: {execution.name}")
    log.info(f"URL: {execution.url}")
    log.info(f"{'='*80}\n")