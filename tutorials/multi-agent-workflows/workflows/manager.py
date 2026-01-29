"""
Manager-Worker workflow - hierarchical coordination with active supervision.

This workflow implements the manager-worker pattern where a manager agent:
1. Analyzes the task and delegates to specialist workers
2. Monitors worker outputs and provides feedback
3. Requests revisions if quality issues found
4. Coordinates between workers to resolve conflicts
5. Synthesizes final deliverable from all worker outputs

Unlike planner (static planning) or debate (peer collaboration), this uses
hierarchical management with active supervision throughout execution.

Usage:
    python -m workflows.manager --local --request "Build a REST API for user management"
"""

from typing import List
from dataclasses import dataclass
import flyte
import json
import re

# Auto-import all agent modules to trigger @agent and @tool decorator registration
from agents import import_all_agents
import_all_agents()
from config import base_env, OPENAI_API_KEY
from utils.logger import Logger, setup_logging
from utils.decorators import agent_registry
from openai import AsyncOpenAI

# Initialize trace logger for structured JSONL output
trace_logger = Logger(path="manager_trace_log.jsonl", verbose=False)
# Initialize standard logger for console output
log = setup_logging(__name__)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WorkerTask:
    """Task delegated to a worker"""
    task_id: int
    agent: str
    description: str
    dependencies: List[int]

@dataclass
class ManagerReview:
    """Manager's review of worker output"""
    approved: bool
    quality_score: int
    issues: List[str]
    feedback: str

@dataclass
class WorkerResult:
    """Result from worker including revisions"""
    task_id: int
    agent: str
    description: str
    initial_output: str
    final_output: str
    revisions: List[str]
    review: ManagerReview

@dataclass
class ManagerResult:
    """Final result from manager-worker workflow"""
    project: str
    delegation_plan: List[WorkerTask]
    worker_results: List[WorkerResult]
    final_synthesis: str
    total_tasks: int
    total_revisions: int
    success: bool

# ----------------------------------
# Manager-Worker Orchestrator
# ----------------------------------

env = base_env

@env.task
async def manager_workflow(
    user_request: str,
    quality_threshold: int = 7,
    max_revisions_per_task: int = 2
) -> ManagerResult:
    """
    Manager-worker workflow with hierarchical coordination and active supervision.

    Args:
        user_request: The project/task to accomplish
        quality_threshold: Minimum quality score (1-10) to approve worker output
        max_revisions_per_task: Maximum revision cycles per worker task

    Returns:
        ManagerResult: Complete execution with manager reviews and final synthesis
    """
    log.info("=" * 80)
    log.info(f"MANAGER-WORKER WORKFLOW")
    log.info(f"Project: {user_request}")
    log.info(f"Quality threshold: {quality_threshold}/10")
    log.info("=" * 80)

    # ----------------------------------
    # Initialization
    # ----------------------------------
    # Set up LLM client for manager agent (planning, reviewing, synthesis)
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Available specialist workers that manager can delegate to
    available_workers = list(agent_registry.keys())

    # ----------------------------------
    # PHASE 1: Manager Creates Delegation Plan
    # ----------------------------------
    # Manager analyzes project and breaks it into worker tasks with dependencies
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 1 - MANAGER PLANNING")
    log.info(f"{'='*80}")

    planning_prompt = f"""You are a manager agent coordinating specialist workers.

Project: {user_request}

Available workers:
{chr(10).join([f"- {worker}" for worker in available_workers])}

Break this project into discrete tasks for workers. For each task:
1. Choose appropriate worker
2. Write clear task description
3. Identify dependencies (which tasks must complete first)

Respond in JSON format:
{{
  "tasks": [
    {{
      "task_id": 0,
      "agent": "worker_name",
      "description": "specific task for this worker",
      "dependencies": []
    }},
    ...
  ]
}}

Keep tasks focused and manageable. Use dependencies to ensure proper ordering."""

    log.info("\n[Manager] Analyzing project and creating delegation plan...")

    planning_response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": planning_prompt}]
    )

    raw_plan = planning_response.choices[0].message.content

    # Parse plan with robust JSON extraction
    try:
        plan_data = json.loads(raw_plan)
    except json.JSONDecodeError:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_plan, re.DOTALL)
        if json_match:
            plan_data = json.loads(json_match.group(1))
        else:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_plan, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(0))
            else:
                raise ValueError(f"Could not parse manager's plan")

    # Create WorkerTask objects
    delegation_plan = [
        WorkerTask(
            task_id=task["task_id"],
            agent=task["agent"],
            description=task["description"],
            dependencies=task.get("dependencies", [])
        )
        for task in plan_data["tasks"]
    ]

    log.info(f"\n[Manager] Created plan with {len(delegation_plan)} task(s):")
    for task in delegation_plan:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else " (no dependencies)"
        log.info(f"  Task {task.task_id}: {task.agent} - {task.description[:60]}...{deps}")

    await trace_logger.log(
        phase="planning",
        num_tasks=len(delegation_plan)
    )

    # ----------------------------------
    # PHASE 2: Supervised Execution with Quality Gates
    # ----------------------------------
    # Manager delegates tasks to workers, reviews outputs, and requests revisions if needed
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 2 - SUPERVISED EXECUTION")
    log.info(f"{'='*80}")

    worker_results = []
    completed_tasks = {}

    # ----------------------------------
    # Dependency-Aware Execution Loop
    # ----------------------------------
    # Execute tasks respecting dependencies (similar to planner workflow)
    pending_tasks = list(delegation_plan)

    while pending_tasks:
        # ----------------------------------
        # Identify Ready Tasks (Dependencies Satisfied)
        # ----------------------------------
        # Partition pending tasks into: ready to execute vs waiting on dependencies
        ready_tasks = []
        remaining_tasks = []

        for task in pending_tasks:
            deps_satisfied = all(dep_id in completed_tasks for dep_id in task.dependencies)
            if deps_satisfied:
                ready_tasks.append(task)
            else:
                remaining_tasks.append(task)

        # Circular dependency detection
        if not ready_tasks:
            log.error("[Manager] No tasks ready but pending tasks remain (circular dependency?)")
            break

        # ----------------------------------
        # Worker Delegation with Manager Supervision
        # ----------------------------------
        # Execute ready tasks sequentially (manager reviews each output before delegating next)
        for task in ready_tasks:
            log.info(f"\n[Manager] Delegating Task {task.task_id} to {task.agent} worker...")
            log.info(f"[Manager] Task: {task.description}")

            # ----------------------------------
            # Context Injection - How Results Flow to Workers
            # ----------------------------------
            # If this task depends on previous tasks, inject their results into the prompt
            context = ""
            if task.dependencies:
                context = "\n\nContext from previous tasks:\n"
                for dep_id in task.dependencies:
                    context += f"Task {dep_id} result: {completed_tasks[dep_id][:200]}...\n"

            worker_task = task.description + context

            # ----------------------------------
            # Dynamic Worker Routing
            # ----------------------------------
            # Look up worker agent from registry (populated at import time via decorators)
            worker_func = agent_registry.get(task.agent)
            if not worker_func:
                # Unknown worker - the manager hallucinated or requested invalid agent
                log.error(f"[Manager] Unknown worker '{task.agent}'")
                worker_results.append(WorkerResult(
                    task_id=task.task_id,
                    agent=task.agent,
                    description=task.description,
                    initial_output="",
                    final_output="",
                    revisions=[f"Error: Unknown worker {task.agent}"],
                    review=ManagerReview(
                        approved=False,
                        quality_score=0,
                        issues=[f"Unknown worker: {task.agent}"],
                        feedback="Cannot execute"
                    )
                ))
                completed_tasks[task.task_id] = ""
                continue

            # ----------------------------------
            # Worker Executes Initial Task
            # ----------------------------------
            # Worker produces first-pass response (may require revision based on manager feedback)
            result = await worker_func(worker_task)
            current_output = getattr(result, 'summary', result.final_result)
            initial_output = str(current_output)

            log.info(f"[Worker {task.agent}] Completed initial output: {str(current_output)[:150]}...")

            # ----------------------------------
            # Manager Review and Revision Cycle
            # ----------------------------------
            # Manager evaluates quality and requests improvements until approval or max revisions
            revisions_history = []

            # ----------------------------------
            # Iterative Review Loop
            # ----------------------------------
            # Manager reviews output → Worker revises → Repeat until quality threshold met
            for revision_num in range(max_revisions_per_task + 1):
                # ----------------------------------
                # Manager Quality Assessment
                # ----------------------------------
                # Manager evaluates worker output on quality score (1-10) and identifies issues
                review_prompt = f"""You are a manager reviewing a worker's output.

                    Original task: {task.description}
                    Worker output: {current_output}

                    Evaluate the output:
                    1. Quality score (1-10)
                    2. List any issues or problems
                    3. Approve (true/false) - approve if score >= {quality_threshold}
                    4. Feedback for improvement (if not approved)

                    Respond in JSON format:
                    {{
                    "quality_score": <1-10>,
                    "issues": ["issue 1", "issue 2"],
                    "approved": true/false,
                    "feedback": "detailed feedback if not approved"
                    }}"""

                log.info(f"\n[Manager] Reviewing worker output...")

                review_response = await client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.3,
                    messages=[{"role": "user", "content": review_prompt}]
                )

                raw_review = review_response.choices[0].message.content

                # ----------------------------------
                # Parse Manager's Quality Assessment
                # ----------------------------------
                # Robust JSON extraction handles markdown wrapping
                try:
                    review_data = json.loads(raw_review)
                except json.JSONDecodeError:
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_review, re.DOTALL)
                    if json_match:
                        review_data = json.loads(json_match.group(1))
                    else:
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_review, re.DOTALL)
                        if json_match:
                            review_data = json.loads(json_match.group(0))
                        else:
                            # Fallback
                            review_data = {
                                "quality_score": quality_threshold,
                                "issues": [],
                                "approved": True,
                                "feedback": ""
                            }

                review = ManagerReview(
                    approved=review_data["approved"],
                    quality_score=review_data["quality_score"],
                    issues=review_data.get("issues", []),
                    feedback=review_data.get("feedback", "")
                )

                log.info(f"[Manager] Quality score: {review.quality_score}/10")
                if review.issues:
                    log.info(f"[Manager] Issues found: {', '.join(review.issues)}")

                # ----------------------------------
                # Approval Decision - Check Quality Threshold
                # ----------------------------------
                # If quality meets threshold, accept output and move to next task
                if review.approved:
                    log.info(f"[Manager] APPROVED - Task {task.task_id} meets quality standards")
                    break
                elif revision_num < max_revisions_per_task:
                    # ----------------------------------
                    # Revision Request - Worker Improves Output
                    # ----------------------------------
                    # Manager provides specific feedback; worker produces revised output
                    log.info(f"[Manager] NEEDS REVISION - Requesting improvements...")
                    log.info(f"[Manager] Feedback: {review.feedback}")

                    # Build revision task with original task + previous output + manager feedback
                    revision_task = f"""Original task: {task.description}

Your previous output:
{current_output}

Manager feedback:
{review.feedback}

Please revise your output to address the manager's feedback."""

                    result = await worker_func(revision_task)
                    current_output = getattr(result, 'summary', result.final_result)
                    revisions_history.append(review.feedback)

                    log.info(f"[Worker {task.agent}] Submitted revision: {str(current_output)[:150]}...")
                else:
                    # ----------------------------------
                    # Max Revisions Reached - Accept Current Output
                    # ----------------------------------
                    # After max revision cycles, accept output even if below quality threshold
                    log.warning(f"[Manager] Max revisions reached - accepting current output")
                    review.approved = True  # Accept despite issues
                    break

            # ----------------------------------
            # Record Worker Result
            # ----------------------------------
            # Store complete execution history: initial output, final output, all revisions, and review
            worker_result = WorkerResult(
                task_id=task.task_id,
                agent=task.agent,
                description=task.description,
                initial_output=initial_output,
                final_output=str(current_output),
                revisions=revisions_history,
                review=review
            )
            worker_results.append(worker_result)
            completed_tasks[task.task_id] = str(current_output)

            # Log task completion
            await trace_logger.log(
                task_id=task.task_id,
                agent=task.agent,
                quality_score=review.quality_score,
                num_revisions=len(revisions_history),
                approved=review.approved
            )

        # Update pending tasks list (remove completed, keep waiting)
        pending_tasks = remaining_tasks

    # ----------------------------------
    # PHASE 3: Final Synthesis
    # ----------------------------------
    # Manager integrates all worker outputs into coherent final deliverable
    log.info(f"\n{'='*80}")
    log.info(f"PHASE 3 - FINAL SYNTHESIS")
    log.info(f"{'='*80}")

    # ----------------------------------
    # Collect All Worker Outputs
    # ----------------------------------
    # Aggregate final outputs from all workers (post-revision) for synthesis
    all_outputs = "\n\n".join([
        f"Task {wr.task_id} ({wr.agent}): {wr.description}\nOutput: {wr.final_output}"
        for wr in worker_results
    ])

    # ----------------------------------
    # Manager Synthesizes Final Deliverable
    # ----------------------------------
    # LLM combines all worker outputs into unified, coherent final result
    synthesis_prompt = f"""You are a manager synthesizing the final deliverable.

        Original project: {user_request}

        All worker outputs:
        {all_outputs}

        Create a coherent final deliverable that:
        1. Integrates all worker outputs
        2. Ensures consistency and quality
        3. Presents a complete solution to the project

        Provide the final synthesized deliverable:"""

    log.info("\n[Manager] Synthesizing final deliverable from all worker outputs...")

    synthesis_response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    final_synthesis = synthesis_response.choices[0].message.content.strip()

    log.info(f"\n[Manager] Final deliverable: {final_synthesis[:200]}...")

    # ----------------------------------
    # Calculate Success Metrics
    # ----------------------------------
    # Count total revisions across all tasks and check if all workers were approved
    total_revisions = sum(len(wr.revisions) for wr in worker_results)
    success = all(wr.review.approved for wr in worker_results)

    log.info(f"\n{'='*80}")
    log.info(f"PROJECT COMPLETE")
    log.info(f"Tasks: {len(worker_results)}, Total revisions: {total_revisions}")
    log.info(f"Success: {success}")
    log.info(f"{'='*80}")

    # ----------------------------------
    # Return Complete Execution Trace
    # ----------------------------------
    # Package delegation plan, all worker results with revisions, and final synthesis
    return ManagerResult(
        project=user_request,
        delegation_plan=delegation_plan,
        worker_results=worker_results,
        final_synthesis=final_synthesis,
        total_tasks=len(worker_results),
        total_revisions=total_revisions,
        success=success
    )


# ----------------------------------
# CLI Entry Point
# ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manager-worker workflow with hierarchical coordination",
        epilog="Example: python -m workflows.manager --local --request 'Build a calculator with add, subtract, multiply'"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run workflow locally using flyte.init() instead of remote execution"
    )
    parser.add_argument(
        "--request",
        type=str,
        required=True,
        help="The project/task to accomplish"
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=7,
        help="Minimum quality score (1-10) to approve outputs (default: 7)"
    )
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=2,
        help="Maximum revision cycles per task (default: 2)"
    )

    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        log.info("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        log.info("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"\n=== Manager-Worker Multi-Agent Workflow ===")
    log.info(f"Project: {args.request}")
    log.info(f"Quality threshold: {args.quality_threshold}/10")
    log.info(f"Max revisions per task: {args.max_revisions}\n")

    # Execute the workflow
    execution = flyte.run(
        manager_workflow,
        user_request=args.request,
        quality_threshold=args.quality_threshold,
        max_revisions_per_task=args.max_revisions
    )

    log.info(f"\n{'='*80}")
    log.info(f"Execution: {execution.name}")
    log.info(f"URL: {execution.url}")
    log.info(f"{'='*80}\n")
