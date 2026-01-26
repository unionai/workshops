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

import sys
from pathlib import Path
from typing import List
from dataclasses import dataclass
import flyte
import json

# Add workflows directory to Python path for imports
workflows_dir = Path(__file__).parent
sys.path.insert(0, str(workflows_dir))

# Import agents (imports register them in agent_registry via decorators)
from agents.math_agent import math_agent
from agents.string_agent import string_agent
from agents.web_search_agent import web_search_agent
from agents.code_agent import code_agent
from agents.weather_agent import weather_agent
from config import base_env, OPENAI_API_KEY
from utils.logger import Logger
from utils.decorators import agent_registry
from openai import AsyncOpenAI

# Initialize logger
logger = Logger(path="manager_trace_log.jsonl", verbose=False)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WorkerTask:
    """Task delegated to a worker"""
    task_id: int
    agent: str
    description: str
    dependencies: List[int]  # Task IDs this depends on

@dataclass
class ManagerReview:
    """Manager's review of worker output"""
    approved: bool
    quality_score: int  # 1-10
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
    revisions: List[str]  # History of revision feedback
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
    print("=" * 80)
    print(f"MANAGER-WORKER WORKFLOW")
    print(f"Project: {user_request}")
    print(f"Quality threshold: {quality_threshold}/10")
    print("=" * 80)

    # Initialize OpenAI client for manager agent
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    available_workers = ["math", "string", "web_search", "code", "weather"]

    # Step 1: Manager analyzes and creates delegation plan
    print(f"\n{'='*80}")
    print(f"PHASE 1 - MANAGER PLANNING")
    print(f"{'='*80}")

    planning_prompt = f"""You are a manager agent coordinating specialist workers.

Project: {user_request}

Available workers:
{chr(10).join([f"- {worker}: {worker} specialist" for worker in available_workers])}

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

    print("\n[Manager] Analyzing project and creating delegation plan...")

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
        import re
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

    print(f"\n[Manager] Created plan with {len(delegation_plan)} task(s):")
    for task in delegation_plan:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else " (no dependencies)"
        print(f"  Task {task.task_id}: {task.agent} - {task.description[:60]}...{deps}")

    await logger.log(
        phase="planning",
        num_tasks=len(delegation_plan)
    )

    # Step 2: Execute tasks with manager supervision
    print(f"\n{'='*80}")
    print(f"PHASE 2 - SUPERVISED EXECUTION")
    print(f"{'='*80}")

    worker_results = []
    completed_tasks = {}  # task_id -> result

    # Execute tasks respecting dependencies
    pending_tasks = list(delegation_plan)

    while pending_tasks:
        # Find tasks ready to execute (dependencies met)
        ready_tasks = []
        remaining_tasks = []

        for task in pending_tasks:
            deps_satisfied = all(dep_id in completed_tasks for dep_id in task.dependencies)
            if deps_satisfied:
                ready_tasks.append(task)
            else:
                remaining_tasks.append(task)

        if not ready_tasks:
            print("[Manager] ERROR: No tasks ready but pending tasks remain (circular dependency?)")
            break

        # Execute ready tasks (could parallelize but we'll do sequential for manager review)
        for task in ready_tasks:
            print(f"\n[Manager] Delegating Task {task.task_id} to {task.agent} worker...")
            print(f"[Manager] Task: {task.description}")

            # Build context from dependencies
            context = ""
            if task.dependencies:
                context = "\n\nContext from previous tasks:\n"
                for dep_id in task.dependencies:
                    context += f"Task {dep_id} result: {completed_tasks[dep_id][:200]}...\n"

            worker_task = task.description + context

            # Get worker agent
            worker_func = agent_registry.get(task.agent)
            if not worker_func:
                print(f"[Manager] ERROR: Unknown worker '{task.agent}'")
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

            # Worker executes task
            result = await worker_func(worker_task)
            current_output = getattr(result, 'summary', result.final_result)
            initial_output = str(current_output)

            print(f"[Worker {task.agent}] Completed initial output: {str(current_output)[:150]}...")

            # Manager reviews and potentially requests revisions
            revisions_history = []

            for revision_num in range(max_revisions_per_task + 1):
                # Manager reviews worker output
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

                print(f"\n[Manager] Reviewing worker output...")

                review_response = await client.chat.completions.create(
                    model="gpt-4o",
                    temperature=0.3,
                    messages=[{"role": "user", "content": review_prompt}]
                )

                raw_review = review_response.choices[0].message.content

                # Parse review
                try:
                    review_data = json.loads(raw_review)
                except json.JSONDecodeError:
                    import re
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

                print(f"[Manager] Quality score: {review.quality_score}/10")
                if review.issues:
                    print(f"[Manager] Issues found: {', '.join(review.issues)}")

                if review.approved:
                    print(f"[Manager] ✅ APPROVED - Task {task.task_id} meets quality standards")
                    break
                elif revision_num < max_revisions_per_task:
                    print(f"[Manager] ❌ NEEDS REVISION - Requesting improvements...")
                    print(f"[Manager] Feedback: {review.feedback}")

                    # Request revision from worker
                    revision_task = f"""Original task: {task.description}

Your previous output:
{current_output}

Manager feedback:
{review.feedback}

Please revise your output to address the manager's feedback."""

                    result = await worker_func(revision_task)
                    current_output = getattr(result, 'summary', result.final_result)
                    revisions_history.append(review.feedback)

                    print(f"[Worker {task.agent}] Submitted revision: {str(current_output)[:150]}...")
                else:
                    print(f"[Manager] ⚠️  Max revisions reached - accepting current output")
                    review.approved = True  # Accept despite issues
                    break

            # Store worker result
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
            await logger.log(
                task_id=task.task_id,
                agent=task.agent,
                quality_score=review.quality_score,
                num_revisions=len(revisions_history),
                approved=review.approved
            )

        pending_tasks = remaining_tasks

    # Step 3: Manager synthesizes final deliverable
    print(f"\n{'='*80}")
    print(f"PHASE 3 - FINAL SYNTHESIS")
    print(f"{'='*80}")

    all_outputs = "\n\n".join([
        f"Task {wr.task_id} ({wr.agent}): {wr.description}\nOutput: {wr.final_output}"
        for wr in worker_results
    ])

    synthesis_prompt = f"""You are a manager synthesizing the final deliverable.

Original project: {user_request}

All worker outputs:
{all_outputs}

Create a coherent final deliverable that:
1. Integrates all worker outputs
2. Ensures consistency and quality
3. Presents a complete solution to the project

Provide the final synthesized deliverable:"""

    print("\n[Manager] Synthesizing final deliverable from all worker outputs...")

    synthesis_response = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[{"role": "user", "content": synthesis_prompt}]
    )

    final_synthesis = synthesis_response.choices[0].message.content.strip()

    print(f"\n[Manager] Final deliverable: {final_synthesis[:200]}...")

    total_revisions = sum(len(wr.revisions) for wr in worker_results)
    success = all(wr.review.approved for wr in worker_results)

    print(f"\n{'='*80}")
    print(f"PROJECT COMPLETE")
    print(f"Tasks: {len(worker_results)}, Total revisions: {total_revisions}")
    print(f"Success: {success}")
    print(f"{'='*80}")

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
        print("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        print("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    print(f"\n=== Manager-Worker Multi-Agent Workflow ===")
    print(f"Project: {args.request}")
    print(f"Quality threshold: {args.quality_threshold}/10")
    print(f"Max revisions per task: {args.max_revisions}\n")

    # Execute the workflow
    execution = flyte.run(
        manager_workflow,
        user_request=args.request,
        quality_threshold=args.quality_threshold,
        max_revisions_per_task=args.max_revisions
    )

    print(f"\n{'='*80}")
    print(f"Execution: {execution.name}")
    print(f"URL: {execution.url}")
    print("Click the link above to view execution details in the Flyte UI")
    print(f"{'='*80}\n")