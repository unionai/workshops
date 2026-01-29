"""
Reflection workflow - iterative self-improvement through critique.

This workflow implements the reflection pattern where agents:
1. Generate initial response to the task
2. Reflect on the quality/correctness of the response
3. Refine based on reflection feedback
4. Repeat until satisfactory or max iterations

This enables self-improvement and higher quality outputs!

Usage:
    python -m workflows.reflection --local --request "Your task here"
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
trace_logger = Logger(path="reflection_trace_log.jsonl", verbose=False)
# Initialize standard logger for console output
log = setup_logging(__name__)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class ReflectionIteration:
    """Single iteration of generate → reflect → refine cycle"""
    iteration: int
    response: str
    reflection: str
    quality_score: int  # 1-10 scale
    issues_found: List[str]
    improvements_made: str

@dataclass
class ReflectionResult:
    """Final result from reflection workflow"""
    task: str
    agent_used: str
    initial_response: str
    final_response: str
    iterations: List[ReflectionIteration]
    total_iterations: int
    final_quality_score: int
    converged: bool

# ----------------------------------
# Reflection Orchestrator
# ----------------------------------

env = base_env

@env.task
async def reflection_workflow(
    user_task: str,
    quality_threshold: int = 8,
    max_iterations: int = 5
) -> ReflectionResult:
    """
    Reflection workflow that iteratively improves agent outputs through self-critique.

    Args:
        user_task: The task to accomplish
        quality_threshold: Minimum quality score (1-10) to consider satisfactory
        max_iterations: Maximum refinement iterations

    Returns:
        ReflectionResult: Complete iteration history and final refined output
    """
    log.info("=" * 80)
    log.info(f"REFLECTION WORKFLOW - Task: {user_task}")
    log.info(f"Quality threshold: {quality_threshold}/10, Max iterations: {max_iterations}")
    log.info("=" * 80)

    # ----------------------------------
    # Initialization
    # ----------------------------------
    # Set up LLM client for critique and refinement
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # ----------------------------------
    # PHASE 1: Agent Selection
    # ----------------------------------
    # LLM analyzes task and selects most appropriate specialist agent
    log.info("\n[Reflection] Step 1: Selecting appropriate agent...")

    available_agents = list(agent_registry.keys())
    agent_selection_prompt = f"""Given this task, which agent is most appropriate?

        Task: {user_task}

        Available agents: {', '.join(available_agents)}

        Respond with ONLY the agent name (e.g., "math")."""

    agent_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": agent_selection_prompt}]
    )

    selected_agent = agent_response.choices[0].message.content.strip().lower()
    log.info(f"[Reflection] Selected agent: {selected_agent}")

    # ----------------------------------
    # PHASE 2: Initial Response Generation
    # ----------------------------------
    # Selected agent produces first-pass response (may be imperfect)
    log.info(f"\n[Reflection] Step 2: Getting initial response from {selected_agent} agent...")

    # Look up agent from registry (populated at import time)
    agent_func = agent_registry.get(selected_agent)
    if not agent_func:
        raise ValueError(f"Unknown agent: {selected_agent}")

    # Execute agent to get initial response
    result = await agent_func(user_task)
    current_response = getattr(result, 'summary', result.final_result)
    initial_response = current_response 

    log.info(f"[Reflection] Initial response: {current_response[:200]}...")

    # ----------------------------------
    # PHASE 3: Iterative Critique and Refinement Loop
    # ----------------------------------
    # Repeatedly: Critique → Check Quality → Refine (until threshold met or max iterations)
    iterations = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        log.info(f"\n{'='*80}")
        log.info(f"ITERATION {iteration}")
        log.info(f"{'='*80}")

        # ----------------------------------
        # Critique Current Response
        # ----------------------------------
        # LLM acts as critic, evaluating quality and identifying specific issues
        reflection_prompt = f"""You are a critic evaluating the quality of a response.

            Task: {user_task}
            Current Response: {current_response}

            Analyze this response and provide:
            1. Quality score (1-10, where 10 is perfect)
            2. List of specific issues or areas for improvement
            3. Suggestions for refinement

            Respond in JSON format:
            {{
            "quality_score": <1-10>,
            "issues": ["issue 1", "issue 2", ...],
            "suggestions": "Detailed suggestions for improvement"
            }}

            Be critical but constructive. If the response is excellent, give it a high score and minimal issues."""

        log.info(f"\n[Reflection] Current response: {current_response[:300]}{'...' if len(current_response) > 300 else ''}")
        log.info("[Reflection] Evaluating response quality...")
        reflection_response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[{"role": "user", "content": reflection_prompt}]
        )

        # ----------------------------------
        # Parse Quality Assessment
        # ----------------------------------
        # Robust JSON extraction handles markdown wrapping
        raw_reflection = reflection_response.choices[0].message.content

        try:
            reflection_data = json.loads(raw_reflection)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_reflection, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(1))
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_reflection, re.DOTALL)
                if json_match:
                    reflection_data = json.loads(json_match.group(0))
                else:
                    log.error(f"Could not parse reflection JSON: {raw_reflection}")
                    raise ValueError(f"Could not parse reflection response")

        # Extract structured feedback from critique
        quality_score = reflection_data["quality_score"]
        issues = reflection_data["issues"]
        suggestions = reflection_data["suggestions"]

        log.info(f"\nQuality Score: {quality_score}/10")
        log.info(f"Issues Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            log.info(f"   {i}. {issue}")

        # ----------------------------------
        # Check Quality Threshold (Convergence Criteria)
        # ----------------------------------
        # If quality is satisfactory, exit loop - we're done!
        if quality_score >= quality_threshold:
            log.info(f"\nQuality threshold met! ({quality_score} >= {quality_threshold})")
            converged = True

            # Record this final iteration (no refinement needed)
            iterations.append(ReflectionIteration(
                iteration=iteration,
                response=current_response,
                reflection=raw_reflection,
                quality_score=quality_score,
                issues_found=issues,
                improvements_made="Quality threshold achieved - no further refinement needed"
            ))

            break

        # ----------------------------------
        # Refinement Phase - Address Critique Feedback
        # ----------------------------------
        # LLM generates improved version addressing all identified issues
        log.info(f"\nRefining response based on feedback...")

        refinement_prompt = f"""You are refining a response based on critical feedback.

Original Task: {user_task}
Current Response: {current_response}

Critique:
- Quality Score: {quality_score}/10
- Issues: {', '.join(issues)}
- Suggestions: {suggestions}

Generate an improved response that addresses all the issues and suggestions.
Respond with ONLY the improved response, no explanations or metadata."""

        refinement_response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[{"role": "user", "content": refinement_prompt}]
        )

        refined_response = refinement_response.choices[0].message.content.strip()

        # ----------------------------------
        # Record Iteration History
        # ----------------------------------
        # Store critique, issues, and what was improved (for final result traceability)
        improvements_made = f"Addressed: {', '.join(issues[:3])}" if issues else "General refinement"
        iterations.append(ReflectionIteration(
            iteration=iteration,
            response=current_response,
            reflection=raw_reflection,
            quality_score=quality_score,
            issues_found=issues,
            improvements_made=improvements_made
        ))

        log.info(f"Refined response: {refined_response[:200]}...")

        # ----------------------------------
        # Update for Next Iteration
        # ----------------------------------
        # The refined response becomes the new current_response for next critique cycle
        current_response = refined_response

        # Persist to log file for debugging and analysis
        await trace_logger.log(
            iteration=iteration,
            quality_score=quality_score,
            issues_count=len(issues),
            issues=issues,
            suggestions=suggestions[:200],
            response_length=len(current_response)
        )

    # ----------------------------------
    # Handle Loop Completion
    # ----------------------------------
    # Either converged (quality threshold met) or hit max iterations
    if not converged:
        log.warning(f"Reached maximum iterations ({max_iterations}) without meeting quality threshold")
        final_quality = iterations[-1].quality_score if iterations else 0
    else:
        final_quality = quality_score

    final_response = current_response

    log.info(f"\n{'='*80}")
    log.info(f"WORKFLOW COMPLETE")
    log.info(f"Iterations: {len(iterations)}, Final Quality: {final_quality}/10")
    log.info(f"Converged: {converged}")
    log.info(f"{'='*80}")

    # ----------------------------------
    # Return Complete Iteration Trace
    # ----------------------------------
    # Package initial response, all iterations, and final refined output
    return ReflectionResult(
        task=user_task,
        agent_used=selected_agent,
        initial_response=initial_response,
        final_response=final_response,
        iterations=iterations,
        total_iterations=len(iterations),
        final_quality_score=final_quality,
        converged=converged
    )


# ----------------------------------
# CLI Entry Point
# ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reflection workflow with iterative self-improvement",
        epilog="Example: python -m workflows.reflection --local --request 'Calculate the factorial of 5 and explain the result'"
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
        help="Your task/request"
    )
    parser.add_argument(
        "--quality-threshold",
        type=int,
        default=8,
        help="Minimum quality score (1-10) to accept (default: 8)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum refinement iterations (default: 5)"
    )

    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        log.info("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        log.info("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"\n=== Reflection Multi-Agent Workflow ===")
    log.info(f"Task: {args.request}")
    log.info(f"Quality threshold: {args.quality_threshold}/10")
    log.info(f"Max iterations: {args.max_iterations}\n")

    # Execute the workflow
    execution = flyte.run(
        reflection_workflow,
        user_task=args.request,
        quality_threshold=args.quality_threshold,
        max_iterations=args.max_iterations
    )

    log.info(f"\n{'='*80}")
    log.info(f"Execution: {execution.name}")
    log.info(f"URL: {execution.url}")
    log.info(f"{'='*80}\n")