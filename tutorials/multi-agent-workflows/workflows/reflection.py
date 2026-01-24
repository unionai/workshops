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
logger = Logger(path="reflection_trace_log.jsonl", verbose=False)

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
    print("=" * 80)
    print(f"REFLECTION WORKFLOW - Task: {user_task}")
    print(f"Quality threshold: {quality_threshold}/10, Max iterations: {max_iterations}")
    print("=" * 80)

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Step 1: Determine which agent to use
    print("\n[Reflection] Step 1: Selecting appropriate agent...")

    available_agents = ["math", "string", "web_search", "code", "weather"]

    agent_selection_prompt = f"""Given this task, which agent is most appropriate?

Task: {user_task}

Available agents:
- math: Arithmetic, calculations, mathematical operations
- string: Text analysis, word counting, string manipulation
- web_search: Web searches, fetching webpages, finding information online
- code: Python code execution, programming tasks
- weather: Weather information lookup

Respond with ONLY the agent name (e.g., "math")."""

    agent_response = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": agent_selection_prompt}]
    )

    selected_agent = agent_response.choices[0].message.content.strip().lower()
    print(f"[Reflection] Selected agent: {selected_agent}")

    # Step 2: Get initial response from agent
    print(f"\n[Reflection] Step 2: Getting initial response from {selected_agent} agent...")

    agent_func = agent_registry.get(selected_agent)
    if not agent_func:
        raise ValueError(f"Unknown agent: {selected_agent}")

    result = await agent_func(user_task)
    current_response = getattr(result, 'summary', result.final_result)
    initial_response = current_response

    print(f"[Reflection] Initial response: {current_response[:200]}...")

    # Step 3: Iterative reflection and refinement
    iterations = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}")

        # Reflect on current response
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

        print("\n[Reflection] Evaluating response quality...")
        reflection_response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[{"role": "user", "content": reflection_prompt}]
        )

        # Parse reflection with robust JSON extraction
        raw_reflection = reflection_response.choices[0].message.content

        try:
            reflection_data = json.loads(raw_reflection)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_reflection, re.DOTALL)
            if json_match:
                reflection_data = json.loads(json_match.group(1))
            else:
                # Try to find any JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_reflection, re.DOTALL)
                if json_match:
                    reflection_data = json.loads(json_match.group(0))
                else:
                    print(f"[ERROR] Could not parse reflection JSON: {raw_reflection}")
                    raise ValueError(f"Could not parse reflection response")

        quality_score = reflection_data["quality_score"]
        issues = reflection_data["issues"]
        suggestions = reflection_data["suggestions"]

        print(f"\n📊 Quality Score: {quality_score}/10")
        print(f"🔍 Issues Found: {len(issues)}")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")

        # Check if we've met quality threshold
        if quality_score >= quality_threshold:
            print(f"\n✅ Quality threshold met! ({quality_score} >= {quality_threshold})")
            converged = True

            # Record final iteration
            iterations.append(ReflectionIteration(
                iteration=iteration,
                response=current_response,
                reflection=raw_reflection,
                quality_score=quality_score,
                issues_found=issues,
                improvements_made="Quality threshold achieved - no further refinement needed"
            ))

            break

        # Refine the response
        print(f"\n🔧 Refining response based on feedback...")

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

        # Record this iteration
        improvements_made = f"Addressed: {', '.join(issues[:3])}" if issues else "General refinement"
        iterations.append(ReflectionIteration(
            iteration=iteration,
            response=current_response,
            reflection=raw_reflection,
            quality_score=quality_score,
            issues_found=issues,
            improvements_made=improvements_made
        ))

        print(f"📝 Refined response: {refined_response[:200]}...")

        # Update current response for next iteration
        current_response = refined_response

        # Log to file
        await logger.log(
            iteration=iteration,
            quality_score=quality_score,
            issues_count=len(issues),
            issues=issues,
            suggestions=suggestions[:200],  # Truncate for logging
            response_length=len(current_response)
        )

    # Handle case where we didn't converge
    if not converged:
        print(f"\n⚠️  Reached maximum iterations ({max_iterations}) without meeting quality threshold")
        final_quality = iterations[-1].quality_score if iterations else 0
    else:
        final_quality = quality_score

    final_response = current_response

    print(f"\n{'='*80}")
    print(f"WORKFLOW COMPLETE")
    print(f"Iterations: {len(iterations)}, Final Quality: {final_quality}/10")
    print(f"Converged: {converged}")
    print(f"{'='*80}")

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
        print("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        print("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    print(f"\n=== Reflection Multi-Agent Workflow ===")
    print(f"Task: {args.request}")
    print(f"Quality threshold: {args.quality_threshold}/10")
    print(f"Max iterations: {args.max_iterations}\n")

    # Execute the workflow
    execution = flyte.run(
        reflection_workflow,
        user_task=args.request,
        quality_threshold=args.quality_threshold,
        max_iterations=args.max_iterations
    )

    print(f"\n{'='*80}")
    print(f"Execution: {execution.name}")
    print(f"URL: {execution.url}")
    print("Click the link above to view execution details in the Flyte UI")
    print(f"{'='*80}\n")