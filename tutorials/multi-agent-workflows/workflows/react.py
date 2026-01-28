"""
ReAct (Reason + Act) workflow - adaptive agent execution with reflection.

This workflow implements the ReAct pattern where the planner:
1. Observes the current state
2. Reasons about what to do next
3. Acts by executing ONE agent
4. Reflects on the result
5. Decides if the goal is achieved or continues

This is much more flexible than the static planning approach!

Usage:
    python -m workflows.react --local --request "Your goal here"
"""

import sys
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
import flyte
import asyncio
import json

# Add workflows directory to Python path for imports
workflows_dir = Path(__file__).parent
sys.path.insert(0, str(workflows_dir))

# Import agents (imports register them in agent_registry via decorators)
from agents.math_agent import math_agent, MathAgentResult
from agents.string_agent import string_agent, StringAgentResult
from agents.web_search_agent import web_search_agent, WebSearchAgentResult
from agents.code_agent import code_agent, CodeAgentResult
from agents.weather_agent import weather_agent, WeatherAgentResult
from config import base_env, OPENAI_API_KEY
from utils.logger import Logger
from utils.decorators import agent_registry
from openai import AsyncOpenAI

# Initialize logger
logger = Logger(path="react_trace_log.jsonl", verbose=False)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class ReActStep:
    """Single step in ReAct execution"""
    step_number: int
    thought: str          # Reasoning about what to do
    action_agent: str     # Which agent to call
    action_task: str      # What task to give the agent
    observation: str      # Result from the agent
    reflection: str       # Analysis of the result


@dataclass
class ReActResult:
    """Final result from ReAct workflow"""
    goal: str
    steps: List[ReActStep]
    final_answer: str
    total_steps: int
    goal_achieved: bool


# ----------------------------------
# ReAct Orchestrator
# ----------------------------------

env = base_env

@env.task
async def react_workflow(user_goal: str, max_steps: int = 10) -> ReActResult:
    """
    ReAct workflow that iteratively plans and executes until goal is achieved.

    Args:
        user_goal: The user's goal to accomplish
        max_steps: Maximum number of steps to prevent infinite loops

    Returns:
        ReActResult: Complete execution trace and final answer
    """
    print("=" * 80)
    print(f"ReAct WORKFLOW - Goal: {user_goal}")
    print("=" * 80)

    # ----------------------------------
    # Initialization
    # ----------------------------------
    # Set up LLM client and tracking structures for the iterative loop
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Track all steps in the ReAct cycle (Thought → Action → Observation → Reflection)
    steps: List[ReActStep] = []
    context_history = []  # Sliding window of recent steps for LLM context
    goal_achieved = False

    # Available specialist agents that can be called during execution
    available_agents = ["math", "string", "web_search", "code", "weather"]

    # ----------------------------------
    # Main ReAct Loop
    # ----------------------------------
    # Each iteration: Reason about what to do → Act → Observe result → Reflect on progress
    for step_num in range(1, max_steps + 1):
        print(f"\n{'='*80}")
        print(f"STEP {step_num}")
        print(f"{'='*80}")

        # ----------------------------------
        # Build Context from Previous Steps
        # ----------------------------------
        # Include last 3 steps to prevent context window explosion while maintaining continuity
        if context_history:
            history_text = "\n\n".join([
                f"Step {s['step']}: {s['thought']}\n"
                f"Action: {s['action_agent']} - {s['action_task']}\n"
                f"Result: {s['observation']}\n"
                f"Reflection: {s['reflection']}"
                for s in context_history[-3:]  # Sliding window: only last 3 steps
            ])
        else:
            history_text = "No previous steps."

        # ----------------------------------
        # THOUGHT Phase - Reason About Next Action
        # ----------------------------------
        # LLM decides: What should I do next to achieve the goal?
        system_msg = f"""You are a ReAct agent using the Reason + Act pattern.

        Goal: {user_goal}

        Available agents:
        {chr(10).join([f"- {agent}" for agent in available_agents])}

        Previous steps:
        {history_text}

        Your task: Decide what to do next to achieve the goal.

        Respond in JSON format:
        {{
        "thought": "Your reasoning about what to do next and why",
        "action_agent": "agent_name",
        "action_task": "specific task for the agent",
        "goal_achieved": false,
        "final_answer": null
        }}

        OR if the goal is achieved:
        {{
        "thought": "Why I believe the goal is now achieved",
        "action_agent": null,
        "action_task": null,
        "goal_achieved": true,
        "final_answer": "The complete answer to the user's goal"
        }}

        IMPORTANT:
        - Think step-by-step
        - Only do ONE action at a time
        - Set goal_achieved=true ONLY when you have the final answer
        - Be specific about what you want the agent to do
        """

        print("\n[ReAct] Reasoning about next action...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Goal: {user_goal}\n\nWhat should we do next?"}
            ]
        )

        # ----------------------------------
        # Parse LLM Decision
        # ----------------------------------
        # Robust JSON extraction handles LLMs wrapping JSON in markdown or text
        raw_response = response.choices[0].message.content

        # Try direct JSON parse first
        try:
            decision = json.loads(raw_response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group(1))
            else:
                # Try to find any JSON object in the response
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
                if json_match:
                    decision = json.loads(json_match.group(0))
                else:
                    # Last resort: log and fail gracefully
                    print(f"[ERROR] Could not parse JSON from response: {raw_response}")
                    raise ValueError(f"LLM did not return valid JSON. Response: {raw_response[:200]}")

        # Extract reasoning and goal status from decision
        thought = decision["thought"]
        goal_achieved = decision["goal_achieved"]

        print(f"\n💭 Thought: {thought}")

        # ----------------------------------
        # Check for Goal Completion
        # ----------------------------------
        # If LLM determines goal is achieved, exit the loop with final answer
        if goal_achieved:
            final_answer = decision["final_answer"]
            print(f"\n✅ Goal achieved!")
            print(f"📝 Final answer: {final_answer}")
            break

        # ----------------------------------
        # ACTION Phase - Execute Agent
        # ----------------------------------
        # LLM chose an agent and task, now we execute it
        action_agent = decision["action_agent"]
        action_task = decision["action_task"]

        print(f"\n🎯 Action: Call {action_agent} agent")
        print(f"📋 Task: {action_task}")

        # Dynamic agent routing using registry (populated at import time)
        agent_func = agent_registry.get(action_agent)
        if not agent_func:
            observation = f"ERROR: Unknown agent '{action_agent}'"
        else:
            result = await agent_func(action_task)
            # Use summary field if available (e.g., web_search), otherwise use final_result
            observation = getattr(result, 'summary', result.final_result)

        print(f"\n📊 Observation: {observation[:200]}{'...' if len(observation) > 200 else ''}")

        # ----------------------------------
        # REFLECTION Phase - Evaluate Progress
        # ----------------------------------
        # LLM reflects on whether the action was helpful and if we're closer to the goal
        reflection_prompt = f"""Based on this action and result, reflect on:
        1. Was this action helpful?
        2. Did we get the information we need?
        3. Are we closer to the goal?

        Action: {action_agent} - {action_task}
        Result: {observation}

        Provide a brief reflection (1-2 sentences)."""

        reflection_response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=150,
            messages=[
                {"role": "user", "content": reflection_prompt}
            ]
        )

        reflection = reflection_response.choices[0].message.content.strip()
        print(f"\n🤔 Reflection: {reflection}")

        # ----------------------------------
        # Record Step in Execution History
        # ----------------------------------
        # Store complete step data (Thought → Action → Observation → Reflection)
        step_record = ReActStep(
            step_number=step_num,
            thought=thought,
            action_agent=action_agent,
            action_task=action_task,
            observation=observation,
            reflection=reflection
        )
        steps.append(step_record)  # Full history for final result

        # Add to sliding context window (for next iteration's LLM prompt)
        context_history.append({
            "step": step_num,
            "thought": thought,
            "action_agent": action_agent,
            "action_task": action_task,
            "observation": observation,
            "reflection": reflection
        })

        # Persist to log file for debugging and analysis
        await logger.log(
            step=step_num,
            thought=thought,
            action_agent=action_agent,
            action_task=action_task,
            observation=observation[:500],  # Truncate for logging
            reflection=reflection
        )

    # ----------------------------------
    # Handle Loop Completion
    # ----------------------------------
    # If we exited the loop without achieving goal (hit max_steps), provide fallback answer
    if not goal_achieved:
        print(f"\n⚠️  Reached maximum steps ({max_steps}) without achieving goal")
        final_answer = f"Could not achieve goal in {max_steps} steps. Last observation: {observation}"

    print(f"\n{'='*80}")
    print(f"WORKFLOW COMPLETE - {len(steps)} steps executed")
    print(f"{'='*80}")

    # ----------------------------------
    # Return Complete Execution Trace
    # ----------------------------------
    # Package all steps and final answer into structured result for caller
    return ReActResult(
        goal=user_goal,
        steps=steps,                    # Full Thought→Action→Observation→Reflection history
        final_answer=final_answer,      # Final answer (from LLM or fallback)
        total_steps=len(steps),
        goal_achieved=goal_achieved     # True if LLM declared success, False if hit max_steps
    )


# ----------------------------------
# CLI Entry Point
# ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ReAct workflow with adaptive planning",
        epilog="Example: python -m workflows.react --local --request 'Find GDP of France and Germany'"
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
        help="Your goal/request"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
        help="Maximum steps (default: 10)"
    )

    args = parser.parse_args()

    # Initialize Flyte based on local/remote flag
    if args.local:
        print("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        print("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    print(f"\n=== ReAct Multi-Agent Workflow ===")
    print(f"Goal: {args.request}")
    print(f"Max steps: {args.max_steps}\n")

    # Execute the workflow
    execution = flyte.run(
        react_workflow,
        user_goal=args.request,
        max_steps=args.max_steps
    )

    print(f"\n{'='*80}")
    print(f"Execution: {execution.name}")
    print(f"URL: {execution.url}")
    print("Click the link above to view execution details in the Flyte UI")
    print(f"{'='*80}\n")