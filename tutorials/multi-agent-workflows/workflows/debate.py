"""
Debate/Ensemble workflow - multiple agents debate to reach consensus.

This workflow implements the debate pattern where multiple agents:
1. Independently solve the same task in parallel
2. Review each other's solutions
3. Debate and refine through multiple rounds
4. Converge on a final answer via judge synthesis or voting

This improves accuracy and provides diverse perspectives!

Usage:
    python -m workflows.debate --local --request "Calculate 5 factorial" --agents math,math,code --rounds 2
"""

from typing import List, Dict
from dataclasses import dataclass
import flyte
import json
import re
import asyncio

# Auto-import all agent modules to trigger @agent and @tool decorator registration
from agents import import_all_agents
import_all_agents()
from config import base_env, OPENAI_API_KEY
from utils.logger import Logger, setup_logging
from utils.decorators import agent_registry
from openai import AsyncOpenAI

# Initialize trace logger for structured JSONL output
trace_logger = Logger(path="debate_trace_log.jsonl", verbose=False)
# Initialize standard logger for console output
log = setup_logging(__name__)

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class AgentResponse:
    """Single agent's response in a debate round"""
    agent_id: str  # e.g., "math_0", "code_1"
    agent_type: str  # e.g., "math", "code"
    response: str
    confidence: int = 0  # Optional: 1-10 confidence score

@dataclass
class DebateRound:
    """Single round of debate"""
    round_number: int
    responses: List[AgentResponse]
    critiques: List[str]  # Each agent's critique of others' responses

@dataclass
class DebateResult:
    """Final result from debate workflow"""
    task: str
    participating_agents: List[str]
    initial_round: DebateRound
    debate_rounds: List[DebateRound]
    final_synthesis: str
    total_rounds: int
    consensus_achieved: bool

# ----------------------------------
# Debate Orchestrator
# ----------------------------------

env = base_env

@env.task
async def debate_workflow(
    user_task: str,
    agent_names: List[str] = None,
    num_debate_rounds: int = 2,
    synthesis_method: str = "judge"
) -> DebateResult:
    """
    Debate workflow where multiple agents solve the same task and debate solutions.

    Args:
        user_task: The task for all agents to solve
        agent_names: List of agent names to participate (default: ["math", "math", "code"])
        num_debate_rounds: Number of debate rounds after initial response (default: 2)
        synthesis_method: How to reach final answer - "judge" or "vote" (default: "judge")

    Returns:
        DebateResult: Complete debate history and final consensus
    """
    log.info("=" * 80)
    log.info(f"DEBATE WORKFLOW - Task: {user_task}")
    log.info(f"Participants: {agent_names}, Debate rounds: {num_debate_rounds}")
    log.info("=" * 80)

    # ----------------------------------
    # Initialization
    # ----------------------------------
    # Set up LLM client for debate facilitation and final synthesis
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Default configuration: 3 agents (can be same type for diversity via randomness)
    if not agent_names:
        agent_names = ["math", "math", "code"]

    # ----------------------------------
    # Validate Participating Agents
    # ----------------------------------
    # Ensure all requested agents exist in registry
    for agent_name in agent_names:
        if agent_name not in agent_registry:
            available = list(agent_registry.keys())
            raise ValueError(
                f"Unknown agent '{agent_name}'. "
                f"Available agents: {available}"
            )

    num_agents = len(agent_names)
    log.info(f"\n[Debate] {num_agents} agents will participate")

    # ----------------------------------
    # ROUND 0: Independent Initial Responses
    # ----------------------------------
    # All agents solve the same task in parallel WITHOUT seeing each other's work
    # This prevents groupthink and ensures diverse initial perspectives
    log.info(f"\n{'='*80}")
    log.info(f"ROUND 0 - INITIAL RESPONSES")
    log.info(f"{'='*80}")
    log.info(f"\n[Debate] All agents solving task in parallel...")

    # ----------------------------------
    # Collect Initial Responses in Parallel
    # ----------------------------------
    async def get_initial_response(agent_name: str, agent_id: str) -> AgentResponse:
        """Execute single agent to get their independent initial response"""
        agent_func = agent_registry[agent_name]
        result = await agent_func(user_task)
        response_text = getattr(result, 'summary', result.final_result)

        log.info(f"[Debate] {agent_id} ({agent_name}): {str(response_text)[:100]}...")

        return AgentResponse(
            agent_id=agent_id,
            agent_type=agent_name,
            response=str(response_text)
        )

    # Create unique IDs for tracking (e.g., "math_0", "math_1", "code_2")
    agent_ids = [f"{agent_names[i]}_{i}" for i in range(num_agents)]

    # Execute all agents in parallel - no coordination, pure independent thinking
    initial_responses = await asyncio.gather(*[
        get_initial_response(agent_names[i], agent_ids[i])
        for i in range(num_agents)
    ])

    # ----------------------------------
    # Record Round 0 (Initial Responses)
    # ----------------------------------
    initial_round = DebateRound(
        round_number=0,
        responses=initial_responses,
        critiques=[]  # No critiques in round 0
    )

    await trace_logger.log(
        round=0,
        phase="initial_responses",
        num_responses=len(initial_responses)
    )

    # ----------------------------------
    # Setup for Debate Rounds
    # ----------------------------------
    # Track all debate rounds and maintain current responses for next iteration
    debate_rounds = []
    current_responses = initial_responses

    # ----------------------------------
    # ROUNDS 1-N: Iterative Debate and Refinement
    # ----------------------------------
    # Each round: agents see all responses → critique others → refine their own
    for round_num in range(1, num_debate_rounds + 1):
        log.info(f"\n{'='*80}")
        log.info(f"ROUND {round_num} - DEBATE & REFINEMENT")
        log.info(f"{'='*80}")

        # ----------------------------------
        # Build Shared Context for All Agents
        # ----------------------------------
        # Each agent sees ALL responses (including their own) from previous round
        responses_context = "\n\n".join([
            f"{resp.agent_id} ({resp.agent_type}) says:\n{resp.response}"
            for resp in current_responses
        ])

        log.info(f"\n[Debate] Agents reviewing each other's responses...")

        # ----------------------------------
        # Critique and Refinement Phase
        # ----------------------------------
        async def debate_response(agent_name: str, agent_id: str, my_response: str) -> tuple:
            """
            Single agent critiques peer responses and refines their own answer.

            Returns: (critique_text, refined_response_with_confidence)
            """

            # ----------------------------------
            # Peer Review Prompt
            # ----------------------------------
            # Agent sees all responses and must critique others + refine their own
            debate_prompt = f"""You are participating in a multi-agent debate to solve this task:

TASK: {user_task}

All agents' current responses:
{responses_context}

Your previous response was:
{my_response}

Now:
1. Critique the other agents' responses - what are their strengths and weaknesses?
2. Defend or refine your own response based on what you learned
3. Provide your final improved answer

Respond in JSON format:
{{
  "critique": "Your analysis of other responses",
  "refined_response": "Your improved answer to the task",
  "confidence": <1-10>
}}"""

            response = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[{"role": "user", "content": debate_prompt}]
            )

            raw_response = response.choices[0].message.content

            # ----------------------------------
            # Parse Critique and Refined Response
            # ----------------------------------
            # Robust JSON extraction with fallback handling
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    # Try to find any JSON object
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        # Fallback: keep original response if parsing fails
                        data = {
                            "critique": "Could not parse critique",
                            "refined_response": my_response,
                            "confidence": 5
                        }

            # Extract structured feedback from agent
            critique = data.get("critique", "")
            refined = data.get("refined_response", my_response)
            confidence = data.get("confidence", 5)  # Self-reported confidence (1-10)

            log.info(f"[Debate] {agent_id} refined their response (confidence: {confidence}/10)")

            return (
                critique,  # Agent's critique of other responses
                AgentResponse(
                    agent_id=agent_id,
                    agent_type=agent_name,
                    response=refined,        # Improved response after seeing peers
                    confidence=confidence    # Used later for voting synthesis
                )
            )

        # ----------------------------------
        # Execute All Critiques in Parallel
        # ----------------------------------
        # All agents critique and refine simultaneously (no sequential bias)
        debate_results = await asyncio.gather(*[
            debate_response(agent_names[i], agent_ids[i], current_responses[i].response)
            for i in range(num_agents)
        ])

        # ----------------------------------
        # Unpack Debate Results
        # ----------------------------------
        # Separate critiques from refined responses
        critiques = [critique for critique, _ in debate_results]
        refined_responses = [response for _, response in debate_results]

        # ----------------------------------
        # Record This Debate Round
        # ----------------------------------
        round_data = DebateRound(
            round_number=round_num,
            responses=refined_responses,  # Responses AFTER refinement
            critiques=critiques           # What each agent said about others
        )
        debate_rounds.append(round_data)

        # ----------------------------------
        # Update for Next Round
        # ----------------------------------
        # Refined responses become the current responses for next iteration
        current_responses = refined_responses

        # Persist to log file
        await trace_logger.log(
            round=round_num,
            phase="debate",
            num_critiques=len(critiques),
            avg_confidence=sum(r.confidence for r in refined_responses) / len(refined_responses)
        )

    # ----------------------------------
    # FINAL SYNTHESIS: Reach Consensus
    # ----------------------------------
    # After all debate rounds, synthesize final answer from all perspectives
    log.info(f"\n{'='*80}")
    log.info(f"FINAL SYNTHESIS")
    log.info(f"{'='*80}")

    # Collect all final responses with confidence scores
    final_responses_text = "\n\n".join([
        f"{resp.agent_id} ({resp.agent_type}) - Confidence: {resp.confidence}/10\n{resp.response}"
        for resp in current_responses
    ])

    # ----------------------------------
    # Synthesis Method: Vote vs Judge
    # ----------------------------------
    if synthesis_method == "vote":
        # ----------------------------------
        # VOTING: Highest Confidence Wins
        # ----------------------------------
        # Simple democratic approach - agent with highest self-confidence wins
        winner = max(current_responses, key=lambda r: r.confidence)
        final_synthesis = f"Winner by confidence vote: {winner.agent_id}\n\n{winner.response}"
        log.info(f"[Debate] Synthesis method: voting")
        log.info(f"[Debate] Winner: {winner.agent_id} with confidence {winner.confidence}/10")
    else:
        # ----------------------------------
        # JUDGE: LLM Synthesizes Best Parts
        # ----------------------------------
        # Meta-agent combines strongest points from all responses
        judge_prompt = f"""You are a judge synthesizing the final answer from a multi-agent debate.

Original task: {user_task}

Final responses from all agents:
{final_responses_text}

Synthesize the best final answer by:
1. Identifying the strongest points from each agent
2. Resolving any disagreements
3. Providing a clear, accurate final answer

Provide only the final synthesized answer, no meta-commentary."""

        log.info(f"[Debate] Synthesis method: judge")
        judge_response = await client.chat.completions.create(
            model="gpt-4o",
            temperature=0.3,
            messages=[{"role": "user", "content": judge_prompt}]
        )

        final_synthesis = judge_response.choices[0].message.content.strip()
        log.info(f"[Debate] Judge has synthesized final answer")

    log.info(f"\nFinal answer: {final_synthesis[:200]}...")

    # ----------------------------------
    # Consensus Detection
    # ----------------------------------
    # Check if agents converged to similar conclusions (high avg confidence)
    avg_confidence = sum(r.confidence for r in current_responses) / len(current_responses)
    consensus_achieved = avg_confidence >= 7  # Threshold: 7/10 average confidence

    log.info(f"\n{'='*80}")
    log.info(f"WORKFLOW COMPLETE")
    log.info(f"Total rounds: {num_debate_rounds + 1}, Average confidence: {avg_confidence:.1f}/10")
    log.info(f"Consensus achieved: {consensus_achieved}")
    log.info(f"{'='*80}")

    # ----------------------------------
    # Return Complete Debate Trace
    # ----------------------------------
    # Package initial round, all debate rounds, synthesis, and consensus info
    return DebateResult(
        task=user_task,
        participating_agents=agent_names,        # Which agents participated
        initial_round=initial_round,             # Round 0: independent responses
        debate_rounds=debate_rounds,             # Rounds 1-N: critique + refinement
        final_synthesis=final_synthesis,         # Final answer (via vote or judge)
        total_rounds=num_debate_rounds + 1,      # +1 for initial round
        consensus_achieved=consensus_achieved    # True if high avg confidence
    )


# ----------------------------------
# CLI Entry Point
# ----------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Debate workflow with multiple agents reaching consensus",
        epilog="Example: python -m workflows.debate --local --request 'Calculate 5 factorial' --agents math,math,code --rounds 2"
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
        help="The task for all agents to solve"
    )
    parser.add_argument(
        "--agents",
        type=str,
        default="math,math,code",
        help="Comma-separated list of agents (default: math,math,code)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)"
    )
    parser.add_argument(
        "--synthesis",
        type=str,
        choices=["judge", "vote"],
        default="judge",
        help="Synthesis method: judge or vote (default: judge)"
    )

    args = parser.parse_args()

    # Parse agent list
    agent_list = [agent.strip() for agent in args.agents.split(",")]

    # Initialize Flyte based on local/remote flag
    if args.local:
        log.info("Running workflow LOCALLY with flyte.init()")
        flyte.init()
    else:
        log.info("Running workflow REMOTELY with flyte.init_from_config()")
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"\n=== Debate Multi-Agent Workflow ===")
    log.info(f"Task: {args.request}")
    log.info(f"Agents: {agent_list}")
    log.info(f"Debate rounds: {args.rounds}")
    log.info(f"Synthesis: {args.synthesis}\n")

    # Execute the workflow
    execution = flyte.run(
        debate_workflow,
        user_task=args.request,
        agent_names=agent_list,
        num_debate_rounds=args.rounds,
        synthesis_method=args.synthesis
    )

    log.info(f"\n{'='*80}")
    log.info(f"Execution: {execution.name}")
    log.info(f"URL: {execution.url}")
    log.info(f"{'='*80}\n")