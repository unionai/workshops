"""
Sequential workflow - a self-contained write → review → edit pipeline.

Three agents defined inline, executed in order, each passing its result to the next.

Usage:
    python -m workflows.sequential --local --input "Write a function to check if a number is prime"
"""

import flyte
from openai import AsyncOpenAI
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY
from utils.logger import Logger, setup_logging

trace_logger = Logger(path="sequential_trace_log.jsonl", verbose=False)
log = setup_logging(__name__)
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4.1-nano"

# ------------------------------------------------------------------
# Agents — each is a simple Flyte task that calls the LLM
# ------------------------------------------------------------------

env = base_env

@dataclass
class AgentResult:
    output: str
    error: str = ""


@env.task
async def writer_agent(task: str) -> AgentResult:
    """Write code based on the user's request."""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Python developer. Write clean, working code. Return ONLY the code, no explanation."},
            {"role": "user", "content": task},
        ],
    )
    return AgentResult(output=response.choices[0].message.content)


@env.task
async def reviewer_agent(task: str) -> AgentResult:
    """Review code and provide feedback."""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a code reviewer. Give concise, actionable feedback on correctness, style, and edge cases."},
            {"role": "user", "content": task},
        ],
    )
    return AgentResult(output=response.choices[0].message.content)


@env.task
async def editor_agent(task: str) -> AgentResult:
    """Edit code based on review feedback."""
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Python developer. Apply the requested changes to the code. Return ONLY the improved code, no explanation."},
            {"role": "user", "content": task},
        ],
    )
    return AgentResult(output=response.choices[0].message.content)


# ------------------------------------------------------------------
# Pipeline — just a list of (name, agent_func, prompt_template) tuples
# ------------------------------------------------------------------

PIPELINE = [
    ("writer", writer_agent, "Write Python code for: {input}"),
    ("reviewer", reviewer_agent, "Review this code:\n\n{previous}"),
    ("editor", editor_agent, "Here is the code:\n\n{first}\n\nApply this feedback:\n\n{previous}"),
]


# ------------------------------------------------------------------
# Sequential orchestrator
# ------------------------------------------------------------------

@env.task
async def sequential_workflow(user_input: str) -> dict:
    """Execute the write → review → edit pipeline."""
    log.info(f"Sequential pipeline | Input: {user_input}")

    results = {}
    previous = ""

    for i, (name, agent_func, template) in enumerate(PIPELINE):
        task = (template
                .replace("{input}", user_input)
                .replace("{previous}", str(previous))
                .replace("{first}", str(results.get("writer", ""))))

        log.info(f"Step {i} ({name}): {task[:120]}")

        try:
            result = await agent_func(task)
            previous = result.output
            results[name] = previous
            log.info(f"  Result: {previous[:150]}")
        except Exception as e:
            log.error(f"  Error: {e}")
            return {"steps": list(results.keys()), "final_result": "", "success": False, "error": str(e)}

        await trace_logger.log(step=i, agent=name, task=task, result=previous[:500])

    log.info(f"Done.")
    return {"steps": list(results.keys()), "final_result": previous, "success": True}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequential write → review → edit pipeline")
    parser.add_argument("--local", action="store_true", help="Run locally with flyte.init()")
    parser.add_argument("--request", type=str, required=True, help="What code to write")
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"Input: {args.request}")
    execution = flyte.run(sequential_workflow, user_input=args.request)
    log.info(f"Execution: {execution.name} | URL: {execution.url}")
