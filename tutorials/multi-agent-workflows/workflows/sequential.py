"""
Sequential workflow - execute agents in a predefined order, passing results forward.

Usage:
    python -m workflows.sequential --local --pipeline calculate-explain --input "Calculate 5 factorial"
"""

import flyte
from agents import import_all_agents
import_all_agents()
from config import base_env
from utils.logger import Logger, setup_logging
from utils.decorators import agent_registry

trace_logger = Logger(path="sequential_trace_log.jsonl", verbose=False)
log = setup_logging(__name__)

# Pipelines: list of (agent_name, task_template) tuples.
# {input} = original user input, {previous} = last step's result.
PIPELINES = {
    "write-review-edit": [
        ("code", "Write Python code for: {input}"),
        ("string", "Review this code and count how many lines: {previous}"),
        ("code", "Add docstrings to this code: {previous}"),
    ],
    "search-summarize": [
        ("web_search", "Search for: {input}"),
        ("string", "Count words in this search result: {previous}"),
    ],
    "calculate-explain": [
        ("math", "{input}"),
        ("string", "Analyze this number and describe it: {previous}"),
    ],
    "weather-analyze": [
        ("weather", "Get weather for: {input}"),
        ("string", "Count the words in this weather report: {previous}"),
    ],
    "multi-calc": [
        ("math", "Calculate: {input}"),
        ("math", "Multiply the previous result by 2: {previous}"),
        ("math", "Add 10 to the previous result: {previous}"),
    ],
}

env = base_env

@env.task
async def sequential_workflow(user_input: str, pipeline_name: str = "write-review-edit") -> dict:
    """Execute agents in a predefined order, passing each result to the next step."""
    if pipeline_name not in PIPELINES:
        raise ValueError(f"Unknown pipeline '{pipeline_name}'. Available: {list(PIPELINES.keys())}")

    pipeline = PIPELINES[pipeline_name]
    log.info(f"Sequential pipeline '{pipeline_name}' ({len(pipeline)} steps) | Input: {user_input}")

    previous = ""
    steps = []

    for i, (agent_name, template) in enumerate(pipeline):
        task = template.replace("{input}", user_input).replace("{previous}", str(previous))
        log.info(f"Step {i}: {agent_name} | {task[:120]}")

        agent_func = agent_registry.get(agent_name)
        if not agent_func:
            log.error(f"Unknown agent: {agent_name}")
            steps.append({"step": i, "agent": agent_name, "error": f"Unknown agent: {agent_name}"})
            return {"pipeline_name": pipeline_name, "steps": steps, "final_result": "", "success": False}

        try:
            result = await agent_func(task)
            previous = getattr(result, 'summary', result.final_result)
            log.info(f"  Result: {str(previous)[:150]}")
            steps.append({"step": i, "agent": agent_name, "task": task, "result": str(previous)})
        except Exception as e:
            log.error(f"  Error: {e}")
            steps.append({"step": i, "agent": agent_name, "task": task, "error": str(e)})
            return {"pipeline_name": pipeline_name, "steps": steps, "final_result": "", "success": False}

        await trace_logger.log(step=i, agent=agent_name, task=task, result=str(previous)[:500])

    log.info(f"Done. Final result: {str(previous)[:150]}")
    return {"pipeline_name": pipeline_name, "steps": steps, "final_result": str(previous), "success": True}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequential agent pipeline")
    parser.add_argument("--local", action="store_true", help="Run locally with flyte.init()")
    parser.add_argument("--input", type=str, required=True, help="Your input/request")
    parser.add_argument("--pipeline", type=str, default="write-review-edit",
                        choices=list(PIPELINES.keys()), help="Pipeline to execute")
    args = parser.parse_args()

    if args.local:
        flyte.init()
    else:
        flyte.init_from_config(".flyte/config.yaml")

    log.info(f"Pipeline: {args.pipeline} | Input: {args.input}")
    execution = flyte.run(sequential_workflow, user_input=args.input, pipeline_name=args.pipeline)
    log.info(f"Execution: {execution.name} | URL: {execution.url}")
