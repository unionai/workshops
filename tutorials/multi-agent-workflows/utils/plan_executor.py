import json
import asyncio
from utils.logger import Logger
from utils.decorators import agent_tools, tool_registry

logger = Logger()


def parse_plan_from_response(raw_plan: str) -> list:
    """
    Parse a tool execution plan from LLM response.
    Handles both clean JSON and responses wrapped in markdown code blocks.

    Args:
        raw_plan: Raw text response from LLM

    Returns:
        list: Parsed plan as list of tool call dictionaries

    Raises:
        ValueError: If plan cannot be parsed
    """
    # Try to parse directly first
    try:
        return json.loads(raw_plan)
    except json.JSONDecodeError as e:
        # If that fails, try to extract JSON from markdown code blocks or surrounding text
        import re

        print(f"[WARN] Direct JSON parse failed: {e}")
        print(f"[WARN] Attempting to extract JSON from response...")

        # Try to find JSON within markdown code blocks first
        code_block_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw_plan, re.DOTALL)
        if code_block_match:
            try:
                plan = json.loads(code_block_match.group(1))
                print("[INFO] Successfully extracted JSON from markdown code block")
                return plan
            except json.JSONDecodeError:
                pass

        # If no code block, try to find any JSON array in the text
        json_match = re.search(r'\[(?:[^\[\]]*|\{[^}]*\})*\]', raw_plan, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group(0))
                print("[INFO] Successfully extracted JSON array from text")
                return plan
            except json.JSONDecodeError as e2:
                print(f"[ERROR] Extracted text is not valid JSON: {e2}")
                print(f"[ERROR] Extracted text:\n{json_match.group(0)}")
                print(f"[ERROR] Full LLM response:\n{raw_plan}")
                raise ValueError(f"Could not extract valid JSON array from LLM response")
        else:
            print(f"[ERROR] Could not find JSON array pattern in LLM response:\n{raw_plan}")
            raise ValueError(f"Could not extract valid JSON array from LLM response")


async def execute_tool_plan(plan: list, agent: str) -> dict:
    """
    Execute a plan by calling tools in sequence.
    This is the core execution logic - agents call their LLM to get the plan,
    then pass it here for execution.

    Args:
        plan: List of tool calls [{"tool": "name", "args": [...], "reasoning": "..."}]
        agent: Which agent's toolset to use

    Returns:
        dict: {"final_result": ..., "steps": [...]} or {"error": ..., "steps": [...]}
    """
    toolset = agent_tools.get(agent, tool_registry)

    steps_log = []
    last_result = None

    #----------------------------------
    # Execute the plan
    #----------------------------------
    try:
        for step in plan:
            tool_name = step["tool"]
            args = step["args"]
            reasoning = step.get("reasoning", "")
            args = [last_result if str(a).lower() == "previous" else a for a in args]

            if tool_name in toolset:
                # Tools are now async, so we need to await them
                tool_func = toolset[tool_name]
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(*args)
                else:
                    result = tool_func(*args)
            else:
                await logger.log(tool=tool_name, args=args, error="Unknown tool", reasoning=reasoning)
                raise ValueError(f"Unknown tool: {tool_name}")

            await logger.log(tool=tool_name, args=args, result=result, reasoning=reasoning)
            steps_log.append({"tool": tool_name, "args": args, "result": result, "reasoning": reasoning})
            last_result = result

        return {"final_result": last_result, "steps": steps_log}

    except Exception as e:
        await logger.log(tool=tool_name if "tool_name" in locals() else "unknown", args=args if "args" in locals() else [], error=str(e))
        return {"error": str(e), "steps": steps_log}