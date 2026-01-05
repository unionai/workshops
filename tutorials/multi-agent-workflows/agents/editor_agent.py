"""
This module defines the editor_agent, which can review and improve written content.
"""

import flyte
from openai import AsyncOpenAI

from utils.decorators import agent
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY

# ----------------------------------
# Agent-Specific Configuration
# ----------------------------------
EDITOR_AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.5,
    "max_tokens": 1500,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class EditorAgentResult:
    """Result from editor agent execution"""
    final_result: str
    error: str = ""  # Empty if no error


# ----------------------------------
# Editor Agent Task Environment
# ----------------------------------
env = base_env
# Future: If you need agent-specific dependencies, create separate environments:
# env = flyte.TaskEnvironment(
#     name="code_agent_env",
#     image=base_env.image.with_pip_packages(["numpy", "pandas"]),
#     secrets=base_env.secrets,
#     resources=flyte.Resources(cpu=2, mem="4Gi")
# )


@env.task
@agent("editor")
async def editor_agent(task: str) -> EditorAgentResult:
    """
    Editor agent that reviews and improves written content.
    Uses LLM directly to analyze and enhance content quality.

    Args:
        task (str): The editing task to perform (should include content to review).

    Returns:
        EditorAgentResult: The improved content.
    """
    print(f"[Editor Agent] Processing: {task}")

    # Initialize client inside task for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_msg = """
You are a professional content editor. Your job is to review and improve written content.

Review the content for:
- Clarity and readability
- Proper structure and flow
- Grammar and style
- Completeness and accuracy

Then return an IMPROVED version of the content that:
- Fixes any issues you found
- Enhances clarity and engagement
- Maintains the original intent and key information
- Keeps proper markdown formatting

Return ONLY the improved content, no commentary or explanation about changes.
"""

    try:
        response = await client.chat.completions.create(
            model=EDITOR_AGENT_CONFIG["model"],
            temperature=EDITOR_AGENT_CONFIG["temperature"],
            max_tokens=EDITOR_AGENT_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": task}
            ]
        )

        improved_content = response.choices[0].message.content
        print(f"[Editor Agent] Generated {len(improved_content)} characters of improved content")

        return EditorAgentResult(
            final_result=improved_content,
            error=""
        )
    except Exception as e:
        print(f"[Editor Agent] Error: {str(e)}")
        return EditorAgentResult(
            final_result="",
            error=str(e)
        )
