"""
This module defines the writer_agent, which can create written content based on research.
"""

import flyte
from openai import AsyncOpenAI
import os

from utils.decorators import agent
from dataclasses import dataclass
from config import base_env, OPENAI_API_KEY

# ----------------------------------
# Agent-Specific Configuration
# ----------------------------------
WRITER_AGENT_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 1500,
}

# ----------------------------------
# Data Models
# ----------------------------------

@dataclass
class WriterAgentResult:
    """Result from writer agent execution"""
    final_result: str
    error: str = ""  # Empty if no error


# ----------------------------------
# Writer Agent Task Environment
# ----------------------------------
env = base_env
# Future: If you need agent-specific dependencies, create separate environments:
# env = flyte.TaskEnvironment(
#     name="code_agent_env",
#     image=base_env.image.with_pip_packages(["numpy", "pandas"]),
#     secrets=base_env.secrets,
#     resources=flyte.Resources(cpu=2, mem="4Gi")
# )

@env.task(
    retries=3,  # Retry up to 3 times on failure
    timeout=60,  # Timeout after 60 seconds
)
@agent("writer")
async def writer_agent(task: str) -> WriterAgentResult:
    """
    Writer agent that creates written content based on research and requirements.
    Uses LLM directly to generate content without intermediate tools.

    This task demonstrates Flyte's automatic retry capability - if it fails,
    Flyte will automatically retry up to 3 times.

    Args:
        task (str): The writing task to perform (should include research context).

    Returns:
        WriterAgentResult: The written content.
    """
    print(f"[Writer Agent] Processing: {task[:100]}...")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    system_msg = """
You are a professional content writer. Your job is to create well-structured, engaging content based on the research provided.

Write clear, informative content that:
- Has a compelling title (using # for markdown)
- Is well-organized with sections (using ## for subheadings)
- Synthesizes the research into coherent paragraphs
- Is approximately 200-400 words
- Uses proper markdown formatting

Return ONLY the written content, no preamble or explanation.
"""

    try:
        response = await client.chat.completions.create(
            model=WRITER_AGENT_CONFIG["model"],
            temperature=WRITER_AGENT_CONFIG["temperature"],
            max_tokens=WRITER_AGENT_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": task}
            ]
        )

        content = response.choices[0].message.content
        print(f"[Writer Agent] Generated {len(content)} characters of content")

        return WriterAgentResult(
            final_result=content,
            error=""
        )
    except Exception as e:
        print(f"[Writer Agent] Error: {str(e)}")
        return WriterAgentResult(
            final_result="",
            error=str(e)
        )
