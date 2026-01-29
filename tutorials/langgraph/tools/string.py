"""String tools — reusable across LangGraph agents."""

import logging
from langchain_core.tools import tool
import flyte

log = logging.getLogger(__name__)


@tool
@flyte.trace
async def word_count(text: str) -> int:
    """Count the number of words in a string."""
    count = len(text.split())
    log.info(f"[String] word_count: {count}")
    return count


@tool
@flyte.trace
async def letter_count(text: str) -> int:
    """Count the number of alphabetic characters in a string."""
    count = sum(c.isalpha() for c in text)
    log.info(f"[String] letter_count: {count}")
    return count


@tool
@flyte.trace
async def reverse_string(text: str) -> str:
    """Reverse a string."""
    result = text[::-1]
    log.info(f"[String] reverse: {result[:50]}")
    return result


string_tools = [word_count, letter_count, reverse_string]