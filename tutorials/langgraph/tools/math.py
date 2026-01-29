"""Math tools — reusable across LangGraph agents."""

import logging
from langchain_core.tools import tool
import flyte

log = logging.getLogger(__name__)


@tool
@flyte.trace
async def add(a: float, b: float) -> float:
    """Add two numbers together. Use this for addition."""
    log.info(f"[Math] {a} + {b} = {a + b}")
    return a + b


@tool
@flyte.trace
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers. Use this for multiplication."""
    log.info(f"[Math] {a} * {b} = {a * b}")
    return a * b


@tool
@flyte.trace
async def power(a: float, b: float) -> float:
    """Raise a number to a power. Use this for exponentiation like factorial components or squares."""
    log.info(f"[Math] {a} ** {b} = {a ** b}")
    return a ** b


math_tools = [add, multiply, power]