from utils.decorators import tool
import flyte
from typing import Union

# Type alias for numeric inputs - LLMs might pass strings, ints, or floats
Numeric = Union[int, float, str]


@tool(agent="math")
@flyte.trace
async def add(a: Numeric, b: Numeric) -> float:
    """
    Adds two numbers together.

    Args:
        a: The first number (can be int, float, or numeric string).
        b: The second number (can be int, float, or numeric string).

    Returns:
        float: The sum of the two numbers.
    """
    a_val = float(a)
    b_val = float(b)
    print(f"[Math Tool] Adding {a_val} + {b_val}")
    return a_val + b_val


@tool(agent="math")
@flyte.trace
async def multiply(a: Numeric, b: Numeric) -> float:
    """
    Multiplies two numbers.

    Args:
        a: The first number (can be int, float, or numeric string).
        b: The second number (can be int, float, or numeric string).

    Returns:
        float: The product of the two numbers.
    """
    a_val = float(a)
    b_val = float(b)
    print(f"[Math Tool] Multiplying {a_val} * {b_val}")
    return a_val * b_val


@tool(agent="math")
@flyte.trace
async def power(a: Numeric, b: Numeric) -> float:
    """
    Raises a number to the power of another number.

    Args:
        a: The base number (can be int, float, or numeric string).
        b: The exponent (can be int, float, or numeric string).

    Returns:
        float: The result of raising the base to the given exponent.
    """
    a_val = float(a)
    b_val = float(b)
    print(f"[Math Tool] Calculating {a_val} ** {b_val}")
    return a_val ** b_val