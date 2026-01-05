from utils.decorators import tool
import flyte

@tool(agent="math")
@flyte.trace
async def add(a: int|float, b: int|float) -> float:
    """
    Adds two numbers together.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The sum of the two numbers.
    """
    print(f"TOOL CALL: Adding {a} and {b}")
    return float(a + b)


@tool(agent="math")
@flyte.trace
async def multiply(a, b) -> float:
    """
    Multiplies two numbers.

    Args:
        a (int or float): The first number.
        b (int or float): The second number.

    Returns:
        int or float: The product of the two numbers.
    """
    return float(a * b)

@tool(agent="math")
@flyte.trace
async def power(a, b) -> float:
    """
    Raises a number to the power of another number.

    Args:
        a (int or float): The base number.
        b (int or float): The exponent.

    Returns:
        int or float: The result of raising the base to the given exponent.
    """
    return float(a ** b)