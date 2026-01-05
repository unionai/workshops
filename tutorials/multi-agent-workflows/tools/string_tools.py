from utils.decorators import tool
import flyte


@tool(agent="string")
@flyte.trace
async def word_count(s: str) -> int:
    """Calculates the number of words in the given string.

    Args:
        s (str): The input string to analyze.

    Returns:
        int: The total count of words in the string.
    """
    return len(s.split())


@tool(agent="string")
@flyte.trace
async def letter_count(s: str) -> int:
    """Calculates the number of alphabetic characters in the given string.

    Args:
        s (str): The input string to analyze.

    Returns:
        int: The total count of alphabetic characters in the string.
    """
    return sum(c.isalpha() for c in s)