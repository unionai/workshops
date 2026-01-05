"""
Smart context summarization utility using LLM when needed.
"""

from openai import AsyncOpenAI
from config import OPENAI_API_KEY

# Configuration
SUMMARIZER_CONFIG = {
    "model": "gpt-4o-mini",  # Cheap and fast for summarization
    "temperature": 0.3,       # Focused but not completely deterministic
    "max_tokens": 400,        # Target summary length
}

# Thresholds
MIN_LENGTH_TO_SUMMARIZE = 800  # Only summarize if longer than this
TARGET_SUMMARY_LENGTH = 500    # Aim for summaries around this length


async def smart_summarize(text: str, context: str = "general") -> str:
    """
    Intelligently summarize text using LLM if it exceeds threshold.
    Falls back to simple truncation if LLM fails.

    Args:
        text: The text to summarize
        context: Context hint for summarization (e.g., "web_search", "code_output")

    Returns:
        Summarized text or original if below threshold
    """
    # If text is already short, return as-is
    if len(text) <= MIN_LENGTH_TO_SUMMARIZE:
        return text

    print(f"[Summarizer] Input length: {len(text)} chars, using LLM summarization...")

    # Initialize client inside function for Flyte secret injection
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Build context-specific prompt
    if context == "web_search":
        focus = "Preserve all URLs, titles, key facts, and numbers. Focus on the most relevant search results."
    elif context == "code_output":
        focus = "Preserve key output values, error messages, and important results."
    else:
        focus = "Preserve the most important information and key details."

    system_msg = f"""You are a precise summarization assistant. Your job is to create concise summaries that preserve critical information.

{focus}

Rules:
- Keep the summary under {TARGET_SUMMARY_LENGTH} characters
- Preserve specific numbers, dates, URLs, and named entities
- Maintain factual accuracy - never invent information
- Use clear, direct language
- Remove redundant or filler content

Return ONLY the summary, no preamble."""

    try:
        response = await client.chat.completions.create(
            model=SUMMARIZER_CONFIG["model"],
            temperature=SUMMARIZER_CONFIG["temperature"],
            max_tokens=SUMMARIZER_CONFIG["max_tokens"],
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Summarize this:\n\n{text}"}
            ]
        )

        summary = response.choices[0].message.content.strip()
        print(f"[Summarizer] Output length: {len(summary)} chars (reduced by {len(text) - len(summary)} chars)")

        return summary

    except Exception as e:
        print(f"[Summarizer] LLM summarization failed: {e}, falling back to truncation")
        return _fallback_summarize(text)


def _fallback_summarize(text: str, max_chars: int = TARGET_SUMMARY_LENGTH) -> str:
    """
    Fallback summarization using sentence-aware truncation.
    Used when LLM summarization fails.

    Args:
        text: Text to summarize
        max_chars: Maximum character length

    Returns:
        Truncated text at sentence boundary
    """
    if len(text) <= max_chars:
        return text

    # Split into sentences (simple approach)
    sentences = []
    for delimiter in ['. ', '! ', '? ', '\n\n']:
        text = text.replace(delimiter, '.|SPLIT|')

    parts = text.split('|SPLIT|')
    summary = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check if adding this sentence would exceed limit
        potential_length = len(summary) + len(part) + 2
        if potential_length > max_chars and summary:
            # Already have some content, stop here
            break

        summary += part + ". "

    # If we got at least one sentence, return it
    if summary:
        result = summary.strip()
        if not result.endswith('...'):
            result += "..."
        return result

    # Last resort: truncate at last space
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > max_chars * 0.8:  # Only use if we're not losing too much
        return truncated[:last_space] + "..."

    return truncated + "..."