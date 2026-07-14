"""Talking to the model: the callback, the sandbox rules, and the usage tally.

`Agent` takes `call_llm` as a plain async callback, which is why this file can talk
to Claude with the official SDK rather than routing through a gateway. It is also
the only provider-specific code in the tutorial: swap `counting_llm()` for any
chat-completions endpoint and nothing else changes.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from flyte.ai.agents._llm import LLMMessage

MODEL = os.getenv("CODE_MODE_MODEL", "claude-opus-4-8")
MAX_TOKENS = 16_000


# Appended to the SDK's ORCHESTRATOR_SYNTAX_PROMPT, which covers the forbidden
# statements but not these. Models trip over all three reliably.
SANDBOX_RULES = """\
Further sandbox rules:
- EVERY PROGRAM RUNS IN A FRESH SANDBOX. Nothing persists between programs — not
  variables, not query results, nothing. A name you defined last turn does not
  exist this turn. So do the whole job in ONE program: fetch, compute, render, and
  return. Do not plan to "query now and analyse next turn"; there is no next turn
  that can see what you fetched.
- Primitives have no methods. Use `str(x)`, `round(x, 2)`, `len(x)` — never
  `x.__str__()`, `x.format(...)`, or `"a".join(...)`.
- The only method you may call is `.append()` on a list.
- Build strings with `+` and `str()`. There are no f-strings and no `%` formatting.
- `None` is common in query results (a borough with no matching trips). Check for
  it with `if x is None: continue` before doing arithmetic on it.
"""


@dataclass
class Usage:
    """What one agent run cost, measured at the LLM boundary."""

    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    tool_calls: int = 0
    # Every program, not just the last: `AgentResult.code` only keeps the final one,
    # which hides the work whenever the model takes more than one turn.
    programs: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# The agent speaks the OpenAI convention (`tool_calls`, `role: "tool"`); these two
# translate it to Anthropic and back. Only step 4's sequential agent needs them —
# code mode never passes tools.


def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-convention messages -> Anthropic content blocks."""
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")

        if role == "tool":
            # Anthropic carries tool results as a user turn, keyed back by id.
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": str(msg.get("content", "")),
            }
            # Consecutive results belong in one user turn.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content: list[dict[str, Any]] = []
            if msg.get("content"):
                content.append({"type": "text", "text": msg["content"]})
            for call in msg["tool_calls"]:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args,
                    }
                )
            out.append({"role": "assistant", "content": content})
            continue

        text = msg.get("content") or ""  # Anthropic rejects empty content
        if text:
            out.append({"role": role or "user", "content": text})

    return out


def _to_anthropic_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI function schemas -> Anthropic tool schemas."""
    return [
        {
            "name": (fn := tool.get("function", tool)).get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        }
        for tool in tools or []
    ]


def counting_llm() -> tuple[callable, Usage]:
    """An LLM callback, plus the tally of what it spends.

    A "turn" is one round-trip to the model — the number step 4 is about.
    """
    import anthropic

    client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    usage = Usage()

    async def call_llm(model, system, messages, tools) -> LLMMessage:
        request: dict[str, Any] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": _to_anthropic(messages),
            "thinking": {"type": "adaptive"},
        }
        if tools:
            request["tools"] = _to_anthropic_tools(tools)

        start = time.perf_counter()
        response = await client.messages.create(**request)

        usage.seconds += time.perf_counter() - start
        usage.turns += 1
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens

        # Back to the agent's convention.
        text_parts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id or f"call_{uuid.uuid4().hex[:12]}",
                        "name": block.name,
                        "arguments": block.input or {},
                    }
                )

        text = "\n".join(text_parts)
        usage.tool_calls += len(tool_calls)

        if "```python" in text:
            for chunk in text.split("```python")[1:]:
                usage.programs.append(chunk.split("```")[0].strip())

        return LLMMessage(content=text, tool_calls=tool_calls, raw=response)

    return call_llm, usage
