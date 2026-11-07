"""The Claude client, and an adapter so `flyte.ai.agents.Agent` can use it.

`client()` is the Claude API (ANTHROPIC_API_KEY) by default, or Claude on Vertex AI when
RPA_LLM_PROVIDER=vertex (Google application default credentials, no API key).

`make_call_llm` adapts `flyte.ai.agents.Agent` (which keeps history in OpenAI chat shape) to
the Anthropic Messages API. Each assistant turn's raw content blocks are cached by tool-call
ids and replayed verbatim, so thinking blocks go back to the model unchanged.
"""

from __future__ import annotations

import functools
import json
from typing import Any

from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from flyte.ai.agents import LLMMessage

import config


@functools.cache
def client() -> AsyncAnthropic | AsyncAnthropicVertex:
    if config.LLM_PROVIDER == "vertex":
        if not config.VERTEX_PROJECT:
            raise RuntimeError("RPA_LLM_PROVIDER=vertex needs VERTEX_PROJECT (your GCP project id)")
        return AsyncAnthropicVertex(project_id=config.VERTEX_PROJECT, region=config.VERTEX_REGION)
    return AsyncAnthropic()  # reads ANTHROPIC_API_KEY


class RefusalError(RuntimeError):
    pass


def _tools(openai_tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out = []
    for t in openai_tools or []:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", "")[:4000],
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


def make_call_llm(effort: str = "medium"):
    raw_turns: dict[tuple[str, ...], list[Any]] = {}

    def to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def flush() -> None:
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in messages:
            role = m["role"]
            if role == "tool":
                pending_results.append({
                    "type": "tool_result", "tool_use_id": m["tool_call_id"], "content": str(m.get("content", "")),
                })
                continue
            flush()
            if role == "assistant":
                ids = tuple(c["id"] for c in m.get("tool_calls") or [])
                if ids and ids in raw_turns:
                    out.append({"role": "assistant", "content": raw_turns[ids]})
                elif m.get("tool_calls"):
                    blocks: list[dict[str, Any]] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                    for c in m["tool_calls"]:
                        blocks.append({"type": "tool_use", "id": c["id"], "name": c["function"]["name"],
                                       "input": json.loads(c["function"]["arguments"] or "{}")})
                    out.append({"role": "assistant", "content": blocks})
                elif m.get("content"):
                    out.append({"role": "assistant", "content": m["content"]})
            elif role == "user":
                out.append({"role": "user", "content": m["content"]})
        flush()
        return out

    async def call_llm(model: str, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> LLMMessage:
        async with client().messages.stream(
            model=model,
            max_tokens=32_000,
            system=system,
            messages=to_anthropic(messages),
            tools=_tools(tools),
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        ) as stream:
            resp = await stream.get_final_message()
        if resp.stop_reason == "refusal":
            raise RefusalError(f"model refused: {getattr(resp, 'stop_details', None)}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": b.input} for b in resp.content if b.type == "tool_use"]
        if calls:
            raw_turns[tuple(c["id"] for c in calls)] = [b.model_dump(exclude_none=True) for b in resp.content]
        return LLMMessage(content=text, tool_calls=calls, raw=resp)

    return call_llm
