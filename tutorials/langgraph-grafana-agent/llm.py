"""The model factory: one string picks which model the agent thinks with.

    anthropic:claude-opus-5                 Claude, via the Anthropic API (the default)
    anthropic:claude-haiku-4-5              a cheaper Claude, for the bake-off in step 6
    openai:gpt-4.1                          OpenAI, if you have a key
    vllm:qwen3-8b                           a model you serve yourself on a Union app
    vllm:qwen3-8b@https://host/v1           same, with the endpoint spelled out

Everything else in the tutorial takes a LangChain chat model and does not care which.
"""

from __future__ import annotations

import os

DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "anthropic:claude-opus-5")


def chat_model(spec: str | None = None):
    spec = spec or DEFAULT_MODEL
    provider, _, rest = spec.partition(":")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=rest, max_tokens=4096)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=rest)

    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        model_id, _, base_url = rest.partition("@")
        base_url = base_url or os.environ.get("VLLM_BASE_URL", "")
        if not base_url:
            raise ValueError(
                f"{spec!r} needs an endpoint: either vllm:<model>@<url>/v1 or VLLM_BASE_URL in the environment. "
                "Deploy one with `python serve_model.py`."
            )
        return ChatOpenAI(
            model=model_id,
            base_url=base_url,
            api_key=os.environ.get("VLLM_API_KEY", "not-needed"),
            temperature=0,
            max_tokens=2048,
            # Qwen3 thinks by default. An agent that thinks for 40 seconds before each
            # tool call is not what we want here; tool calling works fine without it.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    raise ValueError(f"unknown model spec {spec!r}; use anthropic:<model>, openai:<model> or vllm:<model>[@url]")


# List prices per million tokens (input, output), for the cost tiles. Approximate; edit freely.
PRICES_PER_M = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
}


def cost_usd(spec: str | None, input_tokens: int, output_tokens: int) -> float | None:
    """What those tokens cost at list price, or None for a model we do not have a price for."""
    name = short_name(spec)
    if name not in PRICES_PER_M:
        return None
    pin, pout = PRICES_PER_M[name]
    return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout


def short_name(spec: str | None = None) -> str:
    """`anthropic:claude-opus-5` -> `claude-opus-5`, for labels."""
    spec = spec or DEFAULT_MODEL
    return spec.split(":", 1)[-1].split("@", 1)[0]
