"""The model call, in one place.

Every step that talks to a model goes through this file, so switching providers
is an environment variable rather than an edit. Two backends:

    LLM_PROVIDER=anthropic   (default) — the official `anthropic` SDK
    LLM_PROVIDER=openai                — the `openai` SDK, which also speaks to
                                         any OpenAI-compatible server: vLLM,
                                         Ollama, LM Studio, llama.cpp

The second one is how you run this whole tutorial against a local model later.
Point OPENAI_BASE_URL at your server and nothing else changes:

    export LLM_PROVIDER=openai
    export OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama
    export LLM_MODEL=llama3.1
    export OPENAI_API_KEY=unused                       # most local servers ignore it

Two functions, because this tutorial only ever needs two things from a model:
`answer()` for prose a human reads, and `extract()` for JSON a program reads.
"""

from __future__ import annotations

import json
import os

# Claude Opus 5. Override with LLM_MODEL — `claude-haiku-4-5` is a good deal
# cheaper if you are running a room full of people through this.
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "anthropic").lower()


def model_id() -> str:
    if explicit := os.environ.get("LLM_MODEL"):
        return explicit
    return DEFAULT_ANTHROPIC_MODEL if provider() == "anthropic" else DEFAULT_OPENAI_MODEL


def describe() -> str:
    """One line naming the backend, for logs and Flyte reports."""
    if provider() == "anthropic":
        return f"anthropic/{model_id()}"
    base = os.environ.get("OPENAI_BASE_URL", "api.openai.com")
    return f"openai-compatible/{model_id()} @ {base}"


def _require_key() -> None:
    """Fail with an instruction rather than a stack trace 40 lines deep."""
    var = "ANTHROPIC_API_KEY" if provider() == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(var):
        raise RuntimeError(
            f"{var} is not set.\n"
            f"  Locally:    put {var}=... in a .env file next to this one\n"
            f"  On cluster: flyte create secret {var} -p flytesnacks -d development"
        )


# ── Prose ─────────────────────────────────────────────────────────────────────

def answer(
    system: str,
    user: str,
    max_tokens: int = 2000,
    effort: str = "medium",
) -> str:
    """Ask for text and get text back.

    `effort` is Claude's thinking-depth dial: low | medium | high | xhigh | max.
    Grounded question-answering over retrieved chunks is not a hard reasoning
    problem, so medium keeps the workshop moving without hurting the answers.
    It is ignored on the OpenAI path, which has no equivalent knob.
    """
    _require_key()

    if provider() == "anthropic":
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=model_id(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
        # content is a list of blocks (thinking, text, ...) — take the text ones.
        return "\n".join(b.text for b in response.content if b.type == "text").strip()

    from openai import OpenAI

    response = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL")).chat.completions.create(
        model=model_id(),
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# ── JSON ──────────────────────────────────────────────────────────────────────

def extract(system: str, user: str, schema: dict, max_tokens: int = 2000) -> dict:
    """Ask for JSON matching `schema` and get a parsed dict back.

    Both providers can *constrain* the output to the schema rather than merely
    being asked for JSON politely, so there is no regex here fishing a `{...}`
    block out of prose. Step 4 leans on this: if fact extraction returned
    malformed JSON, memory would silently stop being written.
    """
    _require_key()

    if provider() == "anthropic":
        import anthropic

        response = anthropic.Anthropic().messages.create(
            model=model_id(),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            # Extraction is a mechanical task — no need to think hard about it.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    from openai import OpenAI

    response = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL")).chat.completions.create(
        model=model_id(),
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extraction", "schema": schema, "strict": True},
        },
    )
    return json.loads(response.choices[0].message.content or "{}")
