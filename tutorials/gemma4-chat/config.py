"""Shared config for the Gemma 4 chat app on Flyte 2.

Switch between the MoE 26B-A4B and the dense 31B by flipping `MODEL`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import flyte
import flyte.app


@dataclass(frozen=True)
class ModelChoice:
    hf_repo: str
    model_id: str          # the name vLLM exposes over its OpenAI API
    app_name: str          # Flyte app name (DNS-safe, lowercase)
    gpu: int | str         # flyte.Resources gpu spec — int=any-type count, str="H100:1" for typed
    max_model_len: int


GEMMA_4_26B_A4B = ModelChoice(
    # `-it` (instruction-tuned). The base (non-it) model doesn't follow chat
    # format or activate thinking mode — it just continues text. The -it
    # repo also ships a chat_template.jinja so vLLM's /v1/chat/completions
    # works without us manually formatting prompts.
    hf_repo="google/gemma-4-26B-A4B-it",
    model_id="gemma-4-26b-a4b-it",
    app_name="gemma4-26b-a4b-it-vllm",
    # `gpu=1` requests any GPU type. On the local devbox the node label is
    # GB10 (DGX Spark), not H100, so a typed `"H100:1"` would silently match
    # nothing and the pod would schedule with zero GPU resources.
    gpu=1,
    max_model_len=8192,
)

GEMMA_4_31B = ModelChoice(
    hf_repo="google/gemma-4-31B-it",
    model_id="gemma-4-31b-it",
    app_name="gemma4-31b-it-vllm",
    gpu=1,                 # change to 2 for TP=2 on a multi-GPU box
    max_model_len=8192,
)

# Pick which model to deploy. Override via env var to switch without code edits.
MODEL = GEMMA_4_31B if os.environ.get("GEMMA_VARIANT") == "31b" else GEMMA_4_26B_A4B

# Name of the Gradio frontend app.
CHAT_APP_NAME = "gemma4-chat-ui"
