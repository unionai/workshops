"""Flyte environments for serving an LLM on whatever GPU you have, including the one in your Mac.

One knob, `ACCELERATOR`, picks the app's pod shape and the torch device:

    apple   gpu="AppleSilicon". The app runs natively on this Mac and torch uses `mps`.
            Needs a devbox started with `flyte start devbox --apple-silicon`. The default on a Mac.
    T4      a 16 GB CUDA card on a cluster (fp16).
    L40s    a 48 GB CUDA card on a cluster (bf16).
    cpu     no GPU. Works anywhere; slow.

    ACCELERATOR=L40s python serve.py      # the same app on a cluster GPU
"""

from __future__ import annotations

import os
import platform

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-1.7B")
MLX_MODEL_NAME = os.getenv("MLX_MODEL_NAME", "mlx-community/Qwen3-1.7B-4bit")
APP_NAME = os.getenv("APP_NAME", "apple-silicon-llm")


def _default_accelerator() -> str:
    return "apple" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cpu"


ACCELERATOR = os.getenv("ACCELERATOR", _default_accelerator())

SHAPES = {
    # The pod only proxies the process on the Mac: these reserve devbox capacity, not the work.
    "apple": flyte.Resources(cpu=1, memory="2Gi", gpu="AppleSilicon"),
    "T4": flyte.Resources(cpu=4, memory="24Gi", gpu="T4:1", disk="32Gi"),
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="32Gi"),
    "cpu": flyte.Resources(cpu=2, memory="8Gi"),
}
if ACCELERATOR not in SHAPES:
    raise ValueError(f"ACCELERATOR={ACCELERATOR!r}; choose one of {list(SHAPES)}")

PIP = ("torch>=2.4", "transformers>=4.51", "fastapi", "uvicorn", "python-dotenv")

PROPAGATED = {
    "ACCELERATOR": ACCELERATOR,
    "MODEL_NAME": MODEL_NAME,
    "MLX_MODEL_NAME": MLX_MODEL_NAME,
    "APP_NAME": APP_NAME,
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
}


def reach(app_env) -> str:
    """The URL a *task* uses to call an app.

    Flyte hands every app the in-cluster address pattern for its neighbours, but a task on a devbox
    only knows the app's public URL, `http://<app>-<project>-<domain>.localhost:30081`, which a pod
    cannot resolve. Inside a pod, swap it for the devbox's in-cluster service name instead.
    """
    from urllib.parse import urlparse

    endpoint = app_env.endpoint
    host = urlparse(endpoint).hostname or ""
    if host.endswith(".localhost") and os.getenv("KUBERNETES_SERVICE_HOST"):
        return f"http://{host.removesuffix('.localhost')}.flyte.svc.cluster.local"
    return endpoint
