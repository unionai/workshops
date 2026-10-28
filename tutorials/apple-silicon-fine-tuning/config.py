"""Flyte environments for fine-tuning on whatever GPU you have, including the one in your Mac.

One knob, `ACCELERATOR`, picks the pod shape and the torch device. Nothing else changes:

    apple   gpu="AppleSilicon". The task runs natively on this Mac and PyTorch uses `mps`.
            Needs a devbox started with `flyte start devbox --apple-silicon`. The default on a Mac.
    T4      a 16 GB CUDA card on a cluster. `cuda`, fp16 inference, fp32 training.
    L40s    a 48 GB CUDA card on a cluster. `cuda`, bf16.
    cpu     no GPU. Slow, but it runs anywhere, including `flyte run --local` on a Linux box.
            The default off a Mac.

    ACCELERATOR=T4 flyte run workflow.py pipeline        # the same code on a cluster

Everything read from the environment here is also handed to the tasks as `env_vars`: the
pipeline task launches its children from inside its own container, which has no `.env`, and
a child's `gpu="AppleSilicon"` request is decided there.
"""

from __future__ import annotations

import os
import platform

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")


def _default_accelerator() -> str:
    return "apple" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cpu"


ACCELERATOR = os.getenv("ACCELERATOR", _default_accelerator())

SHAPES = {
    # The pod only proxies the process on the Mac, so its requests reserve devbox capacity,
    # not the work. Size the run by the Mac's memory instead (a 0.5B LoRA run peaks ~8 GB).
    "apple": flyte.Resources(cpu=1, memory="2Gi", gpu="AppleSilicon"),
    # Known-good shapes on the Union demo cluster.
    "T4": flyte.Resources(cpu=4, memory="24Gi", gpu="T4:1", disk="32Gi"),
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="32Gi"),
    "cpu": flyte.Resources(cpu=2, memory="8Gi"),
}
if ACCELERATOR not in SHAPES:
    raise ValueError(f"ACCELERATOR={ACCELERATOR!r}; choose one of {list(SHAPES)}")

# Pinned loosely on purpose: the Mac environment is built from these by `uv` at run time, and
# torch/transformers both ship macOS arm64 wheels.
PIP = (
    "torch>=2.4",
    "transformers>=4.51",
    "peft>=0.15",
    "datasets>=3.0",
    "python-dotenv",
)

PROPAGATED = {"ACCELERATOR": ACCELERATOR, "MODEL_NAME": MODEL_NAME, "HF_HUB_DISABLE_PROGRESS_BARS": "1"}

gpu_env = flyte.TaskEnvironment(
    name="apple-silicon-ft-gpu",
    image=flyte.Image.from_debian_base().with_pip_packages(*PIP),
    resources=SHAPES[ACCELERATOR],
    env_vars=PROPAGATED,
)

cpu_env = flyte.TaskEnvironment(
    name="apple-silicon-ft-cpu",
    image=flyte.Image.from_debian_base().with_pip_packages("datasets>=3.0", "python-dotenv"),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    env_vars=PROPAGATED,
    depends_on=[gpu_env],
)
