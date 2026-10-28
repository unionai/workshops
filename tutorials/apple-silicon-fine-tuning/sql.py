"""The text-to-SQL task, the device choice, and the scoring. No Flyte in here."""

from __future__ import annotations

import re

SYSTEM = "You are a SQL assistant. Given a schema and a question, reply with a single SQL query and nothing else."


def messages(example: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Schema:\n{example['context']}\n\nQuestion: {example['question']}"},
    ]


def render_train(tokenizer, example: dict) -> str:
    """The full conversation, answer included, in the model's own chat format."""
    return tokenizer.apply_chat_template(
        messages(example) + [{"role": "assistant", "content": example["answer"]}], tokenize=False
    )


def render_prompt(tokenizer, example: dict) -> str:
    """The conversation up to the assistant's turn: what the model sees at inference."""
    return tokenizer.apply_chat_template(messages(example), tokenize=False, add_generation_prompt=True)


def normalize_sql(text: str) -> str:
    """Lower-case, collapse whitespace, strip fences and a trailing semicolon, first statement only."""
    text = re.sub(r"```(?:sql)?", "", text).strip()
    text = text.split(";")[0]
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_match(prediction: str, gold: str) -> bool:
    return normalize_sql(prediction) == normalize_sql(gold)


# ── Devices ───────────────────────────────────────────────────────────────────


def pick_device(accelerator: str):
    """The torch device for an ACCELERATOR value. Fails loudly rather than quietly training on the CPU."""
    import torch

    if accelerator == "apple":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "ACCELERATOR=apple but torch has no mps device. Run this on an Apple Silicon devbox "
                "(flyte start devbox --apple-silicon), or set ACCELERATOR=cpu."
            )
        return torch.device("mps")
    if accelerator in ("T4", "L40s"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"ACCELERATOR={accelerator} but torch has no cuda device.")
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(device):
    """bf16 where the hardware has it (Apple Silicon, Ampere+), fp32 elsewhere (T4, CPU)."""
    import torch

    if device.type == "mps":
        return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def device_name(device) -> str:
    import platform
    import subprocess

    if device.type == "cuda":
        import torch

        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def peak_memory_gb(device) -> float:
    import torch

    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0
