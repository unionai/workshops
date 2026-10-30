"""The knobs. Change MODEL and DATASET for a new model or task; everything else follows.

    MODEL     the Hugging Face repo to fine-tune
    GPU       where the GPU tasks run: L40s (default) or A100 on a cluster, apple on an Apple Silicon devbox
    JUDGE     optional: an Anthropic model to judge replies pairwise (needs the ANTHROPIC_API_KEY secret)
    ARTIFACTS publish the trained weights to Union's artifact registry when there is one (default on; harmless on OSS Flyte)

Three methods share one pipeline (`--method full | lora | qlora`); the README says which fits
which card. Runs on the Union demo cluster, a CUDA devbox (`flyte start devbox --gpu`) or an
Apple Silicon devbox (`flyte start devbox --apple-silicon`, then `GPU=apple`).
"""

from __future__ import annotations

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")
MODEL_ID = os.getenv("MODEL_ID", "qwen3.5-4b")
DATASET = os.getenv("DATASET", "bitext/Bitext-customer-support-llm-chatbot-training-dataset")
GPU = os.getenv("GPU", "L40s")
JUDGE = os.getenv("JUDGE", "")  # e.g. claude-haiku-5-5; empty = no judge, the recomputable scores only
JUDGE_SECRET = os.getenv("JUDGE_SECRET", "ANTHROPIC_API_KEY")

# Union's artifact registry (model cards, versions, lineage) is used when it is there and skipped
# when it is not, so everything runs on open-source Flyte too. ARTIFACTS=0 forces it off.
ARTIFACTS = os.getenv("ARTIFACTS", "1") not in ("0", "false", "no")

SHAPES = {
    "L40s": flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="100Gi", shm="8Gi"),
    "A100": flyte.Resources(cpu=8, memory="64Gi", gpu="A100:1", disk="100Gi", shm="8Gi"),
    "apple": flyte.Resources(cpu=4, memory="24Gi", gpu="AppleSilicon"),  # the Mac's own GPU, via the devbox
}
if GPU not in SHAPES:
    raise ValueError(f"GPU={GPU!r}; choose one of {list(SHAPES)}")

# Pinned to a range verified on the cluster; bump on purpose and re-run. The last two are CUDA-only
# (fused Gated DeltaNet kernels, 4-bit quantization) and are left out of an Apple Silicon image.
TRAIN_PIP = (
    "torch>=2.6",
    "transformers>=5.16.1,<5.23",
    "peft>=0.20.0,<0.22",
    "trl>=1.12.0,<1.18",
    "datasets>=3.0.0",
    "accelerate>=0.34.0",
    "anthropic>=0.40",
    "markdown",
    "python-dotenv",
) + (() if GPU == "apple" else ("flash-linear-attention>=0.5", "bitsandbytes>=0.48.0,<0.51"))

PROPAGATED = {"MODEL": MODEL, "MODEL_ID": MODEL_ID, "DATASET": DATASET, "GPU": GPU, "JUDGE": JUDGE, "JUDGE_SECRET": JUDGE_SECRET, "ARTIFACTS": "1" if ARTIFACTS else "0"}
SECRETS = [flyte.Secret(key=JUDGE_SECRET, as_env_var="ANTHROPIC_API_KEY")] if JUDGE else []

gpu_env = flyte.TaskEnvironment(
    name="qwen35-ft-gpu",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(*TRAIN_PIP),
    resources=SHAPES[GPU],
    env_vars={**PROPAGATED, "HF_HUB_ENABLE_HF_TRANSFER": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    secrets=SECRETS,
)

cpu_env = flyte.TaskEnvironment(
    name="qwen35-ft-cpu",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("datasets>=3.0.0", "markdown", "python-dotenv"),
    # Half a CPU node on purpose: the demo cluster's CPU pool is 4 vCPU / 16 GiB and cpu=4 never schedules.
    resources=flyte.Resources(cpu=2, memory="8Gi", disk="32Gi"),
    env_vars=PROPAGATED,
    depends_on=[gpu_env],
)
