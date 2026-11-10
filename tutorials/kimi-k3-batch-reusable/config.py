"""The knobs. MODEL picks the family; GPU picks the pod shape; the rest is the warm pool and the batch shape.

    MODEL        moonshotai/Kimi-K3 (reasoning_effort low / high / max) or Qwen/Qwen3.5-4B (enable_thinking off / on,
                 the stand-in that fits one L40S)
    MODEL_URI    optional: an object-store directory with the weights (a `flyte.prefetch.hf_model` output). Empty means
                 vLLM pulls MODEL from the Hugging Face Hub inside the worker
    WORKER_CPU / WORKER_MEM / WORKER_DISK   host resources of a whole-node worker; default by model size (see _NODE_SIZING)
    GPU          L40s (default, the Union demo cluster) or MI355Xx8 (one whole 8x MI355X node, on a pool that admits whole nodes only)
    REPLICAS     warm GPU containers in the pool (default 2 for the stand-in, 1 for a whole node)
    IDLE_TTL     seconds the pool stays warm with no work (default 600; hours for Kimi K3, whose load is an hour)
    EFFORT       thinking level (low for Kimi K3, off for the stand-in)
    PRICE_IN / PRICE_OUT   optional $ per million tokens for the cost column

There is no endpoint. The model lives inside a reusable task environment: a pool of long-running GPU containers that
each load it once and keep it in memory between task calls. The batch calls the `generate` task, Flyte routes each call
to a warm replica, and a replica that dies is replaced while its unfinished call is retried on a live one.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import flyte
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "Qwen/Qwen3.5-4B")


def _slug(repo: str) -> str:
    """`moonshotai/Kimi-K3` -> `kimi-k3`; `Qwen/Qwen3.5-4B` -> `qwen3.5-4b`."""
    return repo.rsplit("/", 1)[-1].lower()


MODEL_ID = os.getenv("MODEL_ID", _slug(MODEL))


# ------------------------------------------------------------------
# Model families: everything that differs between Kimi K3 and the stand-in, in one table
# ------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    name: str
    reasoning_parser: str
    tool_parser: str
    # How a client turns thinking up or down. Kimi K3 always thinks and takes `reasoning_effort`
    # (low / high / max); Qwen3.5 has a per-request on/off switch in its chat template.
    efforts: tuple[str, ...]
    default_effort: str
    # Serving defaults: the day-0 vLLM image per accelerator vendor, flags the recipe adds, env vars it sets.
    image_nvidia: str
    image_amd: str
    serve_flags: tuple[str, ...] = ()
    serve_flags_amd: tuple[str, ...] = ()
    serve_env: dict = field(default_factory=dict)
    serve_env_amd: dict = field(default_factory=dict)
    serve_mode: str = "plugin"
    prefetch: bool = True
    min_replicas: int = 0  # a cold start that takes an hour should not happen on the first request

    def extra_body(self, effort: str) -> dict:
        """The per-request field that sets how much the model thinks, in this family's dialect."""
        if effort not in self.efforts:
            raise ValueError(f"effort={effort!r}; {self.name} takes one of {self.efforts}")
        if self.name in ("kimi-k3", "gpt-oss"):  # both take reasoning_effort low / ... in the request
            return {"reasoning_effort": effort}
        return {"chat_template_kwargs": {"enable_thinking": effort == "on"}}


FAMILIES = {
    "kimi-k3": Family(
        name="kimi-k3",
        reasoning_parser="kimi_k3",
        tool_parser="kimi_k3",
        efforts=("low", "high", "max"),
        default_effort="low",
        # vLLM's day-0 images (vllm.ai/blog/2026-07-27-k3; recipes.vllm.ai/moonshotai/Kimi-K3). The NVIDIA one is
        # CUDA 13 and needs an r580+ driver; the AMD recipe says the nightly-rocm100 tag carries fixes the kimi-k3 tag predates.
        image_nvidia="vllm/vllm-openai:kimi-k3",
        image_amd="vllm/vllm-openai-rocm:nightly-rocm100",
        serve_flags=(
            "--trust-remote-code",  # the KimiK3ForConditionalGeneration class ships in the repo
            "--tensor-parallel-size", "8",
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",  # off by default for K3; batch work over shared prefixes depends on it
            "--gpu-memory-utilization", "0.95",
            "--load-format", "fastsafetensors",
        ),
        # The MI355X recipe: CUDA-only MLA prefill backends do not run on ROCm, and it serves text only.
        serve_flags_amd=(
            "--trust-remote-code",
            "--tensor-parallel-size", "8",
            "--kv-cache-dtype", "fp8",
            "--enable-prefix-caching",
            "--gpu-memory-utilization", "0.9",
            "--attention-config", '{"mla_prefill_backend":"ROCM_AITER_FA"}',
            "--additional-config", '{"kda_prefill_backend":"fused"}',
            "--language-model-only",  # every AMD command in the recipe passes this; vision on ROCm is not in the recipe yet
        ),
        serve_env={"VLLM_ENGINE_READY_TIMEOUT_S": "3600", "VLLM_ALLREDUCE_USE_FLASHINFER": "1"},
        serve_env_amd={
            "VLLM_ROCM_USE_AITER": "1",
            "VLLM_ROCM_USE_AITER_MOE_SITUV2": "1",
            "SAFETENSORS_FAST_GPU": "1",
            "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
            "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1200",
            "HSA_NO_SCRATCH_RECLAIM": "1",  # MEC firmware 38; harmless on 177+
        },
        serve_mode="image",
        prefetch=False,
        min_replicas=1,
    ),
    "qwen3.5": Family(
        name="qwen3.5",
        reasoning_parser="qwen3",
        tool_parser="qwen3_coder",
        efforts=("off", "on"),
        default_effort="off",
        image_nvidia="vllm/vllm-openai:v0.26.0",  # only used with SERVE_MODE=image, the Kimi K3 rehearsal
        image_amd="vllm/vllm-openai-rocm:nightly-rocm100",
        serve_flags=(),
        serve_mode="plugin",
        prefetch=True,
        min_replicas=0,
    ),
    # OpenAI's open-weight MoE (117B total, 5.1B active, MXFP4). Same reasoning_effort knob as Kimi K3, so the batch
    # can be rehearsed on it with the same thinking levels; it writes its reasoning and its answer in separate channels.
    "gpt-oss": Family(
        name="gpt-oss",
        reasoning_parser="openai_gptoss",
        tool_parser="openai",
        efforts=("low", "medium", "high"),
        default_effort="low",
        image_nvidia="vllm/vllm-openai:v0.26.0",
        image_amd="vllm/vllm-openai-rocm:nightly-rocm100",
        serve_flags=(),
        serve_mode="image",
        prefetch=True,
        min_replicas=0,
    ),
}


def family_of(model: str) -> Family:
    m = model.lower()
    if "kimi" in m:
        return FAMILIES["kimi-k3"]
    if "gpt-oss" in m:
        return FAMILIES["gpt-oss"]
    if "qwen3" in m:  # Qwen3 and Qwen3.5 share the thinking switch; Qwen3-0.6B is the smallest smoke test
        return FAMILIES["qwen3.5"]
    raise ValueError(f"No family table for {model!r}; add one to FAMILIES in config.py")


FAMILY = family_of(MODEL)
EFFORTS = FAMILY.efforts
DEFAULT_EFFORT = os.getenv("EFFORT", FAMILY.default_effort)

# ------------------------------------------------------------------
# The warm pool
# ------------------------------------------------------------------

GPU = os.getenv("GPU", "L40s")
MODEL_URI = os.getenv("MODEL_URI", "")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
PRICE_IN = float(os.getenv("PRICE_IN", "0") or 0)
PRICE_OUT = float(os.getenv("PRICE_OUT", "0") or 0)

# What vLLM's offline engine needs per family; the same settings the serving tutorial passes to `vllm serve`.
LLM_KWARGS = {
    "kimi-k3": {
        "trust_remote_code": True, "tensor_parallel_size": 8, "kv_cache_dtype": "fp8", "enable_prefix_caching": True,
        "gpu_memory_utilization": 0.9,
    },
    "qwen3.5": {"gpu_memory_utilization": 0.85, "enable_prefix_caching": True},
    "gpt-oss": {"gpu_memory_utilization": 0.9, "enable_prefix_caching": True},  # MXFP4 weights; one MI355X holds them
}[FAMILY.name]


def chat_template_kwargs(effort: str) -> dict:
    """The thinking knob for the offline `LLM.chat` API: the chat template reads it, as the server would."""
    if effort not in EFFORTS:
        raise ValueError(f"effort={effort!r}; {FAMILY.name} takes one of {EFFORTS}")
    return {"reasoning_effort": effort} if FAMILY.name in ("kimi-k3", "gpt-oss") else {"enable_thinking": effort == "on"}


# gpt-oss writes `<|channel|>analysis<|message|>...` then `<|channel|>final<|message|>...`. The worker keeps the special
# tokens for that family and returns only the final channel, so the parser never sees braces from the reasoning.
KEEP_SPECIAL_TOKENS = FAMILY.name == "gpt-oss"


def final_answer(text: str) -> tuple[str, int]:
    """(the answer, characters of reasoning dropped). A no-op for families that do not use channels."""
    if not KEEP_SPECIAL_TOKENS:
        return text, 0
    marker = "<|channel|>final<|message|>"
    if marker not in text:
        return "", len(text)  # ran out of tokens while still reasoning
    head, _, tail = text.rpartition(marker)
    for end in ("<|return|>", "<|end|>", "<|call|>"):
        tail = tail.split(end, 1)[0]
    return tail.strip(), len(head)


def _amd_node_pod() -> flyte.PodTemplate | None:
    """A whole-node GPU pool admits whole nodes only: 8x amd.com/gpu on the container, no selector, no toleration (Kueue adds
    them). `Resources(gpu=flyte.Device(device_class="AMD_GPU", ...))` makes Union add an AWS accelerator affinity these
    nodes never match, so the GPU request lives in the pod template. The kubernetes client is only needed where task
    specs are built (your machine, and the driver pod)."""
    try:
        from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements
    except ImportError:
        return None
    gpus = {"amd.com/gpu": "8"}
    return flyte.PodTemplate(primary_container_name="primary", pod_spec=V1PodSpec(containers=[
        V1Container(name="primary", resources=V1ResourceRequirements(requests=gpus, limits=gpus))]))


# Host CPU, RAM and local disk for a whole-node worker, by model size. The weights live in GPU memory; the host only
# needs room for the downloaded copy and some working space. A whole node is billed the same whatever is requested,
# but a request the granted node cannot satisfy never schedules, so ask for what the model needs, not the maximum.
#   model               weights    cpu  memory   disk
_NODE_SIZING = {
    "kimi":         ("32", "512Gi", "2000Gi"),  # ~1.5 TB of MXFP4 weights
    "gpt-oss-120b": ("16", "128Gi", "200Gi"),   # ~65 GB
    "default":      ("8",  "64Gi",  "100Gi"),   # Qwen3-0.6B / 8B and similar
}
_size = next((v for k, v in _NODE_SIZING.items() if k != "default" and k in MODEL.lower()), _NODE_SIZING["default"])
WORKER_CPU = os.getenv("WORKER_CPU", _size[0])
WORKER_MEM = os.getenv("WORKER_MEM", _size[1])
WORKER_DISK = os.getenv("WORKER_DISK", _size[2])


AMD = GPU.startswith("MI")
SHAPES = {
    "L40s": dict(resources=flyte.Resources(cpu=6, memory="32Gi", gpu="L40s:1", disk="60Gi", shm="auto")),
    "MI355Xx8": dict(resources=flyte.Resources(cpu=WORKER_CPU, memory=WORKER_MEM, disk=WORKER_DISK, shm="auto"), pod_template=_amd_node_pod()),
}
if GPU not in SHAPES:
    raise ValueError(f"GPU={GPU!r}; choose one of {list(SHAPES)}")
REPLICAS = int(os.getenv("REPLICAS", "1" if GPU == "MI355Xx8" else "2"))
IDLE_TTL = int(os.getenv("IDLE_TTL", str(3 * 3600 if FAMILY.name == "kimi-k3" else 600)))
QUEUE = os.getenv("QUEUE", "")  # the queue of the GPU pool, e.g. the one that routes to your MI355X nodes

PROPAGATED = {
    "MODEL": MODEL, "MODEL_ID": MODEL_ID, "MODEL_URI": MODEL_URI, "GPU": GPU, "EFFORT": DEFAULT_EFFORT,
    "MAX_MODEL_LEN": str(MAX_MODEL_LEN), "REPLICAS": str(REPLICAS), "IDLE_TTL": str(IDLE_TTL),
    "WORKER_CPU": WORKER_CPU, "WORKER_MEM": WORKER_MEM, "WORKER_DISK": WORKER_DISK,
    "PRICE_IN": str(PRICE_IN), "PRICE_OUT": str(PRICE_OUT), "HF_HOME": "/tmp/hf",
    "WORKER_IMAGE": os.getenv("WORKER_IMAGE", ""), "DRIVER_IMAGE": os.getenv("DRIVER_IMAGE", ""),
}

# The worker image must be built from Flyte's base. The stock vLLM images set `ENTRYPOINT ["vllm", "serve"]`; a reusable
# worker is started by passing its launcher as the container args, so on those images vLLM reads the launcher as a
# model name and every replica exits with code 2. A `command` in the pod template does not help: Flyte drops it.
if AMD:
    # No Flyte-built ROCm vLLM image yet: this stock image hits the entrypoint problem above (kept for the build path,
    # and for when the GPU pool has a registry). The ROCm image's system Python is externally managed (PEP 668).
    worker_image = (
        flyte.Image.from_base(FAMILY.image_amd)
        .clone(name="batch-llm-vllm-rocm", registry=os.getenv("IMAGE_REGISTRY") or None, extendable=True)
        .with_env_vars({"UV_PYTHON": "/usr/bin/python3", "UV_BREAK_SYSTEM_PACKAGES": "1"})
        .with_pip_packages("flyte>=2.12", "unionai-reuse", "async-lru", "python-dotenv")
    )
else:
    # The vLLM plugin's image (Flyte's debian base + a pinned vLLM and its FlashInfer kernels), the SDK's own pattern for
    # reusable vLLM workers (flyte-sdk examples/ml/batch_inference_streamed.py).
    from flyteplugins.vllm import DEFAULT_VLLM_IMAGE

    worker_image = DEFAULT_VLLM_IMAGE.clone(name="batch-llm-vllm-cuda").with_pip_packages("unionai-reuse", "async-lru", "python-dotenv")

# WORKER_IMAGE: a full URI of the worker image built earlier (`python build.py`). Used as-is, nothing to build. On
# a pool without its own image registry, a GPU env on that pool's queue has its image built by a builder that cannot push;
# building once without the queue and pinning the URI here avoids it.
WORKER_IMAGE = os.getenv("WORKER_IMAGE", "")
if WORKER_IMAGE:
    worker_image = flyte.Image.from_base(WORKER_IMAGE)

worker_env = flyte.TaskEnvironment(
    name=f"batch-llm-worker-{MODEL_ID.replace('.', '-')}",  # one pool per model: a different model is a different warm set
    image=worker_image,
    env_vars=PROPAGATED,
    # concurrency=1: one in-process engine, and each call already hands vLLM a whole chunk as one batch.
    reusable=flyte.ReusePolicy(replicas=REPLICAS, concurrency=1, idle_ttl=IDLE_TTL, scaledown_ttl=IDLE_TTL),  # per replica: a shorter value removes an idle replica before the pool idles
    queue=QUEUE or None,  # the GPU pool can sit on its own queue
    **SHAPES[GPU],
)

# DRIVER_IMAGE: the same idea for the driver. A run is bound to one cluster pool, so on the GPU pool the driver
# runs there too, and its image would go to that pool's builder; build it once without the queue and pin the URI.
DRIVER_IMAGE = os.getenv("DRIVER_IMAGE", "")

driver_env = flyte.TaskEnvironment(
    name="batch-llm-driver",
    image=flyte.Image.from_base(DRIVER_IMAGE) if DRIVER_IMAGE else flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "python-dotenv", "datasets>=3.0.0", "pandas>=2.0", "pyarrow>=15",
        "kubernetes",  # child task specs are built in this pod: the worker's AMD pod template needs the client here too
        "flyteplugins-vllm",  # ...and the worker image definition comes from the vLLM plugin
    ),
    resources=flyte.Resources(cpu=2, memory="6Gi"),
    env_vars=PROPAGATED,
    depends_on=[worker_env],
)


# The shared engine (`engine.py`, the same file as the endpoint tutorials') also has an OpenAI client path; this folder
# only uses its scoring and summaries, so the endpoint helpers are here to satisfy its imports.
def endpoint_for(app_name: str = "") -> str:
    raise RuntimeError("kimi-k3-batch-reusable has no endpoint: the model runs inside the worker pool")


def api_key() -> str:
    return "not-used"


def extra_body(effort: str = DEFAULT_EFFORT) -> dict:
    return FAMILY.extra_body(effort)
