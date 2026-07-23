"""Task environments for distributed GRPO.

Four environments, and the split between them is the whole point of this tutorial:

    cpu_env      orchestration + data prep         (cheap, no GPU)
    verify_env   sandboxed test execution          (REUSABLE pool, CPU)
    learner_env  the gradient step                 (1x L40s)
    rollout_env  vLLM generation                   (REUSABLE pool, L40s)  [Level 2]

The two reusable environments carry a constraint that shapes everything else:

    A reusable TaskEnvironment cannot set `pod_template`.

Flyte raises `ValueError("Cannot set pod_template when environment is reusable.")`
(flyte/_task_environment.py). That matters because the *bubblewrap* sandbox backend
needs CAP_SYS_ADMIN + unconfined AppArmor, and the only way to request those is
`flyte.PodTemplate().allow_nested_sandboxing()` — a pod_template. So a warm verifier
pool cannot run bwrap, and we use the `userns` backend instead. See verify.py for the
security trade-off that implies, and the README section "Why userns and not bubblewrap".
"""

import os

import flyte

# python-dotenv is a *local* convenience for reading .env; on the cluster HF_TOKEN
# arrives as an env var injected from `flyte.Secret`. Keep the import optional:
# every environment imports this module, so anything required at module level here
# would have to ship in every image — including the deliberately lean verifier one.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

HF_TOKEN = os.getenv("HF_TOKEN")

# NOTE: `.with_pip_packages()`, never `.with_requirements()`. The latter stores a
# *relative* path and re-resolves it at runtime, which breaks anywhere the working
# directory isn't yours. requirements.txt stays for the local venv — keep the two
# lists in sync by hand.
TRAIN_PACKAGES = (
    "torch>=2.1.0",
    "transformers>=4.45.0",
    "trl>=0.15.0",
    "peft>=0.13.0",
    "datasets>=3.0.0",
    "accelerate>=0.34.0",
    "huggingface_hub>=0.24.0",
    "unionai-sandbox[flyte]>=0.0.1b15",
)

# The verifier image is deliberately lean: no torch, no vLLM. It gets pulled by up to
# `replicas` pods and every megabyte is cold-start latency paid N times over.
VERIFY_PACKAGES = (
    "unionai-sandbox[flyte]>=0.0.1b15",
    "unionai-reuse",
)

ROLLOUT_PACKAGES = (
    "vllm>=0.6.0",
    "transformers>=4.45.0",
    "huggingface_hub>=0.24.0",
    "unionai-reuse",
)

GPU = os.getenv("GRPO_GPU", "L40s:1")  # lowercase 's' — "L40S:1" is not a valid accelerator

# CPU/RAM for the two GPU environments, overridable per-run.
#
# Kept modest (6 CPU / 48 GiB) on purpose, but NOT because a bigger request fails
# to schedule — in our own testing an 8 CPU / 64 GiB L40s pod scheduled instantly
# when an L40s node was free. A 14B lives in *GPU* memory, so host RAM barely
# matters here; a smaller footprint just fits more node types and leaves headroom
# below the node's allocatable ceiling (kubelet/system-daemon reservations put
# allocatable a little under nominal). The real thing that makes a GPU task wait
# is node *availability* — if the L40s nodepool has no free node and isn't
# autoscaling one, the task sits in WAITING_FOR_RESOURCES regardless of how small
# you make the request. Watch the scheduler message: "N/M nodes are available"
# with no "exceed limits" on the GPU nodepool means capacity, not request size.
GPU_CPU = int(os.getenv("GRPO_GPU_CPU", "6"))
GPU_MEM = os.getenv("GRPO_GPU_MEM", "48Gi")

# One image shared by the orchestrator and the learner. They need the same
# packages, and giving them separate names would build the identical layer stack
# twice — a slow, pointless duplicate on every cold registry.
train_image = flyte.Image.from_debian_base(name="grpo-dist-train").with_pip_packages(*TRAIN_PACKAGES)

# Environments are declared leaf-first, because each one has to name the
# environments its tasks call via `depends_on`. Miss an edge and the run fails at
# dispatch with "Environment '<name>' not found in image cache" — the callee's
# image was never registered for this run. The call graph here is:
#
#   cpu_env (pipeline)  ->  learner_env (train / evaluate / train_distributed)
#   learner_env         ->  verify_env  (verify_shard)      [Levels 1 and 2]
#   learner_env         ->  rollout_env (generate_rollouts) [Level 2]

# ------------------------------------------------------------------
# Verifier pool — the Level 1 fan-out target
# ------------------------------------------------------------------
#
# Capacity is `replicas x concurrency`: up to 20 x 8 = 160 concurrent shards.
# replicas=(2, 20) autoscales — 2 stay warm between rounds so a burst doesn't pay
# cold start, and it scales to 20 under load. A minimum of 2 is recommended by the
# SDK so a parent task occupying one replica can't starve its own children.
#
# concurrency > 1 requires `async` tasks. verify_shard is async.
verify_env = flyte.TaskEnvironment(
    name="grpo-dist-verify",
    image=flyte.Image.from_debian_base(name="grpo-dist-verify").with_pip_packages(*VERIFY_PACKAGES),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    reusable=flyte.ReusePolicy(
        replicas=(2, 20),
        concurrency=8,
        idle_ttl=300,       # keep the whole pool alive between training steps
        scaledown_ttl=120,  # don't drop individual replicas between bursts
    ),
    # NO pod_template here. That is what buys us `reusable` — and costs us bwrap.
)

# ------------------------------------------------------------------
# Rollout pool [Level 2] — vLLM generation workers
# ------------------------------------------------------------------
#
# Reusable so the vLLM engine (tens of seconds to build, plus weight load) is paid
# once per replica instead of once per round. rollout.py caches the engine in a
# module global, which survives across invocations *because* the container is reused.
#
# concurrency=1: vLLM already batches internally and owns the whole GPU. Handing a
# replica two concurrent task calls just makes them fight over the same KV cache.
rollout_env = flyte.TaskEnvironment(
    name="grpo-dist-rollout",
    image=flyte.Image.from_debian_base(name="grpo-dist-rollout").with_pip_packages(*ROLLOUT_PACKAGES),
    resources=flyte.Resources(cpu=GPU_CPU, memory=GPU_MEM, gpu=GPU),
    reusable=flyte.ReusePolicy(
        replicas=(1, 4),
        concurrency=1,
        idle_ttl=600,       # a GPU replica is expensive to rebuild — hold it longer
        scaledown_ttl=300,
    ),
    env_vars={
        # from_debian_base ships the CUDA *runtime* but not the *toolkit*, so there
        # is no nvcc in the image. vLLM's FlashInfer top-k/top-p sampler JIT-compiles
        # a kernel on first use, and with no nvcc that dies at engine init with:
        #   RuntimeError: Could not find nvcc and default cuda_home=... doesn't exist
        # This is NOT GPU-specific — observed on an L40s (Ampere). Forcing vLLM's
        # native PyTorch sampler skips the compile entirely and needs no toolkit.
        # (The alternative is to bake the full CUDA toolkit into the image, which
        # adds gigabytes and minutes to every build for one kernel.)
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    },
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
)

# ------------------------------------------------------------------
# Learner — holds the policy and the optimizer state
# ------------------------------------------------------------------
#
# Not reusable, deliberately: it owns optimizer state and a multi-GB model, and a
# fresh container per round is the correct hygiene. Reuse pays off for short, bursty,
# load-once-run-many work (verification, generation) — not for a long single call.
#
# depends_on names both pools: Level 1 calls verify_shard, Level 2 also calls
# generate_rollouts.
#
# Declaring rollout_env makes every run build the vLLM image, which is multi-GB and
# added ~9 minutes to a measured Level 1 run that never launches a rollout worker.
# `GRPO_LEVEL=1` drops that dependency. Leave it unset (the default) for Level 2 —
# with it set, distributed.py fails at dispatch with
# "Environment 'grpo-dist-rollout' not found in image cache".
LEVEL = os.getenv("GRPO_LEVEL", "2")
_learner_deps = [verify_env] if LEVEL == "1" else [verify_env, rollout_env]

learner_env = flyte.TaskEnvironment(
    name="grpo-dist-learner",
    image=train_image,
    resources=flyte.Resources(cpu=GPU_CPU, memory=GPU_MEM, gpu=GPU),
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
    depends_on=_learner_deps,
)

# ------------------------------------------------------------------
# Orchestration + data prep
# ------------------------------------------------------------------

cpu_env = flyte.TaskEnvironment(
    name="grpo-dist-cpu",
    image=train_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
    depends_on=[learner_env],
)
