"""Prefetch a Hugging Face model into the object store of the pool a queue belongs to.

    QUEUE=<your GPU queue> python prefetch.py Qwen/Qwen3-0.6B --config <your cluster config>
    QUEUE=<your GPU queue> PREFETCH_CPU=8 PREFETCH_MEM=32Gi python prefetch.py moonshotai/Kimi-K3 --config <your cluster config>

Why not `flyte.prefetch.hf_model`: it takes no queue, so it always runs on the default pool and stores the weights in
that pool's bucket. A cluster pool is its own data plane (object store, secrets, registry), and a GPU node in another
pool cannot read that bucket. This runs Flyte's own prefetch task (`store_hf_model_task`: streams files from the Hub
straight to object storage, registers a model artifact versioned by the Hub commit) on the queue you name, so the
weights land where that queue's GPUs can read them.

It uses the default public Flyte image and installs the three packages the prefetch needs when the task starts,
because a pool without a working image registry cannot build or pull a custom image. The task is cached on its
inputs: running it again returns the stored directory without downloading.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import flyte
import flyte.io

PACKAGES = ["huggingface-hub>=0.27.0", "hf-transfer>=0.1.8", "markdown>=3.10"]  # flyte.prefetch's own download list


# Module level, so the pod can load the task by name. Settings come from env vars the launcher sets below.
QUEUE = os.getenv("QUEUE") or None
HF_TOKEN_KEY = os.getenv("HF_TOKEN_KEY") or None

env = flyte.TaskEnvironment(
    name="hf-prefetch",
    resources=flyte.Resources(cpu=os.getenv("PREFETCH_CPU", "4"), memory=os.getenv("PREFETCH_MEM", "16Gi"),
                              disk=os.getenv("PREFETCH_DISK", "50Gi")),  # files stream one at a time: disk need not fit the model
    queue=QUEUE,
    secrets=[flyte.Secret(key=HF_TOKEN_KEY, as_env_var="HF_TOKEN")] if HF_TOKEN_KEY else None,
    env_vars={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
)


def _install(packages: list[str]) -> None:
    """Install into a writable folder and put it on sys.path. The Flyte image's Python is a uv venv (/opt/venv)
    with no pip, and the task's user may not be able to write to it."""
    import shutil

    target = "/tmp/prefetch-packages"
    uv = shutil.which("uv")
    cmd = ([uv, "pip", "install", "--python", sys.executable, "--target", target, *packages] if uv
           else [sys.executable, "-m", "pip", "install", "--target", target, *packages])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"install failed ({proc.returncode}): {proc.stderr[-1500:] or proc.stdout[-1500:]}")
    sys.path.insert(0, target)


@env.task(cache="auto", report=True, produces_artifacts=True)
def prefetch_model(info: str, raw_data_path: str = "") -> flyte.io.Dir:
    _install(PACKAGES)
    from flyte.prefetch._hf_model import store_hf_model_task

    return store_hf_model_task(info, raw_data_path or None)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo")
    p.add_argument("--config", default=None)
    p.add_argument("--artifact_name", default=None,
                   help="artifact to publish under; versions are the Hub commit, so a second pool's copy needs its own name")
    a = p.parse_args()  # queue, sizes and the HF token secret come from QUEUE, PREFETCH_CPU/MEM/DISK, HF_TOKEN_KEY

    from flyte.prefetch._hf_model import HuggingFaceModelInfo

    flyte.init_from_config(a.config)
    info = HuggingFaceModelInfo(repo=a.repo, artifact_name=a.artifact_name).model_dump_json()
    run = flyte.with_runcontext(queue=QUEUE, labels={"model": a.repo.rsplit("/", 1)[-1].replace(".", "-")}).run(prefetch_model, info)
    print(run.name, run.url)
