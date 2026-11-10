# /// script
# requires-python = ">=3.12"
# dependencies = [
#    "flyte>=2.13.1",
# ]
# ///

"""Prefetch a Hugging Face model into the GPU pool's object store.

`flyte prefetch hf-model` runs on the default cluster. This script runs the
same prefetch task on the GPU queue, so the weights are stored
next to the GPUs that load them and the model is still recorded as an artifact.

    uv run prefetch.py openai/gpt-oss-120b
"""

import argparse

import flyte
from flyte.prefetch._hf_model import (
    HF_DOWNLOAD_IMAGE_PACKAGES,
    HuggingFaceModelInfo,
    store_hf_model_task,
)

from common import QUEUE, with_registry

image = with_registry(
    flyte.Image.from_debian_base(python_version=(3, 12)), "spot-prefetch"
).with_pip_packages(*HF_DOWNLOAD_IMAGE_PACKAGES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="Hugging Face repo id, e.g. openai/gpt-oss-120b")
    parser.add_argument("--artifact-name", default=None)
    # Files stream one by one into the store, so little local disk is needed.
    parser.add_argument("--disk", default="50Gi")
    args = parser.parse_args()

    flyte.init_from_config()

    # Build without a queue, then run on the GPU queue with the URI.
    built = flyte.build(image)

    env = flyte.TaskEnvironment(
        name="spot-prefetch-hf-model",
        image=built.uri,
        resources=flyte.Resources(cpu="4", memory="16Gi", disk=args.disk),
        queue=QUEUE,
    )
    task = env.task(report=True, produces_artifacts=True, cache="auto")(store_hf_model_task)

    info = HuggingFaceModelInfo(repo=args.repo, artifact_name=args.artifact_name)
    model_label = args.artifact_name or args.repo.rsplit("/", maxsplit=1)[-1].replace(".", "-")
    # The task wraps a function from another module, so it is pickled, as
    # flyte.prefetch.hf_model does, instead of being re-imported by name.
    run = flyte.with_runcontext(
        queue=QUEUE, interactive_mode=True, labels={"model": model_label}
    ).run(task, info.model_dump_json(), None)
    print(run.name, run.url)


if __name__ == "__main__":
    main()
