"""Shared settings for running on a GPU market pool: whole 8-GPU AMD MI355X nodes, reached through a queue.

The pool is its own Union cluster pool (its own object store and registry), so everything that should run there,
model prefetch included, names its queue, and a run never mixes it with another pool.
"""

import os
import sys

import flyte

QUEUE = os.getenv("GPU_QUEUE") or None  # the queue that routes to the GPU pool

# Registry the GPU nodes pull images from, and the image pull
# secret the builder pushes with. Leave IMAGE_REGISTRY unset to use the builder's
# default registry when the cluster can already pull from it.
REGISTRY = os.getenv("IMAGE_REGISTRY")
REGISTRY_SECRET = os.getenv("IMAGE_REGISTRY_SECRET")

# The market grants whole nodes, so a GPU pod asks for all eight MI355X GPUs.
GPUS_PER_NODE = 8


def with_registry(image: flyte.Image, name: str) -> flyte.Image:
    """Name the image and push it to IMAGE_REGISTRY when one is set."""
    if REGISTRY:
        return image.clone(
            name=name, registry=REGISTRY, registry_secret=REGISTRY_SECRET, extendable=True
        )
    return image.clone(name=name, extendable=True)


def full_node_pod_template() -> flyte.PodTemplate:
    """Request a whole node's AMD GPUs on the task container."""
    from kubernetes.client import (
        V1Container,
        V1EnvVar,
        V1EnvVarSource,
        V1ObjectFieldSelector,
        V1PodSpec,
        V1ResourceRequirements,
    )

    gpus = {"amd.com/gpu": str(GPUS_PER_NODE)}
    # NODE_NAME tells a task which granted node it landed on, for the report's allocated-compute table.
    node_name = V1EnvVar(name="NODE_NAME", value_from=V1EnvVarSource(field_ref=V1ObjectFieldSelector(field_path="spec.nodeName")))
    return flyte.PodTemplate(
        primary_container_name="primary",
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    resources=V1ResourceRequirements(requests=gpus, limits=gpus),
                    env=[node_name],
                )
            ],
        ),
    )


def prebuild_and_reexec(images: dict[str, flyte.Image]) -> None:
    """Build images without a queue, then rerun this script with their URIs set.

    A run sent to a queue builds its images on that queue's cluster. Building
    first without a queue uses the default builder, which skips images that
    already exist. The script then restarts with each URI in the given
    environment variable, which the task environments read instead of building.
    """
    missing = {var: image for var, image in images.items() if not os.getenv(var)}
    if not missing:
        return
    flyte.init_from_config()
    for var, image in missing.items():
        os.environ[var] = flyte.build(image).uri
        print(f"{var}={os.environ[var]}", flush=True)
    os.execv(sys.executable, [sys.executable, *sys.argv])
