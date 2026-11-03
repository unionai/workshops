"""
The four BioNeMo NIMs, self-hosted as Union apps.

Each NIM is NVIDIA's own prebuilt container from nvcr.io, run as a Flyte
``AppEnvironment``. We set ``command`` rather than ``args`` because we are bypassing
Flyte's Python entrypoint entirely — the NIM images ship their own server and know
nothing about Flyte.

The replica policy is the whole trick behind making this feel instant at a booth:

  * GenMol and DiffDock hold a floor of one replica, so the models stay resident in
    GPU memory and a visitor's click costs only inference time.
  * MSA-Search and OpenFold2 scale to zero. They are used for offline structure
    precomputation and for the on-demand "full pipeline" run, neither of which is
    standing in front of a visitor waiting.

Deploy them with::

    flyte deploy playground/nims.py genmol_env diffdock_env
    flyte deploy playground/nims.py msa_env openfold2_env   # only if folding live
"""

from __future__ import annotations

import os

import flyte
import flyte.app

from playground import config

# The NGC API key every NIM needs in order to pull model weights on first start.
NGC_SECRET = flyte.Secret(key=config.NGC_SECRET_KEY, as_env_var="NGC_API_KEY")


def _nim_env(
    name: str,
    nim: str,
    gpu: str,
    replicas: tuple[int, int],
    cpu: int,
    memory: str,
    disk: str,
    *,
    description: str,
    shm: str | None = "auto",
) -> flyte.app.AppEnvironment:
    """
    Build an AppEnvironment wrapping one BioNeMo NIM container.

    NIM containers serve on port 8000 and expose /v1/health/ready. We keep the
    container's own port rather than remapping to 8080 so that the request paths in
    ``clients.py`` match the blueprint notebook exactly.
    """
    return flyte.app.AppEnvironment(
        name=name,
        description=description,
        image=flyte.Image.from_base(config.NIM_IMAGES[nim]),
        # No command override: the NIM image's own ENTRYPOINT starts its server.
        # Flyte would otherwise prepend its Python entrypoint.
        command=[],
        port=config.NIM_PORT,
        resources=flyte.Resources(cpu=cpu, memory=memory, gpu=gpu, shm=shm, disk=disk),
        secrets=[NGC_SECRET],
        env_vars={
            # NIMs write downloaded weights here. See README for making this durable.
            "NIM_CACHE_PATH": "/opt/nim/.cache",
        },
        scaling=flyte.app.Scaling(
            replicas=replicas,
            scaledown_after=config.SCALEDOWN_AFTER_SECONDS,
        ),
        # Authenticated. These endpoints front a GPU, so leaving them open means anyone
        # who discovers the URL can run inference on our hardware.
        #
        # This works only if pipeline tasks reach the NIMs over the cluster-internal
        # endpoint rather than the public gateway. If a run starts failing with 401s
        # from GenMol or DiffDock, that assumption is wrong and the options are to give
        # the tasks a token or to put the auth back to False on these two.
        requires_auth=True,
        timeouts=flyte.app.Timeouts(request=3600),
    )


# ---------------------------------------------------------------------------
# Booth-time NIMs — always warm
# ---------------------------------------------------------------------------

genmol_env = _nim_env(
    name="nim-genmol",
    nim="genmol",
    gpu=config.GPU_GENMOL,
    replicas=config.GENMOL_REPLICAS,
    cpu=config.CPU_PER_NIM,
    memory=config.MEMORY_PER_NIM,
    disk=config.DISK_GENMOL,
    description="GenMol: masked diffusion model for fragment-based molecular generation",
)

diffdock_env = _nim_env(
    name="nim-diffdock",
    nim="diffdock",
    gpu=config.GPU_DIFFDOCK,
    replicas=config.DIFFDOCK_REPLICAS,
    cpu=config.CPU_PER_NIM,
    memory=config.MEMORY_PER_NIM,
    disk=config.DISK_DIFFDOCK,
    description="DiffDock v2: blind protein-ligand docking with batch support",
)

# ---------------------------------------------------------------------------
# Folding NIMs — scale to zero
#
# MSA-Search is the expensive one: it wants ~1.3 TB of sequence databases on fast
# local NVMe. Read the README before deploying it; for the booth we precompute
# structures instead and leave this undeployed.
# ---------------------------------------------------------------------------

msa_env = _nim_env(
    name="nim-msa-search",
    nim="msa",
    gpu=config.GPU_MSA,
    replicas=config.COLD_REPLICAS,
    cpu=24,
    memory="128Gi",
    disk=config.DISK_FOLDING,
    description="MSA-Search (MMseqs2): GPU-accelerated multiple sequence alignment",
)

openfold2_env = _nim_env(
    name="nim-openfold2",
    nim="openfold2",
    gpu=config.GPU_OPENFOLD2,
    replicas=config.COLD_REPLICAS,
    cpu=config.CPU_PER_NIM,
    memory="64Gi",
    disk=config.DISK_FOLDING,
    description="OpenFold2: transformer protein structure prediction from MSA",
)


# Convenient groupings for deploy scripts.
BOOTH_NIMS = (genmol_env, diffdock_env)
FOLDING_NIMS = (msa_env, openfold2_env)
ALL_NIMS = BOOTH_NIMS + FOLDING_NIMS


def nim_url(env: flyte.app.AppEnvironment) -> str:
    """
    The address a pipeline task should use to reach a NIM.

    Not `env.endpoint`. Inside a task that property returns the *public* gateway URL,
    because the internal-endpoint pattern env var is only set in app pods, never in
    task pods. With `requires_auth=True` the gateway answers an unauthenticated task
    with a 302 to a login page, and the run fails with:

        NIM at https://....apps.your-org.hosted.unionai.cloud not ready after 120s
        (last: HTTP 302)

    Knative also publishes each app on cluster-local DNS, which does not pass through
    the external gateway. Tasks run in the same namespace, so they can use that and
    reach the NIM directly — which is how the NIMs stay authenticated to the outside
    world while the pipeline still works.

    Set BIONEMO_NIM_PUBLIC=1 to force the public endpoint, e.g. when driving the NIMs
    from a laptop against an unauthenticated deployment.
    """
    if os.environ.get("BIONEMO_NIM_PUBLIC") == "1":
        return env.endpoint
    namespace = f"{config.PROJECT}-{config.DOMAIN}"
    return f"http://{env.name}.{namespace}.svc.cluster.local"
