"""
Cluster-specific knobs for the BioNeMo playground.

Everything that depends on what the demo cluster actually has — GPU type, how many
replicas stay warm, which NIM image tags — lives here so it can be tuned on-site
without touching pipeline code.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Union deployment target
# ---------------------------------------------------------------------------

PROJECT = "bionemo-playground"
DOMAIN = "development"

# The org has to be stated explicitly when the API launches runs from inside the
# cluster. `flyte.init_in_cluster` infers project and domain but not org, and the SDK
# has no `current_org()` to fall back on, so omitting it fails at run submission with:
#
#   Upload failed for .../spec.pb (org='', project='bionemo-playground', ...):
#   validation error: project_id.organization: must be at least 1 characters
#
# which surfaces in the UI as a 502 and a button that appears to do nothing.
ORG = os.environ.get("BIONEMO_ORG", "demo")

# ---------------------------------------------------------------------------
# GPUs
#
# The blueprint recommends L40S / A100 / H100. Valid Flyte device strings:
#   "A10G", "A100", "A100 80G", "H100", "H200", "L4", "L40s", "T4", ...
#
# Override per-NIM with env vars when the booth cluster has a different mix,
# e.g. BIONEMO_GPU_DIFFDOCK="A100:1".
# ---------------------------------------------------------------------------

#
# A bare integer means "this many GPUs, any type the cluster can provision", which is
# what you want when you don't know the cluster's accelerator inventory. Pinning a type
# that the cluster has no NodePool for produces an unschedulable pod whose event reads
# "node(s) didn't match Pod's node affinity/selector" — easy to mistake for a capacity
# problem when it's really a mismatch.
#
# Default is a generic "1" for exactly that reason. Pin a type once you know what the
# cluster actually offers.


def _gpu(value: str) -> str | int:
    """Interpret a bare integer as a type-agnostic GPU count, anything else as a device string."""
    return int(value) if value.isdigit() else value


DEFAULT_GPU = os.environ.get("BIONEMO_GPU", "1")

# GenMol is happy on a T4 (verified: 1.6 s for 5 molecules) and is not the bottleneck,
# so it is pinned there deliberately to leave the scarce L40S capacity for DiffDock.
GPU_GENMOL = _gpu(os.environ.get("BIONEMO_GPU_GENMOL", "T4:1"))

# DiffDock MUST have an Ada/Ampere+ GPU. NVIDIA's hardware requirements for this
# blueprint list L40S, A100 and H100; T4 (Turing) is not on that list, and the failure
# mode is vicious: on a T4 the request returns HTTP 200 with status "failed", a
# -1000.0 sentinel confidence and empty poses, in ~0.3 s. Triton even reports the
# inference as successful. Nothing anywhere logs an error. The same request on an L40S
# returns 10 real poses in 2.5 s. Do not "simplify" this back to a generic GPU request.
GPU_DIFFDOCK = _gpu(os.environ.get("BIONEMO_GPU_DIFFDOCK", "L40s:1"))
GPU_OPENFOLD2 = _gpu(os.environ.get("BIONEMO_GPU_OPENFOLD2", DEFAULT_GPU))
GPU_MSA = _gpu(os.environ.get("BIONEMO_GPU_MSA", DEFAULT_GPU))

# ---------------------------------------------------------------------------
# NIM container images (from the blueprint's Helm chart values.yaml)
#
# These are gated images on nvcr.io. The cluster's service account needs an
# image pull secret for nvcr.io, and each NIM needs NGC_API_KEY at runtime to
# fetch model weights. See README.md "Prerequisites".
# ---------------------------------------------------------------------------

# Upstream, as published by NVIDIA. These are what the blueprint's Helm chart uses,
# and they are what scripts/mirror_nims.sh copies from.
UPSTREAM_NIM_IMAGES = {
    "msa": "nvcr.io/nim/colabfold/msa-search:1.0",
    "openfold2": "nvcr.io/nim/openfold/openfold2:1.0",
    "genmol": "nvcr.io/nim/nvidia/genmol:1.0",
    "diffdock": "nvcr.io/nim/mit/diffdock:2.0",
}

# Mirrored into the dataplane's own ECR, which is what we actually deploy.
#
# Not a preference — a requirement. Union Apps run on Knative, whose Revision
# reconciler resolves an image tag to a digest *before any pod exists*. That path sees
# only the Revision's imagePullSecrets, the namespace default ServiceAccount, and the
# cluster's `registries-skipping-tag-resolving` list. Union's apps controller sets no
# imagePullSecrets on a Revision and nvcr.io is not in the skip list, so a gated NVIDIA
# image 401s at digest resolution — before the pod-mutating webhook that Union's
# `--type image_pull` secrets depend on can run. Deploying straight from nvcr.io fails
# with:
#
#   failed to resolve image to digest: HEAD .../manifests/1.0: 401 Unauthorized
#
# The cluster's own ECR is already in the skip list and its node role can pull from it
# natively, so mirroring sidesteps both problems with no dataplane change. Run
# scripts/mirror_nims.sh to populate it.
# e.g. 123456789012.dkr.ecr.us-east-2.amazonaws.com. Unset means no mirror: upstream images.
ECR_HOST = os.environ.get("BIONEMO_ECR_HOST", "")

MIRRORED_NIM_IMAGES = {
    "genmol": f"{ECR_HOST}/bionemo-nim/genmol:1.0",
    "diffdock": f"{ECR_HOST}/bionemo-nim/diffdock:2.0",
}

# Set BIONEMO_USE_UPSTREAM_IMAGES=1 to deploy straight from nvcr.io. Only works on a
# cluster whose dataplane has been configured with nvcr.io credentials.
USE_UPSTREAM = os.environ.get("BIONEMO_USE_UPSTREAM_IMAGES") == "1" or not ECR_HOST

NIM_IMAGES = {
    **UPSTREAM_NIM_IMAGES,
    **({} if USE_UPSTREAM else MIRRORED_NIM_IMAGES),
}

# Every BioNeMo NIM serves HTTP on 8000 inside the container.
NIM_PORT = 8000

# DiffDock also runs a Triton server on this port, internal to the container.
DIFFDOCK_TRITON_PORT = 8003

# Name of the Union secret holding the NGC API key. Create it with:
#   flyte create secret ngc-api-key --value "$NGC_API_KEY"
NGC_SECRET_KEY = "ngc-api-key"

# ---------------------------------------------------------------------------
# Warm-replica policy
#
# The booth-time loop is GenMol -> DiffDock. Those two must never cold-start,
# so they hold a floor of 1 replica. MSA and OpenFold2 are only used by the
# "full pipeline" path and by offline structure precomputation, so they scale
# to zero and cost nothing between runs.
# ---------------------------------------------------------------------------

# Replica floors: the single most consequential setting here, because the floor is what
# holds a GPU.
#
# Both NIMs default to a floor of 1, so each holds its node continuously and a
# visitor's click costs only inference time. Scale-to-zero is normally cheap to recover
# from, but not here: once our pod goes away Karpenter reclaims the empty node
# (consolidateAfter 10m), so the next request pays node provisioning plus a
# multi-gigabyte image pull plus NIM startup — about ten minutes, measured. Fine
# overnight, fatal in front of a visitor.
#
# What that costs: DiffDock holds a dedicated L4 (g6.2xlarge) at roughly $0.98/hour,
# about $23/day. GenMol holds a T4 on a g4dn.metal node shared with other workloads
# that exists regardless, so its floor costs nothing extra.
#
# Set BIONEMO_DIFFDOCK_FLOOR=0 / BIONEMO_GENMOL_FLOOR=0 to let them scale down between
# demo days.
DIFFDOCK_FLOOR = int(os.environ.get("BIONEMO_DIFFDOCK_FLOOR", "1"))
GENMOL_FLOOR = int(os.environ.get("BIONEMO_GENMOL_FLOOR", "1"))

# GenMol is ~2 s per request, so one replica covers roughly 30 runs/minute. A second
# would idle and consume a GPU for nothing.
GENMOL_REPLICAS = (GENMOL_FLOOR, max(1, GENMOL_FLOOR))

# DiffDock dominates the booth loop. One replica has the throughput for a booth; a
# second only helps a sustained queue, and since bringing one up takes ~10 minutes it
# cannot absorb a sudden rush. Treat max=2 as insurance, not as the plan.
DIFFDOCK_REPLICAS = (DIFFDOCK_FLOOR, max(2, DIFFDOCK_FLOOR))

# Kept for the folding NIMs and any other always-on service.
WARM_REPLICAS = (1, 2)
COLD_REPLICAS = (0, 1)

# How long a NIM waits after the last request before scaling down.
#
# One hour. Losing a GPU mid-conference is the worst outcome available: GPU capacity in
# this org is shared, and reclaiming an L40S means waiting for Karpenter to provision a
# node and pull a 13.8 GB image — about ten minutes, which no booth visitor will wait
# through.
#
# Note this is belt-and-braces. What actually holds the GPU is the floor of 1 replica
# below; with min=1 the pod never goes away and the node is never released. The
# scaledown window only governs how long a *burst* replica lingers. Start the apps well
# before the doors open so the floor replica has claimed its node.
SCALEDOWN_AFTER_SECONDS = int(os.environ.get("BIONEMO_SCALEDOWN_SECONDS", "3600"))

# ---------------------------------------------------------------------------
# Pipeline defaults (match the blueprint notebook)
# ---------------------------------------------------------------------------

# GenMol
DEFAULT_NUM_MOLECULES = 10
DEFAULT_TEMPERATURE = 1.0
DEFAULT_NOISE = 0.2
DEFAULT_STEP_SIZE = 4
DEFAULT_SCORING = "QED"

# DiffDock
DEFAULT_NUM_POSES = 10
DEFAULT_TIME_DIVISIONS = 20
DEFAULT_NUM_STEPS = 18

# MSA-Search
MSA_DATABASES = ["Uniref30_2302", "colabfold_envdb_202108", "PDB70_220313"]
MSA_E_VALUE = 0.0001
MSA_ITERATIONS = 1
MSA_SEARCH_TYPE = "alphafold2"

# ---------------------------------------------------------------------------
# Timeouts
#
# MSA search is minutes, not seconds. DiffDock batch docking of ~10 ligands
# lands around 30s on an L40S but gets slower with num_poses.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pod sizing
#
# Two hard-won numbers here, both discovered against the real cluster:
#
# CPU: single-GPU instances (g6e.2xlarge, g6.2xlarge) advertise 7910m allocatable,
# not 8000m — Kubernetes reserves ~90m for system daemons. Requesting cpu=8 made the
# pod unschedulable on every single-GPU node in the cluster, and the scheduler
# reported it as "Insufficient cpu" alongside an affinity message, which reads like
# GPU scarcity but is a 1% CPU overflow. Stay at 6.
#
# DISK: NIMs unpack their model store into /opt/nim/workspace, which is ephemeral
# container storage. Without an explicit request the pod inherits a 1Gi limit and
# gets evicted mid-startup with "Pod ephemeral local storage usage exceeds the total
# limit of containers 1Gi" — which surfaces as exit 137 and looks like an OOM. It is
# not. Ask for room to unpack.
# ---------------------------------------------------------------------------

CPU_PER_NIM = int(os.environ.get("BIONEMO_CPU", "6"))
MEMORY_PER_NIM = os.environ.get("BIONEMO_MEMORY", "24Gi")

DISK_GENMOL = os.environ.get("BIONEMO_DISK_GENMOL", "40Gi")
DISK_DIFFDOCK = os.environ.get("BIONEMO_DISK_DIFFDOCK", "80Gi")
DISK_FOLDING = os.environ.get("BIONEMO_DISK_FOLDING", "80Gi")

TIMEOUT_GENMOL = 120.0
TIMEOUT_DIFFDOCK = 600.0
TIMEOUT_OPENFOLD2 = 1800.0
TIMEOUT_MSA = 1800.0
