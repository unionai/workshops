import flyte

# DGX Spark devbox specifics:
#   - platform=linux/arm64 only — host is aarch64; QEMU-emulated amd64
#     segfaults on `uv venv`.
#   - registry="localhost:30000" — push to the devbox-local registry,
#     not ghcr.io/flyteorg (which would 403).
#   - extra_args="--index-strategy unsafe-best-match" — uv otherwise locks
#     `torchmetrics` to whichever index it appears in first (the cu130
#     index, which only has 1.0.3). The flag tells uv to consider all
#     indexes for the best matching version.
base_image = flyte.Image.from_debian_base(
    name="rtdetr-detection-v1",
    # registry="localhost:30000",
    # platform=("linux/arm64",),
).with_requirements(
    "requirements.txt",
    extra_args="--index-strategy unsafe-best-match",
)

gpu_env = flyte.TaskEnvironment(
    name="rtdetr-detection-gpu",
    image=base_image,
    # Untyped gpu=1 — the GB10 node is labeled GB10, and a typed request
    # like "L4:1" or "H100:1" would silently match nothing.
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
)

cpu_env = flyte.TaskEnvironment(
    name="rtdetr-detection-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="6Gi"),
    depends_on=[gpu_env],
)
