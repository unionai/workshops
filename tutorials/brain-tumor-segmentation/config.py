import flyte

# DGX Spark devbox specifics:
#   - platform=linux/arm64 only — host is aarch64; QEMU-emulated amd64
#     segfaults on `uv venv`.
#   - registry="localhost:30000" — push to the devbox-local registry,
#     not ghcr.io/flyteorg (which would 403).
#   - extra_args="--index-strategy unsafe-best-match" — uv otherwise locks
#     packages to whichever index appears first.
base_image = flyte.Image.from_debian_base(
    name="brain-tumor-seg-v1",
    # registry="localhost:30000",
    # platform=("linux/arm64",),
).with_requirements(
    "requirements.txt",
    extra_args="--index-strategy unsafe-best-match",
)

gpu_env = flyte.TaskEnvironment(
    name="brain-tumor-seg-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu=1, shm="8Gi"),
)

cpu_env = flyte.TaskEnvironment(
    name="brain-tumor-seg-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    depends_on=[gpu_env],
)
