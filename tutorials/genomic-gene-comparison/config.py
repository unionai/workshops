import flyte

base_image = flyte.Image.from_debian_base(
    name="genomic-gene-compare-v1",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "torch>=2.9.0",
    "transformers>=4.49.0",
    "accelerate>=0.34.0",
    "numpy",
)

gpu_env = flyte.TaskEnvironment(
    name="genomic-gene-compare-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu=1),
)

cpu_env = flyte.TaskEnvironment(
    name="genomic-gene-compare-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="8Gi"),
)
