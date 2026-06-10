import flyte

base_image = flyte.Image.from_debian_base(
    name="genomic-gene-compare-v1",
).with_requirements("requirements.txt")

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
