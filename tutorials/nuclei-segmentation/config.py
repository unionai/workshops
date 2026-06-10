import flyte

base_image = flyte.Image.from_debian_base(
    name="nuclei-segmentation-v1",
).with_requirements(
    "requirements.txt",
    extra_args="--index-strategy unsafe-best-match",
)

gpu_env = flyte.TaskEnvironment(
    name="nuclei-segmentation-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
)

cpu_env = flyte.TaskEnvironment(
    name="nuclei-segmentation-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="6Gi"),
    depends_on=[gpu_env],
)
