import flyte

base_image = flyte.Image.from_debian_base(
    name="protein-analysis-v1",
).with_requirements(
    "requirements.txt",
)

gpu_env = flyte.TaskEnvironment(
    name="protein-analysis-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
)

cpu_env = flyte.TaskEnvironment(
    name="protein-analysis-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="6Gi"),
    depends_on=[gpu_env],
)
