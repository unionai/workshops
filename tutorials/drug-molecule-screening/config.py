import flyte

base_image = flyte.Image.from_debian_base(
    name="drug-screening-v1",
).with_apt_packages(
    "libxrender1", "libxext6",
).with_requirements(
    "requirements.txt",
)

cpu_env = flyte.TaskEnvironment(
    name="drug-screening-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="6Gi"),
)
