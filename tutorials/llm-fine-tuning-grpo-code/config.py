import os
from dotenv import load_dotenv
import flyte

load_dotenv()

gpu_env = flyte.TaskEnvironment(
    name="grpo-code-gpu",
    image=flyte.Image.from_debian_base(
        name="grpo-code",
    ).with_apt_packages("bubblewrap").with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu="T4:1"),
    pod_template=flyte.PodTemplate().allow_nested_sandboxing(),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="grpo-code-cpu",
    image=flyte.Image.from_debian_base(
        name="grpo-code",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
