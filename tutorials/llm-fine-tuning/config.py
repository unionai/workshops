import os
from dotenv import load_dotenv
import flyte

load_dotenv()

cpu_env = flyte.TaskEnvironment(
    name="llm-finetune-cpu",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="16Gi"),
)

gpu_env = flyte.TaskEnvironment(
    name="llm-finetune-gpu",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu="L4:1"),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

HF_TOKEN = os.getenv("HF_TOKEN")
