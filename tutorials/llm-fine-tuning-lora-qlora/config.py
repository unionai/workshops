import os
from dotenv import load_dotenv
import flyte

load_dotenv()

gpu_env = flyte.TaskEnvironment(
    name="llm-finetune-gpu",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="24Gi", gpu=1),
    secrets=[
        # flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="llm-finetune-cpu",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "datasets>=3.0.0", "markdown", "python-dotenv",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
