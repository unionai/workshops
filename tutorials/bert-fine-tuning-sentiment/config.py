import os
from dotenv import load_dotenv
import flyte

load_dotenv()

base_image = flyte.Image.from_debian_base(
    name="bert-sentiment-v2",
).with_requirements("requirements.txt")

gpu_env = flyte.TaskEnvironment(
    name="bert-finetune-gpu",
    image=base_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu="T4:1"),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="bert-finetune-cpu",
    image=base_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
