import os
from dotenv import load_dotenv
import flyte

load_dotenv()

# No sandbox needed — the reward safely evaluates arithmetic expressions with an
# AST-based evaluator (numbers and + - * / only), so nothing untrusted runs.
gpu_env = flyte.TaskEnvironment(
    name="grpo-countdown-gpu",
    image=flyte.Image.from_debian_base(
        name="grpo-countdown",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu=1),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="grpo-countdown-cpu",
    image=flyte.Image.from_debian_base(
        name="grpo-countdown",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=2, memory="8Gi"),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),  # for download_model (gated repos)
    ],
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
