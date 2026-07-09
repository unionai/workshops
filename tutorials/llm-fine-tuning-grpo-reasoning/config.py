import os
from dotenv import load_dotenv
import flyte

load_dotenv()

# No sandbox needed — the reward safely evaluates arithmetic expressions with an
# AST-based evaluator (numbers and + - * / only), so nothing untrusted runs.
gpu_env = flyte.TaskEnvironment(
    name="grpo-reasoning-gpu",
    image=flyte.Image.from_debian_base(
        name="grpo-reasoning",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu="L4:1"),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="grpo-reasoning-cpu",
    image=flyte.Image.from_debian_base(
        name="grpo-reasoning",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
