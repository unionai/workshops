import os
from dotenv import load_dotenv
import flyte

load_dotenv()

# No sandbox here (unlike the code example) — the math reward just parses the
# model's final answer and compares it to the gold answer, so there's no
# untrusted code to execute. That makes this pipeline simpler and a bit faster.
gpu_env = flyte.TaskEnvironment(
    name="grpo-math-gpu",
    image=flyte.Image.from_debian_base(
        name="grpo-math",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=4, memory="32Gi", gpu="T4:1"),
    secrets=[
        flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN"),
    ],
)

cpu_env = flyte.TaskEnvironment(
    name="grpo-math-cpu",
    image=flyte.Image.from_debian_base(
        name="grpo-math",
    ).with_requirements("requirements.txt"),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[gpu_env],
)

HF_TOKEN = os.getenv("HF_TOKEN")
