import os
from pathlib import Path
from dotenv import load_dotenv
import flyte

# Load .env from the same directory as this file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# ----------------------------------
# Base Task Environment
# ----------------------------------
base_env = flyte.TaskEnvironment(
    name="base_env",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=5,
        scaledown_ttl=60,
    ), # uncomment/comment to toggle task reuse
)

# ----------------------------------
# Local API Keys
# ----------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
