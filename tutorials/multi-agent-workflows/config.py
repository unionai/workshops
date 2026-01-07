import os
from dotenv import load_dotenv
import flyte

load_dotenv()

# ----------------------------------
# Base Task Environment
# ----------------------------------
base_env = flyte.TaskEnvironment(
    name="base_env",
    image=flyte.Image.from_debian_base().with_requirements("requirements.txt"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="SAGE_TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    reusable=flyte.ReusePolicy(
        replicas=2,
        idle_ttl=60,
        concurrency=6,
        scaledown_ttl=60,
    ), # uncomment to enable task reuse
)

# ----------------------------------
# Local API Keys
# ----------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ----------------------------------
# Database configuration
# ----------------------------------

# ----------------------------------
# logging configuration
# ----------------------------------