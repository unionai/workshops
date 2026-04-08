"""
Deploy the fine-tuned model as an OpenAI-compatible API endpoint via vLLM.

Usage:
    # Deploy (uses model from the latest training run)
    python serve.py

    # Test the endpoint
    curl https://your-app-url/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "finetuned-sql", "messages": [{"role": "user", "content": "### Task: Generate a SQL query..."}]}'

    # Or use the OpenAI Python client
    from openai import OpenAI
    client = OpenAI(base_url="https://your-app-url/v1", api_key="na")
    response = client.chat.completions.create(
        model="finetuned-sql",
        messages=[{"role": "user", "content": "..."}],
    )
"""

import flyte
import flyte.app
from flyteplugins.vllm import VLLMAppEnvironment

serving_env = VLLMAppEnvironment(
    name="finetuned-sql-llm",
    model_hf_path="HuggingFaceTB/SmolLM2-135M",  # placeholder, overridden below
    model_id="finetuned-sql",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="L4:1", disk="10Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,
    ),
    requires_auth=False,
)


if __name__ == "__main__":
    flyte.init_from_config()

    # Deploy using the fine-tuned model from a training run.
    # Replace <run-name> with the actual run name from your training pipeline,
    # or omit run_name to use the latest run of the train task.
    app = flyte.serve(
        serving_env.clone_with(
            name=serving_env.name,
            model_hf_path=None,
            model_path=flyte.app.RunOutput(type="directory"),
        )
    )
    print(f"Deployed: {app.url}")
