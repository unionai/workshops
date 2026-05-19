"""
Deploy the fine-tuned model as an OpenAI-compatible API endpoint via vLLM.

Usage:
    # Deploy the latest trained model
    python serve.py

    # Deploy from a specific pipeline or train task run
    python serve.py --run-name <run-name>

    # Test the endpoint
    curl https://your-app-url/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "finetuned-sql", "messages": [{"role": "user", "content": "### Task: Generate a SQL query..."}]}'

    # Or use the OpenAI Python client
    from openai import OpenAI
    client = OpenAI(base_url="https://your-app-url/v1", api_key="na")
    response = client.chat.completions.create(
        model="finetuned-sql",
        messages=[{"role": "user", "content": "..."}],
    )
"""

import argparse

import flyte
import flyte.app
from flyteplugins.vllm import VLLMAppEnvironment

serving_env = VLLMAppEnvironment(
    name="finetuned-sql-llm",
    model_hf_path="HuggingFaceTB/SmolLM2-135M",  # placeholder, overridden at deploy time
    model_id="finetuned-sql",
    resources=flyte.Resources(cpu="4", memory="16Gi", gpu="T4:1", disk="10Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=300,
    ),
    requires_auth=False,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fine-tuned SQL model")
    parser.add_argument(
        "--run-name",
        help="Run name of a specific pipeline or train task run (from Flyte UI). "
             "Default: latest train task output.",
    )
    args = parser.parse_args()

    flyte.init_from_config()

    # With --run-name: deploy the model from that specific run (pipeline or train task).
    # Without: deploy the model from the latest successful train task run.
    if args.run_name:
        run_output = flyte.app.RunOutput(type="directory", run_name=args.run_name)
    else:
        run_output = flyte.app.RunOutput(type="directory", task_name="llm-finetune-cpu.pipeline")

    app = flyte.serve(
        serving_env.clone_with(
            name=serving_env.name,
            model_hf_path=None,
            model_path=run_output,
        )
    )
    print(f"Deployed: {app.url}")
