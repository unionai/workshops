"""Gradio UI for the research pipeline — kicks off the agent as a Flyte task.

Development progression:
  1. Local app + local task:   RUN_MODE=local python app.py
  2. Local app + remote task:  python app.py
  3. Deploy tasks + app:
       flyte deploy workflow.py          # registers tasks & builds images
       flyte deploy app.py serving_env   # deploys the UI
"""

import json
import os

from dotenv import load_dotenv
import flyte
import flyte.app
import flyte.remote as remote

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")
FLYTE_UI_URL = os.getenv("FLYTE_UI_URL", "http://localhost:30080")

serving_env = flyte.app.AppEnvironment(
    name="research-pipeline-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.4", "gradio", "langgraph>=1.0.7", "langchain-openai",
        "tavily-python", "markdown", "python-dotenv", "unionai-reuse",
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    secrets=[
        flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    requires_auth=False,
    port=7860,
    include=[
        "requirements.txt",
        "config.py",
        "workflow.py",
        "graph.py",
        "tools/__init__.py",
        "tools/search.py",
    ],
)

# Pre-registered task reference — fetched from control plane at runtime.
# Deploy tasks first: flyte deploy workflow.py
research_pipeline_task = remote.Task.get(
    "research-pipeline-env.research_pipeline",
    project="flytesnacks",
    domain="development",
    auto_version="latest",
)


def run_query(query, num_topics, max_searches, max_iterations):
    """Kick off the research pipeline as a Flyte task, stream URL then result."""
    if RUN_MODE == "local":
        from workflow import research_pipeline
        task = research_pipeline
    else:
        task = research_pipeline_task

    result = flyte.run(
        task,
        query=query,
        num_topics=int(num_topics),
        max_searches=int(max_searches),
        max_iterations=int(max_iterations),
    )

    # Show the run link immediately
    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url:
        url_str = str(run_url)
        # Rewrite internal cluster URL to the external UI URL
        if "flyte-binary-http" in url_str or "flyte:" in url_str:
            # Extract the path portion (e.g., /v2/domain/development/...)
            from urllib.parse import urlparse
            parsed = urlparse(url_str)
            url_str = f"{FLYTE_UI_URL}{parsed.path}"
        if url_str.startswith("http"):
            link_html = f'<a href="{url_str}" target="_blank">View run on Flyte</a>'
            yield "", link_html
        else:
            link_html = f'<code style="font-size:0.85em;color:#666;">Local run: {url_str}</code>'
            yield "", link_html
    else:
        yield "", "Running..."

    # Wait for completion, then show the report
    result.wait()
    output = json.loads(result.outputs()[0])
    report = output["report"]
    score = output.get("score", "N/A")
    iterations = output.get("iterations", "N/A")

    header = f"**Quality:** {score}/10 | **Iterations:** {iterations}\n\n---\n\n"
    yield header + report, link_html


def create_demo():
    """Build the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Research Agent") as demo:
        gr.Markdown("# Research Agent\nAsk a question — the agent searches the web via Tavily and synthesizes a report.")

        with gr.Row():
            query = gr.Textbox(label="Research Question", placeholder="Compare quantum computing approaches: superconducting vs trapped ion", scale=3)
            submit = gr.Button("Research", variant="primary", scale=1)

        with gr.Row():
            num_topics = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Sub-topics")
            max_searches = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Max searches per topic")
            max_iterations = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Max quality iterations")

        run_link = gr.HTML()
        report = gr.Markdown(label="Report")

        inputs = [query, num_topics, max_searches, max_iterations]
        submit.click(fn=run_query, inputs=inputs, outputs=[report, run_link])
        query.submit(fn=run_query, inputs=inputs, outputs=[report, run_link])

        gr.Examples(
            examples=[
                ["Compare quantum computing approaches: superconducting vs trapped ion"],
                ["What are the pros and cons of electric vehicles?"],
                ["How is AI being used in drug discovery?"],
            ],
            inputs=query,
        )

    return demo


@serving_env.server
def app_server():
    """Launch the Gradio app (called by Flyte on remote deployment)."""
    flyte.init_in_cluster(project="flytesnacks", domain="development")
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    if RUN_MODE != "local":
        flyte.init_from_config()

    create_demo().launch()

# Local app + local task:   RUN_MODE=local python app.py
# Local app + remote task:  python app.py
# Deploy tasks + app:
#   flyte deploy workflow.py          # registers tasks & builds images
#   flyte deploy app.py serving_env   # deploys the UI
