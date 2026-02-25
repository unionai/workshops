"""Gradio UI for the research agent — kicks off the workflow as a Flyte task.

Development progression:
  1. Local app + local task:   RUN_MODE=local python -m agent_research.app
  2. Local app + remote task:  python -m agent_research.app
  3. Full remote:              flyte deploy agent_research/app.py serving_env
"""

import json
import os

from dotenv import load_dotenv

import flyte
import flyte.app

from workflow import research_workflow

load_dotenv()

# Default "remote" — set RUN_MODE=local for fully local development
RUN_MODE = os.getenv("RUN_MODE", "remote")

serving_env = flyte.app.AppEnvironment(
    name="research-agent-ui",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "gradio", "python-dotenv", "markdown",
        "langgraph>=1.0.7", "langchain-openai", "tavily-python", "unionai-reuse",
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[
        flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="SAGE_TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    requires_auth=False,
    port=7860,
)


def run_query(query: str, num_topics: int, max_searches: int):
    """Kick off the research workflow as a Flyte task, stream URL then result."""
    result = flyte.with_runcontext(mode=RUN_MODE).run(
        research_workflow,
        query=query,
        num_topics=num_topics,
        max_searches=max_searches,
    )

    # Show the run link immediately
    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url:
        url_str = str(run_url)
        if url_str.startswith("http"):
            link_html = f'<a href="{url_str}" target="_blank">View run on Flyte</a>'
            yield "Running on Flyte...", link_html
        else:
            link_html = f'<code style="font-size:0.85em;color:#666;">Local run: {url_str}</code>'
            yield "Running locally...", link_html
    else:
        yield "Running...", ""

    # Wait for completion, then parse and display the report
    result.wait()
    output = result.outputs()[0]
    try:
        data = json.loads(output)
        report = data.get("report", output)
    except (json.JSONDecodeError, TypeError):
        report = str(output)

    yield report, link_html


def create_demo():
    """Build the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Research Agent") as demo:
        gr.Markdown("# Research Agent\nAsk a question — parallel agents search the web and synthesize a report.")

        query = gr.Textbox(label="Research Question", placeholder="Compare quantum computing approaches: superconducting vs trapped ion vs photonic")
        with gr.Row():
            num_topics = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Sub-topics")
            max_searches = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Searches per topic")
        submit = gr.Button("Research", variant="primary")
        report = gr.Markdown(label="Report")
        run_link = gr.HTML()

        submit.click(fn=run_query, inputs=[query, num_topics, max_searches], outputs=[report, run_link])
        query.submit(fn=run_query, inputs=[query, num_topics, max_searches], outputs=[report, run_link])

        gr.Examples(
            examples=[
                ["Compare vector databases: Pinecone vs Weaviate vs Chroma", 3, 2],
                ["What is the current state of AI agents in production?", 3, 2],
                ["Pros and cons of moving from REST to GraphQL", 2, 2],
                ["Research the best coffee brewing methods and the science behind them", 3, 1],
            ],
            inputs=[query, num_topics, max_searches],
        )

    return demo


@serving_env.server
def app_server():
    """Launch the Gradio app (called by Flyte on remote deployment)."""
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    # Connect to the cluster for remote task execution
    if RUN_MODE == "remote":
        flyte.init_from_config()

    create_demo().launch()

# Local app + local task:   RUN_MODE=local python -m agent_research.app
# Local app + remote task:  python -m agent_research.app
# Deploy to cluster:        flyte deploy agent_research/app.py serving_env