"""Gradio UI for the research agent — kicks off the agent as a Flyte task.

Development progression:
  1. Local app + local task:   RUN_MODE=local python agent_app.py
  2. Local app + remote task:  python agent_app.py
  3. Full remote:              flyte deploy agent_app.py serving_env
"""

from dotenv import load_dotenv
import os

import flyte
import flyte.app

from agent_research import agent

load_dotenv()

# Default "remote" — set RUN_MODE=local for fully local development
RUN_MODE = os.getenv("RUN_MODE", "remote")

serving_env = flyte.app.AppEnvironment(
    name="research-agent-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "gradio", "langchain-core", "langchain-openai", "langgraph", "ddgs", "python-dotenv",
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    secrets=flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
    requires_auth=False,
    port=7860,
)


def run_query(request: str):
    """Kick off the agent as a Flyte task, stream URL then result."""
    result = flyte.with_runcontext(mode=RUN_MODE).run(agent, request=request)

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

    # Wait for completion, then show the answer
    result.wait()
    answer = result.outputs()[0]
    yield answer, link_html


def create_demo():
    """Build the Gradio interface (deferred to avoid lock serialization issues)."""
    import gradio as gr

    with gr.Blocks(title="Research Agent") as demo:
        gr.Markdown("# Research Agent\nAsk a question — the agent searches the web and does math to find answers.")

        question = gr.Textbox(label="Research Question", placeholder="What is the population of France and what is 10% of it?")
        submit = gr.Button("Submit", variant="primary")
        answer = gr.Textbox(label="Answer", lines=8)
        run_link = gr.HTML()

        submit.click(fn=run_query, inputs=question, outputs=[answer, run_link])
        question.submit(fn=run_query, inputs=question, outputs=[answer, run_link])

        gr.Examples(
            examples=[
                "What is the population of France and what is 10% of it?",
                "Who won the last Super Bowl and by how many points?",
            ],
            inputs=question,
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

# Local app + local task:   RUN_MODE=local python agent_app.py
# Local app + remote task:  python agent_app.py
# Deploy to cluster:        flyte deploy agent_app.py serving_env