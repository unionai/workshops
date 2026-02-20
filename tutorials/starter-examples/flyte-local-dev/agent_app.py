"""Gradio UI for the research agent — kicks off the agent as a Flyte task.

Local:  python agent_app.py
Remote: flyte deploy agent_app.py serving_env
"""

from dotenv import load_dotenv

import flyte
import flyte.app

from research_agent import agent

load_dotenv()

serving_env = flyte.app.AppEnvironment(
    name="research-agent-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "gradio", "langchain-core", "langchain-openai", "langgraph", "ddgs", "python-dotenv",
    ),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    # Agent runs in-process — app needs the key too
    secrets=flyte.Secret(key="SAGE_OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
    requires_auth=False,
    port=7860,
)


def run_query(request: str):
    """Kick off the agent as a Flyte task, stream URL then result."""
    result = flyte.with_runcontext(mode="remote").run(agent, request=request)

    # Show the run URL immediately so you can watch it on the platform
    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url and str(run_url).startswith("http"):
        link_html = f'<a href="{run_url}" target="_blank">View run on Flyte</a>'
        yield "Running on Flyte...", link_html
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
    # Local: load_dotenv() already set OPENAI_API_KEY
    create_demo().launch()