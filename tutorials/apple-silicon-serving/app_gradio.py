"""A chat UI for the model served by serve.py. Plain CPU app; it only talks HTTP to the LLM app.

    python app_gradio.py                                  # deploy next to the LLM app
    SERVER_URL=http://127.0.0.1:8000 python app_gradio.py --local   # against a bare `uvicorn openai_server:app`

The LLM app's address arrives as a parameter (`AppEndpoint`), so this app follows the LLM app
wherever it is deployed: the Mac today, a cluster GPU tomorrow.
"""

from __future__ import annotations

import logging
import os

import flyte
import flyte.app

from config import APP_NAME, MODEL_NAME, PROPAGATED

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ui_env = flyte.app.AppEnvironment(
    name=f"{APP_NAME}-chat",
    image=flyte.Image.from_debian_base().with_pip_packages("gradio>=5", "openai", "python-dotenv"),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    requires_auth=False,
    port=7860,
    env_vars={**PROPAGATED, "GRADIO_ANALYTICS_ENABLED": "0"},
    parameters=[flyte.app.Parameter(name="server_url", value=flyte.app.AppEndpoint(app_name=APP_NAME), env_var="SERVER_URL")],
)


def build_demo(server_url: str):
    import gradio as gr
    from openai import OpenAI

    client = OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key="not-used")

    def respond(message, history, system_prompt, max_tokens, thinking):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")
        ] + [{"role": "user", "content": message}]
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, max_tokens=int(max_tokens), stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        )
        text = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                text += chunk.choices[0].delta.content
                yield text

    with gr.Blocks(title="Local LLM") as demo:
        gr.Markdown(f"## {MODEL_NAME}\nServed by Flyte from `{server_url}`")
        with gr.Row():
            system_prompt = gr.Textbox("You are a concise, helpful assistant.", label="System prompt", scale=3)
            max_tokens = gr.Slider(32, 1024, value=256, step=32, label="Max tokens", scale=1)
            thinking = gr.Checkbox(False, label="Let Qwen3 think", scale=1)
        gr.ChatInterface(respond, additional_inputs=[system_prompt, max_tokens, thinking])
    return demo


@ui_env.server
def launch(server_url: str):
    log.info(f"LLM app at {server_url}")
    build_demo(server_url).launch(server_name="0.0.0.0", server_port=ui_env.get_port().port, share=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run Gradio here against SERVER_URL instead of deploying")
    args = parser.parse_args()
    if args.local:
        build_demo(os.environ.get("SERVER_URL", "http://127.0.0.1:8000")).launch()
    else:
        flyte.init_from_config()
        deployed = flyte.serve(ui_env)
        print(f"Console:  {deployed.url}")
        print(f"Chat UI:  {deployed.endpoint}")
