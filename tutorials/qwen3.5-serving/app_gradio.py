"""A chat UI for the served model. A plain CPU app; it only talks HTTP to the vLLM app.

    python app_gradio.py                                       # deploy next to the vLLM app
    SERVER_URL=https://<endpoint> python app_gradio.py --local # run Gradio here against any endpoint

The vLLM app's address arrives as a parameter (`AppEndpoint`), so the UI follows the model app
wherever it is deployed.
"""

from __future__ import annotations

import logging
import os

import flyte
import flyte.app

from config import APP_NAME, MODEL_ID, PROPAGATED, SECRETS, api_key

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ui_env = flyte.app.AppEnvironment(
    name=f"{APP_NAME}-chat",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages("gradio>=5", "openai>=1.50", "python-dotenv"),
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    requires_auth=False,
    port=7860,
    env_vars={**PROPAGATED, "GRADIO_ANALYTICS_ENABLED": "0"},
    secrets=SECRETS,
    parameters=[flyte.app.Parameter(name="server_url", value=flyte.app.AppEndpoint(app_name=APP_NAME), env_var="SERVER_URL")],
)


def build_demo(server_url: str):
    import gradio as gr
    from openai import OpenAI

    client = OpenAI(base_url=f"{server_url.rstrip('/')}/v1", api_key=api_key())

    def respond(message, history, system_prompt, max_tokens, thinking):
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
            {"role": m["role"], "content": m["content"]} for m in history if m["role"] in ("user", "assistant")
        ] + [{"role": "user", "content": message}]
        stream = client.chat.completions.create(
            model=MODEL_ID, messages=messages, max_tokens=int(max_tokens), stream=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(thinking)}},
        )
        text, reasoning = "", ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if r:
                reasoning += r
            if delta.content:
                text += delta.content
            shown = text
            if reasoning:
                shown = f"<details><summary>thinking ({len(reasoning)} chars)</summary>\n\n{reasoning}\n\n</details>\n\n{text}"
            yield shown

    with gr.Blocks(title=MODEL_ID) as demo:
        gr.Markdown(f"## {MODEL_ID}\nServed by vLLM on Flyte from `{server_url}`")
        with gr.Row():
            system_prompt = gr.Textbox("You are a concise, helpful assistant.", label="System prompt", scale=3)
            max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max tokens", scale=1)
            thinking = gr.Checkbox(False, label="Thinking", scale=1)
        gr.ChatInterface(respond, additional_inputs=[system_prompt, max_tokens, thinking])
    return demo


@ui_env.server
def launch(server_url: str):
    log.info(f"vLLM app at {server_url}")
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
