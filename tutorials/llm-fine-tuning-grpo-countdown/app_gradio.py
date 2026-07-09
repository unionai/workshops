"""
Countdown Solver — Gradio frontend.

Enter numbers and a target; the GRPO-trained model (served by serve.py) returns an
expression, and the server verifies whether it actually solves the puzzle.

Usage:
    # Deploy to cluster (after serve.py is deployed)
    python app_gradio.py

    # Local test (point at a running server)
    SERVER_URL=https://your-app-url python app_gradio.py
"""

import logging
import os
import pathlib

import flyte
import flyte.app

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

app_image = flyte.Image.from_debian_base(
    name="countdown-gradio",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "gradio>=5.0.0",
    "httpx",
)

SERVER_APP_NAME = "grpo-countdown-api"

gradio_env = flyte.app.AppEnvironment(
    name="grpo-countdown-ui",
    image=app_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    requires_auth=False,
    port=7860,
    parameters=[
        flyte.app.Parameter(name="server_url", value="", env_var="SERVER_URL"),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
)


# Numbers, Target. Curated against the deployed model: the first five it solves,
# the last two it misses (a human can see the answer exists, e.g. 3*9-7=20 and
# 5*2*2=20), which honestly shows both what it learned and where it still slips.
EXAMPLES = [
    ["2, 3, 4", 24],   # solves: 2 * 3 * 4
    ["1, 2, 3", 6],    # solves: 1 + 2 + 3
    ["6, 3, 2", 36],   # solves: 6 * 3 * 2
    ["9, 9, 2", 20],   # solves: 9 + 9 + 2
    ["5, 2, 3", 10],   # solves: 5 + 2 + 3
    ["3, 7, 9", 20],   # misses: needs 3 * 9 - 7
    ["5, 2, 2", 20],   # misses: needs 5 * 2 * 2
]


@gradio_env.server
def launch_app(server_url: str):
    import gradio as gr
    import httpx

    api_url = server_url.rstrip("/")
    log.info(f"Connecting to Countdown server: {api_url}")

    def solve(numbers_str: str, target: float):
        try:
            numbers = [int(x) for x in numbers_str.replace(" ", "").split(",") if x != ""]
        except ValueError:
            return "", "⚠️ Enter numbers as comma-separated integers, e.g. 3, 7, 9"
        if not numbers:
            return "", "⚠️ Enter at least one number."

        try:
            resp = httpx.post(
                f"{api_url}/solve",
                json={"numbers": numbers, "target": int(target)},
                timeout=60.0,
            )
            resp.raise_for_status()
            r = resp.json()
        except httpx.ConnectError:
            return "", "Could not connect to the model server. Is it running?"
        except Exception as e:
            return "", f"Error: {e}"

        expr = r.get("expression") or "(no expression)"
        if r.get("correct"):
            status = f"✅ Correct!  {expr} = {int(target)}"
        else:
            val = r.get("value")
            val_str = f" = {val:g}" if val is not None else ""
            status = f"❌ Not quite — the model answered `{expr}`{val_str} (target {int(target)})"
        return expr, status

    with gr.Blocks(title="GRPO Countdown Solver") as demo:
        gr.Markdown(
            "# 🔢 GRPO Countdown Solver\n"
            "Give it some numbers and a target. The model writes an arithmetic "
            "expression that uses **each number once** and equals the target — and "
            "the server **verifies** the answer.\n\n"
            "Powered by a **Qwen2.5-0.5B-Instruct** model trained with **GRPO** on "
            "the Countdown game — it went from ~10% to ~43% solved on held-out puzzles, "
            "learning to solve purely from a correctness reward (no solved examples shown)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                numbers_input = gr.Textbox(
                    label="Numbers (comma-separated)",
                    placeholder="3, 7, 9",
                    value="3, 7, 9",
                )
                target_input = gr.Number(label="Target", value=20, precision=0)
                solve_btn = gr.Button("Solve", variant="primary", size="lg")
            with gr.Column(scale=1):
                expr_output = gr.Textbox(label="Model's expression", lines=1, interactive=False)
                status_output = gr.Markdown(label="Result")

        gr.Examples(
            examples=EXAMPLES,
            inputs=[numbers_input, target_input],
            label="Examples — click to load",
        )

        solve_btn.click(solve, inputs=[numbers_input, target_input],
                        outputs=[expr_output, status_output])
        target_input.submit(solve, inputs=[numbers_input, target_input],
                            outputs=[expr_output, status_output])

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.environ.get("GRADIO_SHARE", "") == "1",
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    server_url = os.environ.get("SERVER_URL", "")

    if server_url:
        launch_app(server_url)
    else:
        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

        from flyte.remote._app import App

        try:
            server_app = App.get(SERVER_APP_NAME)
            server_endpoint = server_app.endpoint
            log.info(f"Found Countdown server at: {server_endpoint}")
        except Exception:
            raise RuntimeError(
                f"Could not find deployed app '{SERVER_APP_NAME}'. "
                "Deploy the model server first with: python serve.py"
            )

        for p in gradio_env.parameters:
            if p.name == "server_url":
                p.value = server_endpoint
                break

        log.info("Deploying Gradio frontend...")
        deployed = flyte.deploy(gradio_env)
        log.info(f"Gradio app deployed: {deployed}")
