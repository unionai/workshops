"""Gradio UI for real-time fraud scoring using Feast online features.

Usage:
    RUN_MODE=local python app.py        # Local app + local tasks
    python app.py                       # Local app + remote tasks
    flyte deploy app.py serving_env     # Deploy to cluster
"""

import json
import os
import pickle
import base64

from dotenv import load_dotenv

import flyte
import flyte.app

from workflow import fraud_detection_pipeline

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

serving_env = flyte.app.AppEnvironment(
    name="fraud-detection-ui",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "gradio", "python-dotenv",
        "feast", "scikit-learn", "xgboost",
        "pandas", "pyarrow", "kagglehub",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    requires_auth=False,
    port=7860,
)


def run_pipeline():
    """Run the full fraud detection pipeline and return results."""
    result = flyte.with_runcontext(mode=RUN_MODE).run(
        fraud_detection_pipeline,
    )

    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url:
        url_str = str(run_url)
        if url_str.startswith("http"):
            link_html = f'<a href="{url_str}" target="_blank">View run on Flyte</a>'
            yield "Running on Flyte...", "", link_html
        else:
            link_html = f'<code style="font-size:0.85em;color:#666;">Local run: {url_str}</code>'
            yield "Running locally...", "", link_html
    else:
        yield "Running...", "", ""

    result.wait()
    output = result.outputs()[0]

    try:
        data = json.loads(output)
        auc = data.get("auc_roc", "N/A")
        scored = data.get("scored", [])

        summary = f"**AUC-ROC:** {auc:.4f}\n\n**Scored {len(scored)} sample transactions**"

        table = "| Txn ID | User | Amount | Actual | Predicted | Fraud Prob |\n"
        table += "|--------|------|--------|--------|-----------|------------|\n"
        for r in scored:
            actual = "Fraud" if r["actual"] else "Legit"
            predicted = "Fraud" if r["predicted"] else "Legit"
            table += f"| {r['transaction_id']} | {r['user_id']} | ${r['amount']:.2f} | {actual} | {predicted} | {r['fraud_probability']:.4f} |\n"

        yield summary, table, link_html

    except (json.JSONDecodeError, TypeError):
        yield str(output), "", link_html


def create_demo():
    """Build the Gradio interface."""
    import gradio as gr

    with gr.Blocks(title="Fraud Detection") as demo:
        gr.Markdown(
            "# Fraud Detection Pipeline\n"
            "Run the full pipeline: fetch features from Feast, train XGBoost, score transactions."
        )

        submit = gr.Button("Run Pipeline", variant="primary")
        summary = gr.Markdown(label="Summary")
        results = gr.Markdown(label="Scored Transactions")
        run_link = gr.HTML()

        submit.click(
            fn=run_pipeline,
            inputs=[],
            outputs=[summary, results, run_link],
        )

    return demo


@serving_env.server
def app_server():
    """Launch the Gradio app (called by Flyte on remote deployment)."""
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    if RUN_MODE == "remote":
        flyte.init_from_config()

    create_demo().launch()