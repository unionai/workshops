"""
Emotion Classifier — Gradio frontend.

Type text and see emotion predictions with confidence scores and
an attention heatmap showing which words the model focuses on.

Usage:
    # Deploy to cluster (auto-discovers the serve.py endpoint)
    python app_gradio.py

    # Local test (requires serve.py running)
    SERVER_URL=https://your-app-url python app_gradio.py
"""

import logging
import os
import pathlib

import flyte
import flyte.app

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Image & environment
# ------------------------------------------------------------------

app_image = flyte.Image.from_debian_base(
    name="emotion-classifier-gradio",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "gradio>=5.0.0",
    "httpx",
)

SERVER_APP_NAME = "emotion-classifier-api"

gradio_env = flyte.app.AppEnvironment(
    name="emotion-classifier-ui",
    image=app_image,
    resources=flyte.Resources(cpu=1, memory="2Gi"),
    requires_auth=False,
    port=7860,
    parameters=[
        flyte.app.Parameter(
            name="server_url",
            value="",
            env_var="SERVER_URL",
        ),
    ],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,
    ),
)


# ------------------------------------------------------------------
# Example texts
# ------------------------------------------------------------------

EXAMPLES = [
    "I am so happy right now, everything is going great!",
    "I feel really sad and lonely tonight",
    "I can't believe they surprised me with a puppy!",
    "This makes me so angry, I can't stand it",
    "I'm terrified of what might happen next",
    "I love you more than anything in the world",
    "I feel so proud of what we accomplished together",
    "The news was shocking, I never expected that",
    "I'm worried about the upcoming exam results",
    "She gave me the sweetest compliment and I'm blushing",
]

EMOTION_COLORS = {
    "sadness": "#4a6fa5",
    "joy": "#f4a261",
    "love": "#e76f51",
    "anger": "#e63946",
    "fear": "#6c567b",
    "surprise": "#2a9d8f",
}

EMOTION_EMOJI = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ------------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------------

@gradio_env.server
def launch_app(server_url: str):
    import gradio as gr
    import httpx

    api_url = server_url.rstrip("/")
    log.info(f"Connecting to emotion classifier: {api_url}")

    def predict(text: str):
        if not text.strip():
            return "", "", ""

        try:
            response = httpx.post(
                f"{api_url}/predict",
                json={"text": text},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return "Could not connect to the model server. Is it running?", "", ""
        except Exception as e:
            return f"Error: {e}", "", ""

        # Build prediction summary
        emotion = result["predicted_emotion"]
        confidence = result["confidence"]
        emoji = EMOTION_EMOJI.get(emotion, "")
        color = EMOTION_COLORS.get(emotion, "#333")

        summary = f"## {emoji} {emotion.title()} ({confidence:.1%})\n\n"

        # Confidence bars for all emotions
        summary += "### Confidence Scores\n\n"
        for score in sorted(result["scores"], key=lambda s: s["score"], reverse=True):
            pct = score["score"] * 100
            bar_emoji = EMOTION_EMOJI.get(score["label"], "")
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            summary += f"`{bar}` **{score['label']}** {bar_emoji} {pct:.1f}%\n\n"

        # Attention heatmap as HTML
        tokens = result.get("tokens", [])
        weights = result.get("attention_weights", [])

        attention_html = ""
        if tokens and weights:
            attention_html = (
                '<div style="font-family:system-ui;padding:16px;">'
                '<h3 style="color:#16213e;margin-bottom:8px;">Attention Heatmap</h3>'
                '<p style="color:#6c757d;font-size:0.85em;margin-bottom:12px;">'
                'Which words the model focuses on for its prediction. Darker = more attention.</p>'
                '<div style="line-height:2.4;font-size:1.1em;">'
            )
            for tok, w in zip(tokens, weights):
                # Gradient: light blue (#dbe9f7) → mid blue (#5a9bd5) → deep navy (#0f3460)
                if w < 0.5:
                    # Low attention: light background, dark text
                    t = w / 0.5
                    r = int(219 + (90 - 219) * t)
                    g = int(233 + (155 - 233) * t)
                    b = int(247 + (213 - 247) * t)
                    text_color = "#1a1a2e"
                else:
                    # High attention: dark background, white text
                    t = (w - 0.5) / 0.5
                    r = int(90 + (15 - 90) * t)
                    g = int(155 + (52 - 155) * t)
                    b = int(213 + (96 - 213) * t)
                    text_color = "#ffffff"
                attention_html += (
                    f'<span style="background:rgb({r},{g},{b});'
                    f'color:{text_color};padding:4px 6px;border-radius:4px;'
                    f'margin:2px;display:inline-block;">{tok}</span>'
                )
            attention_html += '</div></div>'

        # Top attended words
        top_words_md = ""
        if tokens and weights:
            pairs = sorted(zip(weights, tokens), reverse=True)
            top_words_md = "### Most Attended Words\n\n"
            for w, tok in pairs[:5]:
                bar_len = int(w * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                top_words_md += f"`{bar}` **{tok}** ({w:.2f})\n\n"

        return summary + top_words_md, attention_html, ""

    # Build the Gradio UI
    with gr.Blocks(
        title="Emotion Classifier",
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .emotion-card { border-radius: 12px; padding: 20px; }
        """,
    ) as demo:
        gr.Markdown(
            "# Emotion Classifier\n"
            "Enter text and see what emotion the model detects, "
            "with confidence scores and an attention heatmap showing "
            "which words influenced the prediction.\n\n"
            "Powered by a fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) "
            "model trained on the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset "
            "(sadness, joy, love, anger, fear, surprise)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Enter text to classify",
                    placeholder="Type something emotional...",
                    lines=3,
                )
                classify_btn = gr.Button(
                    "Classify Emotion", variant="primary", size="lg",
                )
                gr.Examples(
                    examples=[[ex] for ex in EXAMPLES],
                    inputs=[text_input],
                    label="Try these examples",
                )

            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction")
                attention_output = gr.HTML(label="Attention")
                error_output = gr.Textbox(label="", visible=False)

        classify_btn.click(
            predict,
            inputs=[text_input],
            outputs=[prediction_output, attention_output, error_output],
        )

        text_input.submit(
            predict,
            inputs=[text_input],
            outputs=[prediction_output, attention_output, error_output],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=os.environ.get("GRADIO_SHARE", "") == "1",
        theme=gr.themes.Soft(),
    )


# ------------------------------------------------------------------
# Deploy / local dev
# ------------------------------------------------------------------

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
            log.info(f"Found emotion classifier at: {server_endpoint}")
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
