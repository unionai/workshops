"""
RT-DETR Object Detection — Gradio frontend.

Upload images or use your webcam to detect objects with the fine-tuned
RT-DETR model served by app_server.py.

Usage:
    # Deploy to cluster
    python app_gradio.py

    # Local test (requires app_server.py running on port 8080)
    SERVER_URL=http://localhost:8080 python app_gradio.py
"""

import base64
import io
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
    name="rtdetr-detection-gradio-v1",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "gradio>=5.0.0",
    "httpx",
    "pillow>=10.0.0",
)

# The server app name — used to construct the cluster-internal URL.
SERVER_APP_NAME = "rtdetr-detection-server"

gradio_env = flyte.app.AppEnvironment(
    name="rtdetr-detection-ui",
    image=app_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
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
# Gradio app
# ------------------------------------------------------------------

@gradio_env.server
def launch_app(server_url: str):
    import gradio as gr
    import httpx
    from PIL import Image, ImageDraw, ImageFont

    api_url = server_url.rstrip("/")
    log.info(f"Connecting to detection server: {api_url}")

    # -- Colors for different classes --
    COLORS = [
        "#0f3460", "#06d6a0", "#e94560", "#ffc107",
        "#5a7db5", "#ff6b6b", "#4ecdc4", "#45b7d1",
    ]

    def get_color(label_id: int) -> str:
        return COLORS[label_id % len(COLORS)]

    # -- Drawing helper --
    def _get_font(img_width: int):
        """Get a readable font, trying several common paths."""
        font_size = max(16, img_width // 35)
        for font_name in [
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "Arial.ttf",
            "arial.ttf",
        ]:
            try:
                return ImageFont.truetype(font_name, size=font_size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default(size=font_size)

    def draw_detections(image: Image.Image, detections: list) -> Image.Image:
        """Draw bounding boxes and labels on the image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        font = _get_font(img.width)
        line_width = max(3, img.width // 250)

        for det in detections:
            box = det["box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            color = get_color(det["label_id"])
            label = det["label"]
            score = det["score"]

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Draw label background + text above the box
            caption = f"{label} {score:.0%}"
            text_bbox = draw.textbbox((x1, y1), caption, font=font)
            text_h = text_bbox[3] - text_bbox[1]
            # Position label above the box, or inside if at the top edge
            label_y = y1 - text_h - 4 if y1 - text_h - 4 > 0 else y1
            text_bbox = draw.textbbox((x1, label_y), caption, font=font)
            pad = 3
            draw.rectangle(
                [text_bbox[0] - pad, text_bbox[1] - pad,
                 text_bbox[2] + pad, text_bbox[3] + pad],
                fill=color,
            )
            draw.text((x1, label_y), caption, fill="white", font=font)

        return img

    # -- Detection function --
    def detect_image(image_path: str, threshold: float):
        """Send image to the detection server and return annotated image."""
        if image_path is None:
            return None, "No image provided."

        try:
            with open(image_path, "rb") as f:
                files = {"file": ("image.jpg", f, "image/jpeg")}
                response = httpx.post(
                    f"{api_url}/detect",
                    files=files,
                    params={"threshold": threshold},
                    timeout=30.0,
                )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return None, "Could not connect to detection server. Is it running?"
        except Exception as e:
            return None, f"Error: {e}"

        detections = result["detections"]
        image = Image.open(image_path).convert("RGB")
        annotated = draw_detections(image, detections)

        # Build summary text
        if not detections:
            summary = "No objects detected."
        else:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            parts = [f"{count}x {label}" for label, count in counts.items()]
            summary = f"Found {len(detections)} object(s): {', '.join(parts)}"

            # Per-detection details
            summary += "\n\n"
            for i, d in enumerate(detections):
                box = d["box"]
                summary += (
                    f"  {i+1}. {d['label']} ({d['score']:.0%}) "
                    f"[{box['x1']:.0f}, {box['y1']:.0f}, "
                    f"{box['x2']:.0f}, {box['y2']:.0f}]\n"
                )

        return annotated, summary

    # -- Webcam detection function --
    def detect_webcam(frame, threshold: float):
        """Send a webcam frame to the detection server."""
        if frame is None:
            return None, "No frame captured."

        try:
            # Convert numpy frame to JPEG bytes
            img = Image.fromarray(frame)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            response = httpx.post(
                f"{api_url}/detect_base64",
                json={"image": img_b64},
                params={"threshold": threshold},
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.ConnectError:
            return None, "Could not connect to detection server."
        except Exception as e:
            return None, f"Error: {e}"

        detections = result["detections"]
        annotated = draw_detections(img, detections)

        if not detections:
            summary = "No objects detected."
        else:
            counts: dict[str, int] = {}
            for d in detections:
                counts[d["label"]] = counts.get(d["label"], 0) + 1
            parts = [f"{count}x {label}" for label, count in counts.items()]
            summary = f"Found {len(detections)} object(s): {', '.join(parts)}"

        return annotated, summary

    # -- Build the Gradio UI --
    with gr.Blocks(title="RT-DETR Object Detection") as demo:
        gr.Markdown(
            "# RT-DETR Object Detection\n"
            "Upload an image or use your webcam to detect objects "
            "with the fine-tuned RT-DETR model."
        )

        with gr.Tab("Upload Image"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        type="filepath",
                        label="Upload Image",
                        height=400,
                    )
                    upload_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    upload_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    upload_output = gr.Image(label="Detections", height=400)
                    upload_details = gr.Textbox(
                        label="Details",
                        lines=6,
                        interactive=False,
                    )

            upload_btn.click(
                detect_image,
                inputs=[upload_input, upload_threshold],
                outputs=[upload_output, upload_details],
            )

        with gr.Tab("Webcam"):
            with gr.Row():
                with gr.Column():
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Webcam",
                        height=400,
                    )
                    webcam_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.95,
                        value=0.5,
                        step=0.05,
                        label="Confidence Threshold",
                    )
                    webcam_btn = gr.Button("Detect Objects", variant="primary")
                with gr.Column():
                    webcam_output = gr.Image(label="Detections", height=400)
                    webcam_details = gr.Textbox(
                        label="Details",
                        lines=4,
                        interactive=False,
                    )

            webcam_btn.click(
                detect_webcam,
                inputs=[webcam_input, webcam_threshold],
                outputs=[webcam_output, webcam_details],
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
        # Local dev — connect to a running server
        launch_app(server_url)
    else:
        # Deploy to cluster — auto-discover the server endpoint
        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

        from flyte.remote._app import App

        try:
            server_app = App.get(SERVER_APP_NAME)
            server_endpoint = server_app.endpoint
            log.info(f"Found detection server at: {server_endpoint}")
        except Exception:
            raise RuntimeError(
                f"Could not find deployed app '{SERVER_APP_NAME}'. "
                "Deploy the model server first with: python app_server.py"
            )

        for p in gradio_env.parameters:
            if p.name == "server_url":
                p.value = server_endpoint
                break

        log.info("Deploying Gradio frontend...")
        deployed = flyte.deploy(gradio_env)
        log.info(f"Gradio app deployed: {deployed}")
