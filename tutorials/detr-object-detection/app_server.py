"""
RT-DETR Object Detection — FastAPI model server.

Serves a fine-tuned RT-DETR model for object detection inference.
Accepts image uploads and returns bounding box predictions.

Usage:
    # Deploy with a training run name (loads model from Flyte output)
    TRAINING_RUN=<run_name> python app_server.py

    # Deploy with a local model path
    MODEL_PATH=/path/to/finetuned_model python app_server.py

    # Local test
    python app_server.py
"""

import base64
import io
import logging
import os
import sys

import fastapi
import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Image & environment
# ------------------------------------------------------------------

app_image = flyte.Image.from_debian_base(
    name="rtdetr-detection-server-v1",
).with_pip_packages(
    "flyte[tui]>=2.0",
    "torch>=2.9.0",
    "torchvision>=0.24.0",
    "transformers>=4.49.0",
    "pillow>=10.0.0",
    "fastapi",
    "uvicorn",
    "python-multipart",
)

fastapi_app = fastapi.FastAPI(title="RT-DETR Object Detection API")

server_env = FastAPIAppEnvironment(
    name="rtdetr-detection-server",
    app=fastapi_app,
    image=app_image,
    resources=flyte.Resources(cpu=4, memory="16Gi", gpu=1),
    requires_auth=False,
    port=8080,
    parameters=[
        flyte.app.Parameter(
            name="model_path",
            value="",
            download=True,
            env_var="MODEL_PATH",
        ),
    ],
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,
    ),
)

# ------------------------------------------------------------------
# Startup — load model into GPU memory
# ------------------------------------------------------------------


@server_env.on_startup
async def on_startup(model_path: flyte.io.Dir):
    """Load the fine-tuned model and processor into memory."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    model_dir = model_path.path if hasattr(model_path, "path") else str(model_path)
    if not model_dir:
        log.warning("No model_path provided — server will return errors until a model is loaded.")
        return

    log.info(f"Loading model from: {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForObjectDetection.from_pretrained(model_dir).to(device)
    model.eval()

    fastapi_app.state.processor = processor
    fastapi_app.state.model = model
    fastapi_app.state.device = device
    fastapi_app.state.id2label = model.config.id2label

    log.info(f"Model loaded on {device} — classes: {model.config.id2label}")


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@fastapi_app.get("/health")
async def health():
    model_loaded = getattr(fastapi_app.state, "model", None) is not None
    return {"status": "ready" if model_loaded else "starting", "model_loaded": model_loaded}


@fastapi_app.get("/classes")
async def get_classes():
    if getattr(fastapi_app.state, "id2label", None) is None:
        raise fastapi.HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": fastapi_app.state.id2label}


@fastapi_app.post("/detect")
async def detect(
    file: fastapi.UploadFile = fastapi.File(...),
    threshold: float = 0.5,
):
    """Run object detection on an uploaded image.

    Returns bounding boxes in xyxy format with labels and scores.
    """
    import torch
    from PIL import Image

    if getattr(fastapi_app.state, "model", None) is None:
        raise fastapi.HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = fastapi_app.state.model
    processor = fastapi_app.state.processor
    device = fastapi_app.state.device
    id2label = fastapi_app.state.id2label

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_size, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].cpu().tolist(),
        results["labels"].cpu().tolist(),
        results["boxes"].cpu().tolist(),
    ):
        detections.append({
            "label": id2label.get(label, str(label)),
            "label_id": label,
            "score": round(score, 4),
            "box": {
                "x1": round(box[0], 1),
                "y1": round(box[1], 1),
                "x2": round(box[2], 1),
                "y2": round(box[3], 1),
            },
        })

    return {
        "detections": detections,
        "image_size": {"width": image.width, "height": image.height},
        "threshold": threshold,
    }


@fastapi_app.post("/detect_base64")
async def detect_base64(
    request: fastapi.Request,
    threshold: float = 0.5,
):
    """Run object detection on a base64-encoded image.

    Expects JSON body: {"image": "<base64 string>"}
    Useful for webcam frames sent from the Gradio frontend.
    """
    import torch
    from PIL import Image

    if getattr(fastapi_app.state, "model", None) is None:
        raise fastapi.HTTPException(status_code=503, detail="Model not loaded")

    body = await request.json()
    image_b64 = body.get("image", "")
    if not image_b64:
        raise fastapi.HTTPException(status_code=400, detail="No image provided")

    # Strip data URL prefix if present
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    contents = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    model = fastapi_app.state.model
    processor = fastapi_app.state.processor
    device = fastapi_app.state.device
    id2label = fastapi_app.state.id2label

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_size = torch.tensor([image.size[::-1]], device=device)
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_size, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].cpu().tolist(),
        results["labels"].cpu().tolist(),
        results["boxes"].cpu().tolist(),
    ):
        detections.append({
            "label": id2label.get(label, str(label)),
            "label_id": label,
            "score": round(score, 4),
            "box": {
                "x1": round(box[0], 1),
                "y1": round(box[1], 1),
                "x2": round(box[2], 1),
                "y2": round(box[3], 1),
            },
        })

    return {
        "detections": detections,
        "image_size": {"width": image.width, "height": image.height},
        "threshold": threshold,
    }


# ------------------------------------------------------------------
# Deploy / local dev
# ------------------------------------------------------------------

def _resolve_training_run() -> str | None:
    """Get an explicit training run name from env var, or None to auto-discover."""
    explicit = os.environ.get("TRAINING_RUN", "")
    if explicit:
        log.info(f"Using training run from TRAINING_RUN env var: {explicit}")
        return explicit
    return None


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "")

    model_path_local = os.environ.get("MODEL_PATH", "")

    if model_path_local:
        # Local dev mode — load model from a local directory
        import asyncio
        import uvicorn

        # Create a simple Dir-like object for local dev
        class _LocalDir:
            def __init__(self, p):
                self.path = p
        asyncio.run(on_startup(_LocalDir(model_path_local)))
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)
    else:
        # Deploy to cluster — resolve training run and deploy
        import pathlib

        flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
        run_name = _resolve_training_run()

        # Set up the RunOutput to point at the model directory.
        # The pipeline returns (flyte.io.Dir, str) — the model dir is output 0.
        if run_name:
            run_output = flyte.app.RunOutput(
                type="directory", run_name=run_name,
            )
            log.info(f"Deploying with model from run: {run_name}")
        else:
            run_output = flyte.app.RunOutput(
                type="directory",
                task_name="rtdetr-detection-cpu.pipeline",
            )
            log.info("Deploying with model from latest pipeline run")

        for p in server_env.parameters:
            if p.name == "model_path":
                p.value = run_output
                break
        deployed = flyte.deploy(server_env)
        log.info(f"Server deployed: {deployed}")
