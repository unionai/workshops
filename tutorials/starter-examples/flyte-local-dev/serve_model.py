"""Serve MNIST predictions — loads model saved by the pipeline."""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from torchvision import datasets, transforms
from fastapi import FastAPI

import flyte
from flyte.app import Parameter, RunOutput
from flyte.app.extras import FastAPIAppEnvironment

from ml_pipeline import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH_ENV = "MODEL_PATH"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and dataset on startup."""
    # Remote: MODEL_PATH set by Flyte parameter (see train_and_serve pattern)
    # Local: falls back to model.pt saved by the pipeline
    model_path = Path(os.environ.get(MODEL_PATH_ENV, "model.pt"))
    logger.info(f"Loading model from {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    app.state.model = model
    app.state.device = device

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    app.state.test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
    logger.info("Model and dataset loaded.")

    yield

    logger.info("Shutting down.")


app = FastAPI(title="MNIST Predictor", lifespan=lifespan)

serving_env = FastAPIAppEnvironment(
    name="mnist-predictor",
    app=app,
    description="Predict handwritten digits from MNIST test set",
    parameters=[
        # Remote: resolves model from the latest train task output and sets MODEL_PATH
        Parameter(
            name="model",
            value=RunOutput(task_name="ml_pipeline.pipeline", type="file", getter=(1,)),
            download=True,
            env_var=MODEL_PATH_ENV,
        ),
    ],
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn", "torch", "torchvision",
    ),
    resources=flyte.Resources(cpu=1, memory="4Gi", gpu=1),
    requires_auth=False,
)


@app.get("/predict")
async def predict(index: int = 0) -> dict:
    """Predict digit for a test set image by index."""
    sample, label = app.state.test_dataset[index]
    with torch.no_grad():
        output = app.state.model(sample.unsqueeze(0).to(app.state.device))
        prediction = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][prediction].item()
    return {
        "index": index,
        "prediction": prediction,
        "actual": int(label),
        "confidence": round(confidence, 4),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    # Local: skip RunOutput resolution — lifespan falls back to model.pt
    serving_env.parameters = []

    serve_ctx = flyte.with_servecontext(mode="local")
    local_app = serve_ctx.serve(serving_env)
    local_app.activate(wait=True)
    print(f"App running at {local_app.endpoint}")
    print('Try: curl "http://localhost:8080/predict?index=42"')

    input("Press Enter to shut down...")
    local_app.deactivate(wait=True)

# Local:  python serve_model.py
# Remote: flyte deploy serve_model.py serving_env
