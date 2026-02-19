import torch
from torchvision import datasets, transforms
from fastapi import FastAPI
import httpx

import flyte
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

from cached_ml_pipeline import create_model


app = FastAPI(title="MNIST Predictor")

app_env = FastAPIAppEnvironment(
    name="mnist-predictor",
    app=app,
    description="Predict handwritten digits from MNIST test set",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn", "torch", "torchvision", "httpx"
    ),
    resources=flyte.Resources(cpu=1, memory="1Gi"),
    requires_auth=False,
)


@app.get("/predict")
async def predict(index: int = 0) -> dict:
    """Predict digit for a test set image by index."""
    sample, label = app.state.test_dataset[index]
    with torch.no_grad():
        output = app.state.model(sample.unsqueeze(0))
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
    # Load model saved by the pipeline
    model = create_model()
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()
    app.state.model = model

    # Load test dataset (downloaded by the pipeline)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    app.state.test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    # Serve locally
    flyte.init()
    local_app = flyte.with_servecontext(mode="local").serve(app_env)
    local_app.activate(wait=True)
    print(f"App running at {local_app.endpoint}")

    # Test it
    response = httpx.get(f"{local_app.endpoint}/predict", params={"index": 42})
    print(f"Prediction: {response.json()}")

    input("Press Enter to shut down...")
    local_app.deactivate(wait=True)