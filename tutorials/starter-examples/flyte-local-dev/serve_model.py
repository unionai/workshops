from dotenv import load_dotenv
import httpx
import joblib
from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

import flyte

load_dotenv()
import flyte.app
from flyte.app.extras import FastAPIAppEnvironment

app = FastAPI(title="Iris Predictor")

app_env = FastAPIAppEnvironment(
    name="iris-predictor",
    app=app,
    description="Predict iris species from measurements",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "fastapi", "uvicorn", "scikit-learn", "joblib", "httpx"
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
    requires_auth=False,
)

task_env = flyte.TaskEnvironment(
    name="iris-task-env",
    image=flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
        "scikit-learn", "joblib"
    ),
    resources=flyte.Resources(cpu=1, memory="512Mi"),
)


@task_env.task
async def train_model() -> str:
    """Train a KNN classifier on iris data."""
    iris = load_iris()
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(iris.data, iris.target)
    path = "model.joblib"
    joblib.dump(model, path)
    return path


@app.get("/predict")
async def predict(
    sepal_length: float, sepal_width: float, petal_length: float, petal_width: float
) -> dict:
    """Predict iris species."""
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = app.state.model.predict(features)[0]
    species = ["setosa", "versicolor", "virginica"]
    return {"prediction": species[prediction]}


@app.get("/health")
async def health() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    # Train the model locally
    flyte.init()
    run = flyte.run(train_model)
    model_path = run.outputs()[0]
    app.state.model = joblib.load(model_path)

    # Serve locally
    local_app = flyte.with_servecontext(mode="local").serve(app_env)
    local_app.activate(wait=True)
    print(f"App running at {local_app.endpoint}")

    # Test it
    response = httpx.get(
        f"{local_app.endpoint}/predict",
        params={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    print(f"Prediction: {response.json()}")

    input("Press Enter to shut down...")
    local_app.deactivate(wait=True)
    print("Done!")
