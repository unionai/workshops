import io
import base64

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import flyte
import flyte.report
from flyte.io import File

env = flyte.TaskEnvironment(name="ml_pipeline")


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


@env.task(cache="auto")
async def load_data() -> pd.DataFrame:
    """Load iris dataset — cached after first run."""
    print("Loading iris dataset...")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


@env.task(cache="auto")
async def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test — cached for same input."""
    print("Splitting data...")
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df["target"])
    return train, test


@env.task
async def train(train_df: pd.DataFrame, n_neighbors: int = 3) -> File:
    """Train KNN classifier."""
    print(f"Training KNN with n_neighbors={n_neighbors}...")
    X, y = train_df.drop("target", axis=1), train_df["target"]
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X, y)
    path = "model.joblib"
    joblib.dump(model, path)
    return await File.from_local(path)


@env.task(report=True)
async def evaluate(model_file: File, test_df: pd.DataFrame) -> str:
    """Evaluate model and generate a Flyte report."""
    local_path = await model_file.download()
    model = joblib.load(local_path)

    X, y = test_df.drop("target", axis=1), test_df["target"]
    y_pred = model.predict(X)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
    cm_html = fig_to_html(fig)
    plt.close(fig)

    # Classification report
    report = classification_report(y, y_pred)
    print(report)

    await flyte.report.replace.aio(
        f"<h2>Model Evaluation</h2>"
        f"<h3>Confusion Matrix</h3>{cm_html}"
        f"<h3>Classification Report</h3><pre>{report}</pre>"
    )
    await flyte.report.flush.aio()

    return report


@env.task
async def pipeline(n_neighbors: int = 3) -> str:
    """Full ML pipeline with caching."""
    df = await load_data()
    train_df, test_df = await split_data(df)
    model = await train(train_df, n_neighbors=n_neighbors)
    report = await evaluate(model, test_df)
    return report
