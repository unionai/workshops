"""MNIST training pipeline with W&B experiment tracking via the Flyte W&B plugin.

Layers W&B logging onto the base ML pipeline — same model, same data,
but every metric is tracked in Weights & Biases.

The @wandb_init decorator on the parent task creates a W&B run, and child
tasks automatically share it — all metrics end up in one run.
"""

import json

from dotenv import load_dotenv
load_dotenv()

import flyte
import flyte.report
from flyte.io import File
from flyteplugins.wandb import wandb_init, get_wandb_run

from ml_pipeline import create_model, get_device, fig_to_html, load_data

image = flyte.Image.from_debian_base(python_version=(3, 12)).with_pip_packages(
    "torch", "torchvision", "matplotlib", "flyteplugins-wandb",
)

env = flyte.TaskEnvironment(
    name="wandb_ml_pipeline",
    image=image,
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu=1),
    secrets=flyte.Secret(key="wandb_api_key", as_env_var="WANDB_API_KEY"),
)


@wandb_init
@env.task
async def train(data_dir: str, epochs: int = 5, lr: float = 0.001, batch_size: int = 64) -> tuple[File, str]:
    """Train ResNet18 on MNIST, logging metrics to W&B."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    run = get_wandb_run()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = get_device()
    print(f"Using device: {device}")
    model = create_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Log hyperparameters to W&B
    if run:
        run.config.update({"epochs": epochs, "lr": lr, "batch_size": batch_size, "model": "resnet18"})

    history = {
        "epochs": epochs, "lr": lr, "batch_size": batch_size,
        "train_loss": [], "train_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()
            total += data.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}")

        # Log epoch metrics to W&B
        if run:
            run.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "epoch": epoch + 1})

    path = "model.pt"
    torch.save(model.state_dict(), path)
    model_file = await File.from_local(path)

    return model_file, json.dumps(history)


@wandb_init
@env.task
async def evaluate(model_file: File, data_dir: str) -> tuple[float, float]:
    """Evaluate model on test set, logging results to W&B."""
    import torch
    import torch.nn as nn
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    run = get_wandb_run()

    local_path = await model_file.download()
    device = get_device()
    print(f"Using device: {device}")
    model = create_model().to(device)
    model.load_state_dict(torch.load(local_path, map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    correct = 0
    total = 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            correct += (output.argmax(1) == target).sum().item()
            total += data.size(0)

    test_acc = correct / total
    test_loss = test_loss / total
    print(f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    # Log test results to W&B
    if run:
        run.log({"test_acc": test_acc, "test_loss": test_loss})
        run.summary["test_acc"] = test_acc
        run.summary["test_loss"] = test_loss

    return test_acc, test_loss


@wandb_init(project="flyte-mnist")
@env.task(report=True)
async def pipeline(epochs: int = 5, lr: float = 0.001, batch_size: int = 64) -> tuple[str, File]:
    """Full MNIST pipeline with W&B tracking — train, evaluate, and report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = get_wandb_run()
    if run:
        print(f"W&B run: {run.url}")

    data_dir = await load_data()
    model_file, history_json = await train(data_dir, epochs=epochs, lr=lr, batch_size=batch_size)
    test_acc, test_loss = await evaluate(model_file, data_dir)

    # Build HTML report with training curves
    history = json.loads(history_json)
    epoch_list = list(range(1, history["epochs"] + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epoch_list, history["train_loss"], "b-o", markersize=4)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax2.plot(epoch_list, history["train_acc"], "g-o", markersize=4)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Training Accuracy")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    wandb_link = f'<a href="{run.url}" target="_blank">View in W&B</a>' if run else ""

    await flyte.report.replace.aio(
        f"<h2>MNIST Training Report (W&B)</h2>"
        f"{f'<p>{wandb_link}</p>' if wandb_link else ''}"
        f"<h3>Hyperparameters</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td><b>Epochs</b></td><td>{history['epochs']}</td></tr>"
        f"<tr><td><b>Learning Rate</b></td><td>{history['lr']}</td></tr>"
        f"<tr><td><b>Batch Size</b></td><td>{history['batch_size']}</td></tr>"
        f"</table>"
        f"<h3>Training Curves</h3>{charts_html}"
        f"<h3>Test Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td><b>Test Accuracy</b></td><td>{test_acc:.4f}</td></tr>"
        f"<tr><td><b>Test Loss</b></td><td>{test_loss:.4f}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    return f"Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}", model_file


# Local:  WANDB_API_KEY=your-key flyte run --local --tui wandb_ml_pipeline.py pipeline --epochs 5 --lr 0.001
# Remote: flyte run wandb_ml_pipeline.py pipeline --epochs 5 --lr 0.001