"""
Image Classification Training on Flyte

Fine-tune a pretrained ResNet on the Beans dataset from HuggingFace.

Usage:
    uv run flyte run image_classifier.py pipeline --num_epochs 3
"""

import torch
import torch.nn as nn
from torchvision import models, transforms

import flyte
import flyte.io

env = flyte.TaskEnvironment(
    name="image-classifier",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch", "torchvision", "datasets", "Pillow",
    ),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)


@env.task
async def load_data() -> flyte.io.File:
    """Download the Beans dataset and save as tensors."""
    from datasets import load_dataset

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = load_dataset("beans", split="train")
    images, labels = [], []
    for sample in ds:
        images.append(transform(sample["image"].convert("RGB")))
        labels.append(sample["labels"])

    path = "/tmp/beans_data.pt"
    torch.save({"images": torch.stack(images), "labels": torch.tensor(labels)}, path)
    print(f"Saved {len(images)} samples")
    return await flyte.io.File.from_local(path)


@env.task
async def train(data: flyte.io.File, num_epochs: int = 3, lr: float = 0.001) -> flyte.io.File:
    """Fine-tune ResNet18 on the Beans dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = await data.download()
    dataset = torch.load(data_path, weights_only=False)
    images, labels = dataset["images"].to(device), dataset["labels"].to(device)
    print(f"Training on {len(images)} images, {labels.unique().numel()} classes")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, labels.unique().numel())
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32

    for epoch in range(num_epochs):
        epoch_loss, correct = 0.0, 0
        for i in range(0, len(images), batch_size):
            batch_x = images[i:i + batch_size]
            batch_y = labels[i:i + batch_size]

            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x)
            correct += (preds.argmax(1) == batch_y).sum().item()

        acc = correct / len(images) * 100
        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {epoch_loss / len(images):.4f}, acc: {acc:.1f}%")

    path = "/tmp/resnet_beans.pt"
    torch.save(model.state_dict(), path)
    return await flyte.io.File.from_local(path)


@env.task
async def pipeline(num_epochs: int = 3) -> flyte.io.File:
    """Load data → Train model."""
    data = await load_data()
    model = await train(data, num_epochs=num_epochs)
    return model

# uv run flyte run image_classifier.py pipeline --num_epochs 3