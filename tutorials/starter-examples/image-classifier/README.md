# Image Classification Training

Fine-tune a pretrained ResNet18 on the Beans dataset from HuggingFace.

## What it does

- **`load_data`** — Downloads the Beans dataset (3 classes, ~1000 images), applies ImageNet transforms, saves as tensors
- **`train`** — Fine-tunes ResNet18 with a replaced classification head, trains with Adam + CrossEntropyLoss
- **`pipeline`** — Orchestrates load data -> train

## Setup

```bash
cd tutorials/starter-examples/image-classifier

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Flyte Cluster (for remote runs)

To run remotely, configure your Flyte cluster endpoint:

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

Don't have a cluster? Request access at [flyte.org](https://flyte.org/).

## Run

**Remote:**
```bash
uv run flyte run image_classifier.py pipeline --num_epochs 3
```

**Local:**
```bash
uv run flyte run --local image_classifier.py pipeline --num_epochs 3
```