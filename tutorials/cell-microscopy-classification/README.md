# Cell Microscopy Image Classification

Fine-tune a Vision Transformer (ViT) on blood cell microscopy images using Flyte/Union.ai. The pipeline downloads a HuggingFace image classification dataset, trains a ViT model, evaluates with per-class metrics and confusion matrix, and renders an inference demo with confidence visualizations.

Works with any HuggingFace dataset that has `image` and `label` columns.

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Full pipeline (default: ViT-base on blood cell images)
flyte run --local --tui workflow.py pipeline

# Quick local test
flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2

# Remote execution
flyte run workflow.py pipeline --epochs 10

# Custom dataset or model
flyte run workflow.py pipeline --dataset_name "Falah/Alzheimer_MRI"
flyte run workflow.py pipeline --model_name "google/vit-base-patch16-224-in21k"
```

### Run individual tasks

```bash
flyte run --local --tui workflow.py prepare_data
flyte run --local --tui workflow.py train --data_dir <path>
flyte run --local --tui workflow.py evaluate --finetuned_dir <path> --data_dir <path>
flyte run --local --tui workflow.py inference_demo --finetuned_dir <path> --data_dir <path>
```

## Pipeline

| Task | Environment | Description |
|------|-------------|-------------|
| `prepare_data` | CPU (cached) | Download HF dataset, split train/val, save images by class |
| `train` | GPU | Fine-tune ViT with live loss chart and progress bar |
| `evaluate` | GPU | Per-class accuracy, F1, precision/recall, confusion matrix heatmap |
| `inference_demo` | GPU | Visual grid of predictions with confidence bars |

## Reports

Each task generates a Flyte report with a purple/violet biotech theme:

- **prepare_data**: Dataset stats, class distribution chart, sample image grid
- **train**: Live training loss, learning rate schedule, progress bar
- **evaluate**: Accuracy bar chart, confusion matrix heatmap, metrics table
- **inference_demo**: Image grid with true/predicted labels and confidence bars
