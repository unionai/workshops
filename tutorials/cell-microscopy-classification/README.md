# Cell Microscopy Image Classification

Fine-tune a [Vision Transformer (ViT)](https://huggingface.co/google/vit-base-patch16-224) to classify blood cell types from microscopy images, orchestrated as a reproducible [Flyte](https://docs.flyte.org/) pipeline on [Union.ai](https://www.union.ai/).

## Why This Matters

Peripheral blood smear analysis is one of the most common diagnostic procedures in clinical hematology. Technicians examine stained blood samples under a microscope to identify and count different cell types — a process that is time-consuming, subjective, and prone to inter-observer variability.

Automated classification of blood cells from microscopy images can:

- **Speed up diagnosis** — reduce turnaround from hours to seconds for routine blood work
- **Improve consistency** — eliminate variability between technicians and across labs
- **Scale screening** — enable high-throughput analysis in resource-limited settings where trained hematologists are scarce
- **Support rare cell detection** — flag abnormal or rare cell types (e.g., erythroblasts, plasma cells) that might be missed in manual review

This tutorial classifies 8 cell types found in peripheral blood smears:

| Cell Type | Role |
|-----------|------|
| **Neutrophil** | Most abundant white blood cell; first responder to bacterial infections |
| **Lymphocyte** | Drives adaptive immunity (T-cells, B-cells); key in viral defense |
| **Monocyte** | Differentiates into macrophages; cleans up dead cells and pathogens |
| **Eosinophil** | Fights parasitic infections; involved in allergic responses |
| **Basophil** | Rarest white blood cell; releases histamine in allergic and inflammatory reactions |
| **Platelet** | Cell fragment essential for blood clotting and wound repair |
| **Erythroblast** | Immature red blood cell precursor; elevated counts can signal bone marrow stress |
| **Plasma Cell** | Antibody-producing B-cell; increased presence may indicate infection or myeloma |

## Dataset

This pipeline uses [ehottl/blood_dataset](https://huggingface.co/datasets/ehottl/blood_dataset) by default — 46k labeled microscopy images across the 8 classes above. The pipeline accepts any HuggingFace dataset with `image` and `label` columns, so you can swap in other domains (e.g., Alzheimer's MRI, skin lesion classification) with a single flag.

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

# Remote execution on Union.ai
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
flyte run --local --tui workflow.py inference --finetuned_dir <path> --data_dir <path>
```

## Pipeline

| Task | Environment | What it does |
|------|-------------|--------------|
| `prepare_data` | CPU (cached) | Download HF dataset, split train/val, save images organized by class |
| `train` | GPU | Fine-tune ViT with live loss chart and learning rate schedule |
| `evaluate` | GPU | Per-class accuracy, F1, precision/recall, and confusion matrix heatmap |
| `inference` | GPU | Visual grid of sample predictions with confidence bars |

## Reports

Each task generates a live Flyte report:

- **prepare_data** — Dataset stats, class distribution chart, sample image grid
- **train** — Live training loss curve, learning rate schedule, progress bar
- **evaluate** — Per-class accuracy bar chart, confusion matrix heatmap, metrics table
- **inference** — Image grid with true/predicted labels and confidence bars

## How It Works

1. **Data preparation** — Downloads the dataset from HuggingFace, handles pre-defined or automatic train/val splits, and saves images into class-organized directories. Data prep is cached so re-runs skip the download.

2. **Fine-tuning** — Loads a pre-trained ViT (ImageNet weights) and replaces the classification head for the target classes. Trains with AdamW optimizer and cosine learning rate scheduling. The report updates live with the loss curve as training progresses.

3. **Evaluation** — Runs the fine-tuned model on the validation set, computing per-class precision, recall, F1, and a confusion matrix. This surfaces which cell types the model struggles to distinguish (e.g., monocytes vs. lymphocytes is a classically hard boundary).

4. **Inference** — Picks random validation images and runs predictions with full confidence breakdowns, giving a visual gut-check of model behavior beyond aggregate metrics.

## Project Structure

```
cell-microscopy-classification/
  config.py            # Flyte environment config (CPU/GPU resources, image)
  workflow.py           # Full pipeline: data prep, training, eval, inference
  requirements.txt      # Python dependencies
  README.md
```
