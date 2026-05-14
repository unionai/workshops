# RT-DETRv2 Object Detection

Fine-tune **RT-DETRv2** (real-time DETR, v2) on a custom COCO-format dataset, evaluate with COCO mAP, and deploy a live detection app — all orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why Object Detection Matters

Object detection is one of the most practical applications of deep learning. It powers autonomous vehicles, medical imaging, warehouse robotics, quality control on manufacturing lines, retail analytics, and more. Unlike image classification (which asks "what's in this image?"), object detection answers "what's in this image, where is it, and how many?"

Being able to fine-tune a detector on your own data — your products, your defects, your domain — and deploy it as a reliable service is a superpower for any team building real-world AI applications.

## Why DETR?

Traditional object detectors (Faster R-CNN, YOLO) rely on hand-designed components like anchor boxes and non-maximum suppression (NMS). **DETR** (DEtection TRansformer) replaced all of that with a transformer and a simple set-prediction loss. No anchors, no NMS, no post-processing hacks — just a clean end-to-end architecture.

**RT-DETRv2** takes this further with a hybrid CNN + transformer encoder for real-time speed while keeping the elegant DETR design. It matches or beats YOLO at similar speeds.

| | DETR | RT-DETR / RT-DETRv2 |
|---|---|---|
| End-to-end (no NMS) | yes | yes |
| Encoder | full transformer | hybrid (CNN + lightweight transformer) |
| Throughput | slow | real-time |
| Accuracy on COCO | baseline | matches or beats YOLO at similar speed |

The HuggingFace API is identical across DETR variants, so swapping between them is a one-line change.

## Why Flyte / Union?

ML pipelines are messy. Data prep, training, evaluation, and deployment each have different resource needs, failure modes, and iteration cycles. Flyte gives you:

- **Reproducibility** — every run is versioned, cached, and traceable
- **Resource isolation** — CPU for data prep, GPU for training, lightweight containers for serving
- **Live reports** — watch training loss and mAP charts update in real-time in the UI
- **Seamless deployment** — train a model, then deploy it as a FastAPI endpoint + Gradio app with a few commands
- **Caching** — data prep is cached automatically, so re-runs skip expensive downloads
- **Scale** — run on your laptop or a GPU cluster with the same code

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for data prep, GPU for train/eval |
| `workflow.py` | Pipeline: prepare data → train → evaluate → inference demo |
| `app_server.py` | FastAPI model server — serves the fine-tuned model for inference |
| `app_gradio.py` | Gradio frontend — image upload + webcam detection UI |

## Setup

```bash
cd tutorials/detr-object-detection

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Flyte config (for remote runs)

```bash
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --project workshopdetr \
    --domain development \
    --builder remote
```

## Run the Training Pipeline

### Default (RT-DETRv2-R18 on Union swag stickers)

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test (smoke)

```bash
flyte run --local --tui workflow.py pipeline --epochs 2 --batch_size 2 --demo_images 2
```

### Remote (GPU cluster)

```bash
flyte run workflow.py pipeline --epochs 30
```

### With periodic mAP evaluation

Track mAP during training to catch overfitting or know when to stop:

```bash
flyte run workflow.py pipeline --epochs 50 --eval_every_n_epochs 10
```

### Swap model

```bash
# Larger RT-DETRv2 backbone (ResNet-50)
flyte run workflow.py pipeline --model_name "PekingU/rtdetr_v2_r50vd"

# RT-DETR v1 for comparison
flyte run workflow.py pipeline --model_name "PekingU/rtdetr_r18vd"

# Plain DETR (slower, original architecture)
flyte run workflow.py pipeline --model_name "facebook/detr-resnet-50"
```

### Swap dataset

The pipeline accepts any HF dataset with a COCO-format JSON and image directory:

```bash
flyte run workflow.py pipeline \
  --dataset_repo "your-org/your-coco-dataset" \
  --annotations_path "annotations/train.json" \
  --images_subdir "images"
```

## Pipeline Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model_name` | `PekingU/rtdetr_v2_r18vd` | HuggingFace object-detection model |
| `--dataset_repo` | `sagecodes/union_swag_coco` | HF dataset repo id |
| `--annotations_path` | `swag/train.json` | Path to COCO JSON inside the repo |
| `--images_subdir` | `swag/images` | Path to image directory inside the repo |
| `--epochs` | `30` | Training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `4` | Per-device batch size |
| `--val_fraction` | `0.2` | Fraction of images held out for validation |
| `--threshold` | `0.5` | Score threshold for predictions in eval/demo |
| `--demo_images` | `8` | Number of val images rendered in the inference report |
| `--eval_every_n_epochs` | `None` | Run mAP eval every N epochs during training |

## Deploy the Detection App

After training, deploy a live detection service: a FastAPI model server + Gradio web UI.

### 1. Deploy the model server

The server auto-discovers the latest successful training run:

```bash
# Auto-detect latest pipeline run
python app_server.py

# Or specify a run
TRAINING_RUN=<run_name> python app_server.py
```

### 2. Deploy the Gradio frontend

Auto-discovers the server endpoint:

```bash
python app_gradio.py
```

### Local development

Run both locally without a cluster:

```bash
# Terminal 1: model server (point at a local fine-tuned model)
MODEL_PATH=/path/to/finetuned_model python app_server.py

# Terminal 2: Gradio frontend
SERVER_URL=http://localhost:8080 python app_gradio.py
```

### App API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + model status |
| `/classes` | GET | List detected classes |
| `/detect` | POST | Upload image file, get bounding boxes |
| `/detect_base64` | POST | Send base64 image (webcam), get bounding boxes |
| `/docs` | GET | Interactive API docs (FastAPI auto-generated) |

## Reports

The pipeline generates styled reports visible in the Flyte UI:

- **Training** — live-updating loss chart, LR schedule, and periodic mAP (if enabled). Progress bar with step/epoch counts.
- **Evaluation** — COCO mAP metrics table and bar chart with explanations of each metric.
- **Inference demo** — side-by-side ground truth vs predictions with per-image mAP scores.

## Data Augmentation

The training pipeline applies online augmentations via albumentations to increase data variety without expanding the dataset on disk:

- Horizontal/vertical flip
- Brightness, contrast, hue, and saturation jitter
- Small rotations (+/-15 degrees)
- Random scale (+/-20%)
- Gaussian blur and noise

All augmentations are bbox-aware — bounding box coordinates are automatically transformed to match the augmented image.

## Understanding the Metrics

- **mAP** (mean Average Precision) — the primary COCO metric. Averaged across IoU thresholds 0.50 to 0.95. Higher is better.
- **mAP@50** — mAP at a lenient 50% IoU overlap. Usually higher than mAP.
- **mAP@75** — mAP at a strict 75% IoU overlap. Tests precise box localization.
- **mAR@10** — mean Average Recall with up to 10 detections per image. Measures how many ground-truth objects the model finds.

## Choosing batch size

Inputs are resized to 640x640 by the HF image processor:

| GPU | VRAM | R18 | R50 (`rtdetr_v2_r50vd`) |
|---|---|---|---|
| T4 | 16 GB | **4** (default) | 2 |
| L4 / A10 | 24 GB | 8-16 | 4-8 |
| DGX Spark | 128 GB unified | 16-32 | 16 |

## Notes on the default dataset

`sagecodes/union_swag_coco` has only ~18 images. It's enough to demo the full pipeline end-to-end, but swap in a larger dataset (`--dataset_repo`) for meaningful mAP numbers.
