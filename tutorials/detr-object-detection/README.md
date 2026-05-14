# RT-DETRv2 Object Detection

Fine-tune **RT-DETRv2** (real-time DETR, v2) on a custom COCO-format dataset. The
pipeline downloads a COCO dataset from HuggingFace, trains the model, evaluates
COCO mAP, and renders bounding-box predictions on held-out images. Then deploy a
live detection app with a FastAPI model server and Gradio frontend.

The default dataset is [`sagecodes/union_swag_coco`](https://huggingface.co/datasets/sagecodes/union_swag_coco) — a tiny 2-class
sticker dataset (Flyte / Union stickers) intended for demoing the pipeline, not
for chasing SOTA.

## Why RT-DETR?

| | DETR | RT-DETR / RT-DETRv2 |
|---|---|---|
| End-to-end (no NMS) | yes | yes |
| Encoder | full transformer | hybrid (CNN + lightweight transformer) |
| Throughput | slow | real-time |
| Accuracy on COCO | baseline | matches or beats YOLO at similar speed |

RT-DETRv2 keeps the DETR set-prediction philosophy but adds a hybrid encoder and
IoU-aware query selection. The HuggingFace API is identical to DETR, so swapping
between them is a one-line change.

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

The pipeline accepts any HF dataset that ships a COCO-format JSON and an image
directory:

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
| `--eval_every_n_epochs` | `None` | Run mAP eval every N epochs during training (disabled by default) |

## Deploy the Detection App

After training, deploy a live detection service with a FastAPI model server and
Gradio web UI.

### 1. Deploy the model server

The server auto-discovers the latest successful training run, or you can pass
a specific run name:

```bash
# Auto-detect latest training run
python app_server.py

# Or specify a run
TRAINING_RUN=<run_name> python app_server.py
```

### 2. Deploy the Gradio frontend

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

## Reports

The pipeline generates styled reports visible in the Flyte UI:

- **Training report** — live-updating loss chart, learning rate schedule, and periodic mAP chart (if `eval_every_n_epochs` is set)
- **Evaluation report** — COCO mAP metrics table and bar chart
- **Inference demo** — side-by-side ground truth vs predictions with per-image mAP scores

## Choosing batch size

Inputs are resized to 640x640 by the HF image processor:

| GPU | VRAM | R18 | R50 (`rtdetr_v2_r50vd`) |
|---|---|---|---|
| T4 | 16 GB | **4** (default) — 8 cuts close | 2 |
| L4 / A10 | 24 GB | 8-16 | 4-8 |
| DGX Spark | 128 GB unified | 16-32 | 16 |

## Notes on the default dataset

`sagecodes/union_swag_coco` has only ~18 images. It's enough to demo the
pipeline end-to-end and see qualitatively good predictions, but mAP numbers
won't be impressive — swap in a larger dataset (`--dataset_repo`) for serious
evaluation.
