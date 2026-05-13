# RT-DETRv2 Object Detection

Fine-tune **RT-DETRv2** (real-time DETR, v2) on a custom COCO-format dataset. The
pipeline downloads a COCO dataset from HuggingFace, trains the model, evaluates
COCO mAP, and renders bounding-box predictions on held-out images.

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
| `config.py` | Flyte environments — CPU for data prep, L4 GPU for train/eval |
| `workflow.py` | Pipeline: prepare data → train → evaluate → inference demo |

## Setup

```bash
cd tutorials/detr-object-detection

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

`requirements.txt` is pinned to **CUDA 13 aarch64 PyTorch wheels** for NVIDIA
DGX Spark (GB10 / Blackwell / sm_121). DGX Spark only ships `libcudart.so.13`,
so the default cu12x wheels from PyPI fail to import. PyTorch ≥ 2.9 cu130
wheels include sm_120 kernels which are binary-compatible with sm_121.

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expected: 2.9.x 13.0 / NVIDIA GB10 / (12, 1)
```

You can ignore the `Minimum and Maximum cuda capability ... (8.0) - (12.0)` warning.

**On non-DGX-Spark hosts** (other Blackwell, Ada, Hopper, Ampere boxes), edit
`requirements.txt` and swap the `--extra-index-url` line for the cu12x index
matching your GPU, e.g.:

```
--extra-index-url https://download.pytorch.org/whl/cu124
```

## Run

### Default (RT-DETRv2-R18 on Union swag stickers)

```bash
flyte run --local --tui workflow.py pipeline
```

### Quick test (smoke)

```bash
flyte run --local --tui workflow.py pipeline --epochs 2 --batch_size 2 --demo_images 2
```

### Remote (GPU cluster)

```
flyte create config \
    --endpoint tryv2.hosted.unionai.cloud \
    --project workshops \
    --domain development \
    --builder remote
```


```bash
flyte run workflow.py pipeline --epochs 30
```

Run the command from `tutorials/detr-object-detection` so Flyte builds the image
with the local workflow source and `requirements.txt`.

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
| `--batch_size` | `4` | Per-device batch size (see [Choosing batch size](#choosing-batch-size)) |
| `--val_fraction` | `0.2` | Fraction of images held out for validation |
| `--threshold` | `0.3` | Score threshold for predictions in eval/demo |
| `--demo_images` | `8` | Number of val images rendered in the inference report |

## Choosing batch size

Inputs are resized to 640×640 by the HF image processor. Rough guidance for
`--batch_size` with the default RT-DETRv2-R18 (mixed precision on):

| GPU | VRAM | R18 | R50 (`rtdetr_v2_r50vd`) |
|---|---|---|---|
| T4 | 16 GB | **4** (default) — 8 cuts close | 2 |
| L4 / A10 | 24 GB | 8–16 | 4–8 |
| DGX Spark | 128 GB unified | 16–32 | 16 |

Drop to `--batch_size 2` if you OOM. On DGX Spark, bump to 16+ to keep the GPU
fed.

## Evaluation

`evaluate` runs both the **pretrained base model** and the **fine-tuned model**
over the val split and computes COCO mAP via `torchmetrics`. The base model was
pretrained on COCO's 80 classes, so its predicted labels don't line up with the
custom 2-class label space — that's why base mAP is ~0 by construction. The lift
comes from teaching the decoder to predict our category ids.

Metrics reported: `map`, `map_50`, `map_75`, `mar_1`, `mar_10`.

## Inference Demo

`inference_demo` renders ground-truth boxes (red) next to fine-tuned predictions
(green) for several val images and embeds them in the Flyte report — so you can
eyeball detection quality without leaving the run UI.

## Notes on the default dataset

`sagecodes/union_swag_coco` has only ~18 images. It's enough to demo the
pipeline end-to-end and see qualitatively good predictions on the stickers, but
mAP numbers won't be impressive — swap in a larger dataset (`--dataset_repo`)
for serious numbers.
