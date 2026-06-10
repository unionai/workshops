# Multi-Class Nuclei Segmentation with Mask R-CNN

Fine-tune a Mask R-CNN (ResNet-50 FPN v2) to detect and classify individual
cell nuclei in pathology images. Five nucleus types — **neoplastic** (tumor),
**inflammatory** (immune), **connective** (stromal), **dead** (necrotic), and
**epithelial** (normal) — each get their own colored mask overlay.

**Dataset:** [PanNuke](https://huggingface.co/datasets/RationAI/PanNuke) — 190k nuclei across 19 tissue types with per-instance masks and class labels.

**Model:** Mask R-CNN ResNet-50 FPN v2 (COCO pretrained, fine-tuned for 5-class nuclei segmentation).

## Pipeline

```
prepare_data → train → evaluate → inference
```

| Task | What it does | Report visuals |
|------|-------------|----------------|
| `prepare_data` | Download PanNuke from HuggingFace, organize masks by instance, split folds | Class-colored mask overlays, class distribution chart, sample images |
| `train` | Fine-tune Mask R-CNN with custom PyTorch training loop | Live loss curves (total, mask, classifier, box reg), LR schedule |
| `evaluate` | COCO mAP for both bounding boxes and instance masks | BBox vs Segm mAP bar charts, per-class detection counts |
| `inference` | Side-by-side ground truth vs predicted mask overlays | Class-colored masks with per-nucleus-type counts, confidence scores |

## Setup

```bash
cd tutorials/nuclei-segmentation
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Local (quick test)

```bash
flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2
```

### Local (full training)

```bash
flyte run --local --tui workflow.py pipeline --epochs 10
```

### Remote (Union)

```bash
flyte run workflow.py pipeline --epochs 10
```

### Individual tasks

```bash
# Just prepare and cache the data
flyte run --local --tui workflow.py prepare_data

# Train with custom hyperparameters
flyte run --local --tui workflow.py train \
  --data_dir /path/to/cached/data \
  --epochs 15 \
  --learning_rate 0.003
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | `RationAI/PanNuke` | HuggingFace dataset |
| `train_folds` | `fold1,fold2` | Comma-separated fold names for training |
| `val_fold` | `fold3` | Fold name for validation |
| `epochs` | `10` | Training epochs |
| `batch_size` | `2` | Images per batch (Mask R-CNN is memory-heavy) |
| `learning_rate` | `0.005` | SGD learning rate |
| `score_threshold` | `0.5` | Confidence threshold for predictions |
| `sample_images` | `8` | Number of images in inference demo |

## Nucleus classes

| Class | Color | Description |
|-------|-------|-------------|
| Neoplastic | Red | Tumor/cancer cells |
| Inflammatory | Blue | Immune cells (lymphocytes, macrophages) |
| Connective | Green | Stromal/support cells (fibroblasts) |
| Dead | Gray | Necrotic or apoptotic cells |
| Epithelial | Purple | Normal epithelial cells |

## Architecture notes

This tutorial uses a **custom PyTorch training loop** instead of HuggingFace
Trainer. Torchvision's detection models (Mask R-CNN, Faster R-CNN, etc.) compute
losses internally when given both images and targets in training mode:

```python
model.train()
loss_dict = model(images, targets)  # Returns dict of losses
# loss_dict contains: loss_classifier, loss_box_reg, loss_mask,
#                     loss_objectness, loss_rpn_box_reg
```

Five separate loss components are tracked and charted in the live training
report — giving detailed insight into which parts of the model are converging.

## Visual highlights

The key visual feature is **class-colored instance mask overlays**. Each nucleus
is colored by its cell type (red=neoplastic, blue=inflammatory, etc.) with white
contour lines marking instance boundaries.

The inference report shows side-by-side comparisons with per-class nuclei counts:
- **Left:** Ground truth masks (pathologist annotations)
- **Right:** Model predictions (with confidence scores)

This makes it immediately clear not just *where* the model detects nuclei, but
*what type* it thinks each one is — critical for clinical pathology applications
like tumor grading and immune cell quantification.

## References

- **PanNuke paper:** Gamper et al., "PanNuke Dataset Extension, Insights and Baselines" ([arXiv:2003.10778](https://arxiv.org/abs/2003.10778)) — 200k semi-automatically annotated nuclei across 19 tissue types, quality-controlled by clinical pathologists.
