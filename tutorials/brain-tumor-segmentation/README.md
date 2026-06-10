# 3D Brain Tumor Segmentation with SegResNet

Train a SegResNet to segment brain tumors in 3D MRI volumes. The model learns
to identify three tumor subregions — **enhancing tumor** (active growth),
**necrotic core** (dead tissue), and **peritumoral edema** (swelling) — from
multi-modal MRI scans (T1, T1-contrast, T2, FLAIR).

**Dataset:** [BraTS 2023 GLI](https://huggingface.co/datasets/Angelou0516/brats2023-gli-dataset) — 1,251 glioma cases with 4 co-registered MRI modalities + voxel-level segmentation labels. CC-BY-4.0.

**Model:** SegResNet via [MONAI](https://monai.io/) — a 3D residual encoder-decoder designed for volumetric medical image segmentation.

## Pipeline

```
prepare_data → train → evaluate → inference
```

| Task | What it does | Report visuals |
|------|-------------|----------------|
| `prepare_data` | Download BraTS 2023 from HuggingFace, split train/val | 4-modality MRI panel, axial/coronal/sagittal tumor overlays |
| `train` | Train SegResNet with MONAI (patch-based 3D training) | Live loss curve, validation Dice per region (WT/TC/ET) |
| `evaluate` | Dice score per composite tumor region | Dice bar chart, per-region metrics table |
| `inference` | Side-by-side GT vs predicted 3-plane overlays | Multi-plane tumor overlays for sample cases |

## Setup

```bash
cd tutorials/brain-tumor-segmentation
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Local (quick test — 20 cases, 5 epochs)

```bash
flyte run --local --tui workflow.py pipeline --max_cases 20 --epochs 5
```

### Local (100 cases, 30 epochs)

```bash
flyte run --local --tui workflow.py pipeline
```

### Remote (full dataset)

```bash
flyte run workflow.py pipeline --max_cases 0 --epochs 50
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_repo` | `Angelou0516/brats2023-gli-dataset` | HuggingFace dataset |
| `max_cases` | `100` | Max cases to use (0 = all 1,251) |
| `epochs` | `30` | Training epochs |
| `batch_size` | `1` | Volumes per batch (3D data is memory-heavy) |
| `learning_rate` | `1e-4` | Adam learning rate |
| `patch_size` | `128` | Crop size for training (128³ voxels) |
| `val_fraction` | `0.15` | Fraction of cases for validation |
| `demo_cases` | `4` | Cases to show in inference demo |

## Tumor subregions

| Region | Label | Color | Description |
|--------|-------|-------|-------------|
| NCR | 1 | Yellow | Necrotic/non-enhancing tumor core |
| ED | 2 | Green | Peritumoral edema (swelling) |
| ET | 4 | Red | Gadolinium-enhancing tumor (active growth) |

Evaluation uses **composite regions** (standard BraTS metrics):
- **Whole Tumor (WT)** = NCR + ED + ET — easiest, largest region
- **Tumor Core (TC)** = NCR + ET — core without edema
- **Enhancing Tumor (ET)** = ET only — hardest, smallest region

## MRI modalities

Each patient has 4 co-registered MRI volumes (240x240x155 voxels, 1mm isotropic):

| Modality | Suffix | Clinical purpose |
|----------|--------|-----------------|
| T1 | `t1n` | Structural anatomy |
| T1-contrast | `t1c` | Highlights enhancing tumor (active growth) |
| T2 | `t2w` | Shows water content, swelling |
| FLAIR | `t2f` | Identifies peritumoral edema |

## Architecture notes

This tutorial uses **MONAI** for the entire 3D medical imaging pipeline:
- `LoadImaged` / `NormalizeIntensityd` — NIfTI loading and intensity normalization
- `CropForegroundd` / `RandSpatialCropd` — crop to brain region, then random 128³ patches
- `ConvertToMultiChannelBasedOnBratsClassesd` — converts BraTS labels to 3-channel (WT, TC, ET)
- `sliding_window_inference` — runs inference on full volumes using overlapping patches
- `DiceLoss` + `DiceMetric` — standard BraTS training loss and evaluation metric

**Patch-based training** is necessary because full 240x240x155 volumes with 4 channels
don't fit in GPU memory. The model trains on random 128³ crops, then at inference time
uses sliding window with 50% overlap to produce full-volume predictions.

## Visual highlights

The inference report shows **axial, coronal, and sagittal** views centered on the tumor
mass, with color-coded overlays:
- **Yellow** = necrotic core (dead tissue at tumor center)
- **Green** = peritumoral edema (swelling around tumor)
- **Red** = enhancing tumor (active growth, visible on T1-contrast)

Ground truth and predicted segmentations are shown side-by-side with voxel counts
per subregion, making it easy to see where the model agrees with expert annotations
and where it diverges.

## References

- Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE TMI 2015
- Myronenko, "3D MRI Brain Tumor Segmentation Using Autoencoder Regularization" ([arXiv:1810.11654](https://arxiv.org/abs/1810.11654))
- MONAI Project: [monai.io](https://monai.io/)
