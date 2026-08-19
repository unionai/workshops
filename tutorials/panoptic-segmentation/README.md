# Panoptic Segmentation with Mask2Former

Label **every pixel** of an image in a single pass — each object its own instance mask with a box and confidence, each background region its own class — with [Mask2Former](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic) on COCO, compared against ground truth. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

Panoptic segmentation is the most complete image-understanding task: it unifies **object detection** (find and box the countable "things" — people, cars, cups) and **semantic segmentation** (label the amorphous "stuff" — sky, road, grass) so that no pixel is left unassigned. One model, one forward pass, the whole scene explained.

This is the canonical computer-vision deliverable a perception team ships: colourful instance masks, class labels, confidences, and a scene inventory.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `prepare_data` | Pull the busiest COCO val images (most ground-truth segments) | Sample scenes with segment counts |
| `segment_image` | Run Mask2Former per image, render overlay + inventory (fans out) | (feeds the report) |
| `pipeline` | | **Input / predicted / ground-truth triptych**, per-class object chart, scene inventory |

## The Money Shot

For each scene, three panels side by side: the **input image**, the **predicted panoptic overlay** (every segment a distinct colour, every object boxed and labelled with its confidence), and the **COCO ground-truth panoptic map**. Plus a per-scene inventory — "3 people, a tie, a dining table · sky, floor, wall" — and a bar chart of every object class detected across all scenes.

## The Data

**[nielsr/coco-panoptic-val2017](https://huggingface.co/datasets/nielsr/coco-panoptic-val2017)** — ungated, plain parquet, the COCO 2017 validation split with panoptic ground truth (`segments_info` + a panoptic PNG per image). Scenes are ranked by ground-truth segment count so the report shows dense, information-rich images — a crowded banquet, not an empty field.

The ground-truth panoptic PNG encodes the segment id per pixel as `R + G·256 + B·256²`; the pipeline decodes it and colours each ground-truth segment distinctly for the side-by-side.

## The Model

**[facebook/mask2former-swin-base-coco-panoptic](https://huggingface.co/facebook/mask2former-swin-base-coco-panoptic)** — ungated, native `transformers`. One universal architecture that does instance, semantic, and panoptic segmentation. **~0.3 s per image on CPU** — a GPU would queue longer than it computes, so this runs CPU-only.

Applied **zero-shot**: it is the pretrained COCO panoptic checkpoint, run as-is.

## Reading the Result

- **Pixel coverage ≈ 100%** is the signature of panoptic segmentation — every pixel gets a label. Instance or semantic segmentation alone would leave background (or foreground) unassigned.
- **Predicted vs ground-truth colours do not correspond.** Each panel colours its own segments independently, so match the *regions*, not the hues.
- Ground truth usually carries **more segments** than the prediction: it labels every distant background instance in a crowded scene, while the model recovers the salient objects and large regions cleanly.

## Setup

```bash
cd tutorials/panoptic-segmentation
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Local
flyte run --local --tui workflow.py pipeline --n_images 4

# Remote
flyte run workflow.py pipeline

# More scenes, scan deeper for busy ones
flyte run workflow.py pipeline --n_images 10 --scan 600
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_images` | `6` | Scenes to segment |
| `scan` | `300` | How many val images to scan when ranking by busy-ness |

## Architecture

- **config.py** — CPU-only image + a **reusable** segmentation pool for the fan-out
- **segment.py** — model load/cache, distinct-colour overlay, boxes/labels, GT decode, inventory
- **report_helpers.py** — triptych panels, inventory chips, class chart
- **workflow.py** — tasks and pipeline

**Mask2Former's post-processing requires `scipy`** — without it the failure is an opaque `requires_backends` ImportError at *first inference*, not at import, so it survives every import check and dies mid-run. It is pinned in the image.

Segments are coloured by a golden-angle hue walk, which gives maximally distinct colours without committing to a fixed palette size. Everything embeds as base64, so the report is a single self-contained file.

**Caching is off.** A cached task does not execute its body, and these reports are written by those bodies — a cache hit returns correct outputs with an empty report, silently.

## Related

- [depth-estimation](../depth-estimation/) — a different CV modality: per-pixel depth from a single photo.

## References

- Cheng et al., "Masked-attention Mask Transformer for Universal Image Segmentation" ([arXiv:2112.01527](https://arxiv.org/abs/2112.01527))
- Kirillov et al., "Panoptic Segmentation", CVPR 2019
