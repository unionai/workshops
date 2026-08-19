# Monocular Depth Estimation — and 3D from a Single Photo

Turn one flat photo into per-pixel depth with **[Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)**, score it against a real depth sensor, and reconstruct a 3D parallax view — all zero-shot, no training, on CPU. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

Depth from a single image is one of the more surprising things a vision model can do: a photograph has no explicit distance information, yet a foundation model trained on enough of them learns the monocular cues humans use — perspective, occlusion, texture gradients, familiar object sizes. This demo shows that, and then does the honest thing: it holds the prediction to a physical depth sensor and measures how close it got.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `prepare_data` | Pull real RGB + Kinect-depth pairs from NYU Depth V2 | Sample RGB scenes with depth ranges |
| `estimate_depth` | Predict depth per image, align to ground truth, score, render (fans out) | (feeds the report) |
| `pipeline` | | **RGB / predicted / truth / error panels**, **3D parallax player**, accuracy chart |

## The Money Shot: 3D From One Photo

The report reconstructs a small camera sweep from the predicted depth — near pixels shift more than far ones, and the disocclusion holes behind foreground objects are filled in. A flat photo, given geometry by the model, viewed from angles the camera never saw. It plays as a subtle wobble and is the most immediate way to see that the model recovered real structure, not just a pretty heatmap.

## The Data

**[sayakpaul/nyu_depth_v2](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2)** — **Apache-2.0, ungated.** Indoor scenes, 640×480 RGB each paired with a **Microsoft Kinect** depth capture in metres. That paired ground truth is what makes this an *evaluation* rather than a gallery.

The dataset ships as a loading script plus webdataset tars. Loading scripts were removed in `datasets>=4.0`, so this fetches a tar directly with `hf_hub_download` and reads its HDF5 samples — the working path and the simpler one.

## The Model

**[Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)** — 24.8M params, **Apache-2.0**, ungated, native `transformers`. One forward pass per image, **~0.25 s on CPU**.

> **Licence note:** the Small model is Apache-2.0. The Base and Large variants are **CC-BY-NC** (non-commercial) — use Small unless you have a licence for the others.

## Relative depth, and why alignment is honest not cheating

Depth Anything V2 predicts **relative inverse depth** (disparity): nearer surfaces get larger values, but there is no absolute scale — it cannot tell a doll's house from a real room. To compare against a metric sensor, each prediction is aligned with a single per-image least-squares **scale and shift** (`ground_truth ≈ a · prediction + b`). This is the standard scale-invariant depth protocol; it corrects the one thing a monocular model genuinely cannot know (absolute scale) and nothing else, so the reported error reflects real structural accuracy.

The fitted `a` comes out **negative** — confirmation that the model outputs inverse depth while the sensor measures forward depth.

## Metrics

Standard scale-invariant depth metrics, over valid sensor pixels:

- **AbsRel** — mean absolute relative error, `|pred − gt| / gt`. Lower is better.
- **δ<1.25** — fraction of pixels within 25% of true depth (`max(p/g, g/p) < 1.25`). Higher is better; the headline number.
- **RMSE** — root-mean-square error in metres.

Zero-shot on NYU indoor scenes, per-image δ<1.25 lands around **0.83–0.86** with AbsRel near **0.12** — genuinely good for a model that never saw this data.

## Setup

```bash
cd tutorials/depth-estimation
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

# More scenes, smoother parallax
flyte run workflow.py pipeline --n_images 10 --parallax_frames 16
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_images` | `6` | Scenes to estimate and score |
| `stride` | `40` | Sampling stride through the tar (spreads scenes out) |
| `parallax_frames` | `12` | Frames in the 3D parallax sweep |
| `fps` | `12` | Parallax playback rate |

## Architecture

- **config.py** — CPU-only image + a **reusable** estimation pool for the fan-out
- **depth.py** — model load/cache, GT alignment, metrics, turbo colourisation, parallax
- **report_helpers.py** — panels, metric chart, parallax player
- **workflow.py** — tasks and pipeline

The estimation environment uses a `ReusePolicy` so the model loads once per warm replica rather than once per image. Depth panels are colourised with **turbo** (perceptually smooth, no false banding). Everything embeds as base64, so the report is a single self-contained file.

**Caching is off.** A cached task does not execute its body, and these reports are written by those bodies — a cache hit returns correct outputs with an empty report, silently.

## Where it fails, honestly

A single image gives no depth cue on reflective surfaces, glass, or untextured walls, and the error map lights up exactly there. Monocular depth is an inference from learned priors, not a measurement — which is the whole reason to score it against a sensor rather than trust the pretty picture.

## References

- Yang et al., "Depth Anything V2" ([arXiv:2406.09414](https://arxiv.org/abs/2406.09414))
- Silberman et al., "Indoor Segmentation and Support Inference from RGBD Images" (NYU Depth V2), ECCV 2012
