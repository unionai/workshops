# Lance Streaming for Vision — convert once, stream forever

Turn a swarm of tiny per-sample image files into a single **[Lance](https://lancedb.github.io/lance/)** dataset, then stream it straight from object storage into a real training loop — orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

Uses a **real** detection dataset ([CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) — medical PPE, ~1k images, 5 classes, openly licensed) by default, with a fully-offline synthetic mode as a fallback. Every number in the reports is measured — **nothing is simulated**.

## The problem

Computer-vision training data is often stored as a **huge number of tiny image files** — one small image per sample, frequently multiplied by a per-sample variant dimension (multiple views, crops, tiles, or augmentations), and often paired with a per-sample label mask. A modest dataset therefore explodes into thousands or millions of tiny objects in storage.

The bytes aren't the problem. The problem is that a training loop reads these files **one at a time**, paying a fresh object-store connection setup **per file**. That per-file tax dominates the wall-clock, and it repeats on every epoch of every run.

Two different access patterns, two different tools:

| Access pattern | Right tool |
|---|---|
| Random-access, folder-scoped reads (browse a subset, grab a few files) | **[Volumes](https://docs.union.ai/)** — a mountable, folder-scoped filesystem |
| Streaming the whole dataset into training, run after run | **Lance** — a streaming-optimized, columnar, multimodal format |

Volumes solve the first. They don't fix streaming for training — that's what Lance is for.

## The fix

1. **Convert once.** Fold the whole tree of tiny files into a single Lance dataset. Image bytes, mask bytes, and structured labels (bounding boxes, class ids) all live together in one row — one multimodal, columnar artifact.
2. **Stream forever.** Every run reads that one dataset directly from object storage — the benchmark and eval stream it sequentially, and training draws a **shuffled** random order each epoch. No per-file connection setup, and Lance pulls only the columns each batch needs.

## Why Lance (vs Parquet / WebDataset)?

Parquet is columnar but built for analytical scans, not for feeding a training loop. Tar/WebDataset streams well but can't do random access. Lance does both: fast **row-range streaming** *and* fast **random access** (shuffled reads), first-class **multimodal** columns (raw image bytes next to labels), and zero-copy to Arrow/PyTorch. This tutorial exercises both modes — sequential streaming in the benchmark, and shuffled random access (`Dataset.take`) to feed SGD, which is exactly what a tar-based pipeline can't do efficiently.

`format_comparison.py` makes this concrete — a standalone micro-benchmark (separate from the production pipeline) that writes the same data to Parquet and Lance and measures: sequential scan (a tie), memory to shuffle an epoch (Lance holds one batch, Parquet the whole dataset), and random-row fetch (Lance reads only what you ask for). Run it with `uv run flyte run --local format_comparison.py compare_formats`.

## Why Flyte / Union?

- **Reproducible conversion** — the one-time convert step is a cached, versioned task. Re-runs skip it.
- **Right-sized resources** — CPU for data prep, a T4 GPU for training, same code and image.
- **Live reports** — each stage renders its own charts/gallery in the UI.
- **Streams from object storage with zero setup** — the task's IAM role is used automatically; no credentials to configure.
- **Portable** — runs on your laptop (`--local`) and on a Union cluster unchanged.

## What's here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — `cpu_env` for data/benchmark, `gpu_env` (T4) for train/eval/inference, sharing one CUDA image |
| `lance_dataset.py` | Pure-Python core: synthesize tiny files / load CPPE-5, convert to Lance, sequential + shuffled readers, torch `IterableDataset` (no Flyte dep — run it directly) |
| `workflow.py` | The 6-stage production pipeline |
| `format_comparison.py` | Standalone micro-benchmark: Lance vs Parquet read patterns (not part of the pipeline) |
| `report_helpers.py` | Inline-SVG report styling, bar + line charts (no matplotlib) |
| `requirements.txt` | Pinned deps (`pylance`, `pyarrow`, `pillow`, `torch`/`torchvision`, `torchmetrics`, `datasets`) |

The pipeline stages:

1. **`prepare_tiny_files`** — materialize the raw dataset as tiny per-sample files (image PNG + label JSON). Default: download **CPPE-5** and write each image out individually, recreating the "swarm of tiny objects in storage" layout. `--source synthetic` generates scenes offline instead (adds a mask PNG per sample).
2. **`convert_to_lance`** — one-time conversion of that tree into a single `.lance` dataset.
3. **`benchmark_loading`** — race the two data paths against real object storage: open-per-file (today) vs stream-from-Lance. Renders a throughput chart.
4. **`stream_train`** — a real object detector (torchvision Faster R-CNN, MobileNetV3 backbone) trained on a **T4**, reading **shuffled batches straight from the Lance dataset's `s3://` URI** (random access, no download), with an OneCycle LR schedule. Saves weights to a model `Dir`.
5. **`evaluate`** — load the trained model, stream a held-out validation split from Lance, and compute COCO **mean average precision** (torchmetrics), with per-class AP.
6. **`explore_inference`** — run the detector on validation frames and render **predicted boxes (green) vs ground truth (blue)** with class names and scores.

## Setup

```bash
cd tutorials/lance-streaming-vision

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Quick sanity check of the core (no Flyte, no cluster — uses synthetic data):

```bash
python lance_dataset.py     # generate → convert → stream, all local
```

CPPE-5 downloads from the HuggingFace Hub with no auth. To avoid rate limits (and for faster downloads) you can optionally set `HF_TOKEN`.

## Run

**Local (real CPPE-5, small + fast — CPU):**

```bash
uv run flyte run --local workflow.py pipeline \
  --max_images 60 --resize_max 480 --max_train_steps 20
```

**Local (synthetic, fully offline):**

```bash
uv run flyte run --local workflow.py pipeline \
  --source synthetic --num_groups 1 --scenes_per_group 5 --views_per_scene 10
```

**Remote (Union) — the full demo (real object storage + T4):**

```bash
uv run flyte run workflow.py pipeline
```

The defaults are tuned for a good result (1000 images, 1500 shuffled training steps). The benchmark reads the Flyte `Dir`s straight from object storage (`s3://…`), and `stream_train`/`evaluate`/`explore_inference` run on the T4 — no flags needed.

**Remote — quick pass** (smaller/faster, still real numbers):

```bash
uv run flyte run workflow.py pipeline \
  --max_images 400 --resize_max 480 --train_batch_size 8 --max_train_steps 500
```

### Flyte config (for remote runs)

```bash
flyte create config \
    --endpoint <your-endpoint> \
    --auth-type headless \
    --builder remote \
    --domain development \
    --project flytesnacks
```

## What you'll see

Each task publishes its own report tab: **raw files · convert · benchmark · train · evaluate · explore**. In a representative cluster run (1000 images):

- **Benchmark (Stage 3):** per-file loading ≈ **17 img/s** vs Lance streaming ≈ **215 img/s** — about **12× faster**, measured against real object storage.
- **Evaluation (Stage 5):** COCO **mAP@.50 ≈ 0.75** on the held-out CPPE-5 test split (all five classes), with **mAP@[.50:.95] ≈ 0.38**.
- **Explore (Stage 6):** predicted vs ground-truth boxes drawn on real validation frames.

The exact numbers vary per run; mAP scales with `max_train_steps`.

## Knobs

| Parameter | Default | What it controls |
|---|---|---|
| `source` | `cppe5` | Dataset source — `cppe5` (real, downloaded) or `synthetic` (offline) |
| `max_images` | 1000 | Max CPPE-5 train images to materialize (`cppe5` only) |
| `val_max_images` | 100 | Max validation images for eval/inference (CPPE-5 test has 29) |
| `resize_max` | 480 | Cap the image long side (px). Faster R-CNN resizes to 800 internally, so 480 keeps files small (bigger benchmark gap) with no accuracy cost |
| `batch_size` | 64 | Lance streaming batch size (Stage 3 benchmark) |
| `epochs` | 30 | Max passes over the stream (`max_train_steps` is the real cap) |
| `train_batch_size` | 8 | Detector training batch size (Stage 4) |
| `max_train_steps` | 1500 | Detector steps — the main accuracy/runtime dial |
| `num_groups` / `scenes_per_group` / `views_per_scene` / `img_size` | 2 / 20 / 20 / 96 | Synthetic-only knobs (used when `source=synthetic`) |

With `source=synthetic`, `num_groups × scenes_per_group × views_per_scene` samples become **3× that many tiny files** (image + mask + label). With `source=cppe5`, each image is 2 tiny files (image + label).

### About the benchmark

There's **no simulation** — Stage 3 measures the real thing. On the cluster, `tiny_dir.path` and `lance_dir.path` are `s3://…` URIs, so the per-file path does one genuine object-store GET per file (via Flyte's IO), and the Lance path opens the dataset URI and streams lazily (`lance.dataset("s3://…")`). Both paths are single-threaded and do the same PNG decode, so the comparison isolates *how the bytes are reached*. A `--local` run reads the same code against local disk, where the per-file penalty is small; the object-store gap is what shows up on the cluster.

The speedup is **resolution-dependent**: it's largest when files are small (per-file connection overhead dominates — ~12× at `resize_max=480`) and shrinks as images grow (bytes dominate more — ~7× at `resize_max=800`). That's the point: the tinier and more numerous the files, the more the per-file tax hurts, and the more Lance helps.

## Where this pattern shows up

The default demo is medical PPE, but the "many tiny files → one Lance dataset → stream" pattern fits any dataset that fans out into small per-sample objects, for example:

- **Frames extracted from video**, often across multiple cameras or views
- **Tiled imagery** — microscopy, satellite/aerial, or gigapixel scans split into patches
- **Heavy augmentation pipelines** that materialize many variants per source image
- **Object-detection / segmentation datasets** with an image (+ mask) + labels per sample

Each maps onto the `group / scene / view` layout and the `image` + `mask` + `bboxes` + `object_classes` columns.

## Notes

- **Streaming from object storage — for real.** On the cluster the benchmark, training, evaluation, and inference all open the Lance dataset by its `s3://…` URI (`lance.dataset(dir.path)`) — nothing is downloaded. Lance reads only the row-groups/columns each batch needs. No credentials to configure: the task's IAM role is picked up automatically by Lance's object store. In your own setup you'd point Lance at your bucket; the code is identical.
- **Shuffling via random access.** Training draws a fresh random row order each epoch and fetches it with `Dataset.take` — real SGD shuffling straight off object storage, and a capability sequential-only formats (tar/WebDataset) can't match.
- **GPU.** `stream_train`, `evaluate`, and `explore_inference` run on `gpu_env` (a T4 by default — see `config.py`). Data stages stay on `cpu_env`. If your cluster labels GPUs differently, change the `gpu="T4:1"` request in `config.py` (a typed request that matches no node stays pending). The code is device-agnostic, so `--local` runs fall back to CPU.
- **The detector is real but bounded.** Stage 4 trains a genuine torchvision Faster R-CNN with proper shuffling and an LR schedule; `max_train_steps` caps runtime. Raise it for higher mAP (accuracy scales with training length). On first run it downloads pretrained MobileNetV3 backbone weights (~20 MB); if that's blocked it falls back to a randomly initialized backbone.
- **CPPE-5 download.** Stage 1 pulls the dataset from the HuggingFace Hub (no auth required; set `HF_TOKEN` to lift rate limits). Use `--source synthetic` for a fully offline run.
- **macOS local runs** may print a harmless segfault (`exit 139`) *after* the pipeline finishes — a native-teardown quirk of `pyarrow`/`torch`/`lance` at interpreter shutdown, not a workflow error. Outputs and reports are already complete. It does not occur on Linux (Union cluster).
