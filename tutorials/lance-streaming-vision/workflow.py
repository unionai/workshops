"""Lance + Flyte: convert once, stream forever.

A pipeline that mirrors a common computer-vision data-loading bottleneck: a
training dataset stored as a huge number of tiny image files.

  1. prepare_tiny_files   — materialize the raw dataset as tiny per-sample files
                            (real CPPE-5 by default, or synthetic scenes offline).
  2. convert_to_lance     — convert that pile of tiny files *once* into a single
                            streaming-optimized Lance dataset.
  3. benchmark_loading    — race the two data paths head-to-head: open-per-file
                            (today) vs stream-from-Lance. Live report + chart.
  4. stream_train         — train a real object detector fed entirely by the
                            Lance stream, reporting a loss curve.
  5. evaluate             — COCO mAP on a held-out split streamed from Lance.
  6. explore_inference    — draw predicted vs ground-truth boxes on val frames.

Run locally:
    uv run flyte run --local workflow.py pipeline

Run on Union:
    uv run flyte run workflow.py pipeline
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time

import flyte
import flyte.report
import flyte.storage as storage
import lance
from flyte.io import DataFrame, Dir

import lance_dataset as lds
from config import cpu_env, gpu_env
from report_helpers import bar_chart, line_chart, stat, wrap_report


# ----------------------------------------------------------------------------
# Stage 1: materialize the raw dataset as a "swarm of tiny files"
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def prepare_tiny_files(
    source: str = "cppe5",
    split: str = "train",
    max_images: int = 500,
    resize_max: int = 480,
    # synthetic-only knobs (used when source="synthetic"):
    num_groups: int = 2,
    scenes_per_group: int = 20,
    views_per_scene: int = 20,
    img_size: int = 96,
    seed: int = 0,
) -> Dir:
    """Materialize a dataset split as tiny per-sample files (image PNG + label
    JSON, plus a mask PNG for synthetic). `source="cppe5"` downloads the real
    CPPE-5 detection dataset; `source="synthetic"` generates scenes offline."""
    root = os.path.join(tempfile.mkdtemp(prefix=f"tiny_{split}_"), "raw-files")
    os.makedirs(root, exist_ok=True)
    is_val = split not in ("train",)

    t0 = time.perf_counter()
    if source == "synthetic":
        # A held-out val split: fewer scenes, a different seed.
        stats = lds.write_tiny_files(
            root,
            num_groups=1 if is_val else num_groups,
            scenes_per_group=max(2, scenes_per_group // 4) if is_val else scenes_per_group,
            views_per_scene=views_per_scene,
            img_size=img_size,
            seed=seed + (1 if is_val else 0),
        )
    else:
        stats = lds.write_cppe5_tiny_files(
            root, split=split, max_images=max_images, resize_max=resize_max
        )
    elapsed = time.perf_counter() - t0

    mb = stats["num_bytes"] / 1e6
    avg_kb = stats["num_bytes"] / max(stats["num_samples"], 1) / 1e3
    html = f"""
    <h2>Stage 1 · Raw dataset ({split}) — a swarm of tiny files</h2>
    <p>Source: <span class="highlight">{stats['detail']}</span>. Each sample is
    written as its own tiny image file plus a small label sidecar — deliberately
    recreating how these datasets sprawl across object storage.</p>
    <div class="stat-grid">
      {stat(f"{stats['source']}", "dataset")}
      {stat(f"{stats['num_samples']:,}", "samples")}
      {stat(f"{stats['num_files']:,}", "tiny files on disk")}
      {stat(f"{stats['num_classes']}", "classes")}
      {stat(f"{avg_kb:.1f} KB", "avg image size")}
      {stat(f"{mb:.1f} MB", "total bytes")}
      {stat(f"{elapsed:.1f}s", "prepare time")}
    </div>
    <div class="note">Each image is ~{avg_kb:.1f} KB, but a training loop pays a
    fresh object-store connection setup <b>per file</b>. With {stats['num_files']:,}
    files, that per-file tax — not the bytes — is the bottleneck.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return await Dir.from_local(root)


# ----------------------------------------------------------------------------
# Stage 2: convert the tiny files into a single Lance dataset (once)
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def convert_to_lance(tiny_dir: Dir) -> DataFrame:
    """One-time conversion: tiny files -> a single streaming-optimized Lance
    dataset. This cost is paid once and amortized over every future run."""
    local_root = await tiny_dir.download()

    # Count every incoming tiny file (image + label sidecars) for the report.
    n_tiny = sum(len(fs) for _dp, _d, fs in os.walk(local_root))

    out_dir = tempfile.mkdtemp(prefix="lance_")
    lance_uri = os.path.join(out_dir, "dataset.lance")

    t0 = time.perf_counter()
    conv = lds.build_lance_dataset(local_root, lance_uri)
    elapsed = time.perf_counter() - t0

    mb = conv["num_bytes"] / 1e6
    html = f"""
    <h2>Stage 2 · Convert once → Lance</h2>
    <p>We fold the entire tree of tiny files into a <span class="highlight">single
    Lance dataset</span>. Image bytes, mask bytes, and structured labels (boxes,
    class ids) now live together in one row — one multimodal, columnar,
    streaming-optimized artifact.</p>
    <div class="stat-grid">
      {stat(f"{n_tiny:,}", "tiny files in")}
      {stat(f"{conv['num_rows']:,}", "Lance rows out")}
      {stat(f"{conv['num_fragments']}", "fragments")}
      {stat(f"{mb:.1f} MB", "dataset size")}
      {stat(f"{elapsed:.1f}s", "convert time")}
    </div>
    <div class="note">This is a one-time job. Every training run from here on
    streams this one dataset instead of reopening {n_tiny:,} objects.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    # Hand the Lance dataset off as a typed DataFrame. Flyte uploads the local
    # .lance directory to blob storage, and the "lance" format tells every
    # downstream task how to open it.
    return DataFrame(uri=lance_uri, format="lance")


# ----------------------------------------------------------------------------
# Stage 3: benchmark the two data paths head-to-head
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def benchmark_loading(
    tiny_dir: Dir, ds: lance.LanceDataset, batch_size: int = 64
) -> dict:
    """Race the two data paths **against real object storage** — nothing is
    downloaded and nothing is simulated:

      A. Per-file (today): walk the tiny-file tree and open every image directly
         from the blob store — one object-store GET per file.
      B. Lance: stream the dataset lazily — Lance reads only the row-groups/columns
         it needs, with a single dataset handle.

    `ds` arrives already open: Flyte decoded the "lance" DataFrame into a
    lance.LanceDataset, resolving the URI and object-store credentials for us. On a
    Union cluster both `tiny_dir.path` and `ds.uri` are `s3://` URIs, so this
    measures true object-store behaviour. Locally they're filesystem paths, so it
    measures local disk (still real, just fast)."""

    backend = "object storage" if storage.is_remote(ds.uri) else "local disk"

    # --- Path A: per-file reads straight from storage (no download) ---
    t0 = time.perf_counter()
    n_a = 0
    async for f in tiny_dir.walk(recursive=True):
        if not f.name.endswith(".png") or f.name.endswith(".mask.png"):
            continue
        async with f.open("rb") as fh:
            image_bytes = bytes(await fh.read())
        lds.decode_image(image_bytes)  # same decode work as the Lance path
        n_a += 1
    time_a = time.perf_counter() - t0

    # --- Path B: stream lazily from the Lance dataset (no download) ---
    t0 = time.perf_counter()
    n_b = 0
    for batch in ds.scanner(columns=["image"], batch_size=batch_size).to_batches():
        for png in batch.column("image").to_pylist():
            lds.decode_image(png)
            n_b += 1
    time_b = time.perf_counter() - t0

    thr_a = n_a / time_a if time_a else 0.0
    thr_b = n_b / time_b if time_b else 0.0
    speedup = (thr_b / thr_a) if thr_a else 0.0

    result = {
        "num_samples": n_a,
        "backend": backend,
        "per_file_seconds": round(time_a, 3),
        "lance_seconds": round(time_b, 3),
        "per_file_img_per_s": round(thr_a, 1),
        "lance_img_per_s": round(thr_b, 1),
        "speedup": round(speedup, 1),
    }

    chart = bar_chart(
        labels=["Per-file (today)", "Lance stream"],
        values=[thr_a, thr_b],
        title="Training data throughput (images / second)",
        unit=" img/s",
        colors=["#adb5bd", "#06d6a0"],
    )
    html = f"""
    <h2>Stage 3 · Loading the same data two ways</h2>
    <p>Both paths load the same <b>{n_a:,} images</b> from <b>{backend}</b> and do
    the same PNG decode — the only difference is <b>how the bytes are fetched</b>:</p>
    <ul>
      <li><b>Per-file (today):</b> open each image on its own — one object-store
      GET per file, {n_a:,} round trips.</li>
      <li><b>Lance stream:</b> open the dataset once and stream it — a few large
      sequential reads, no per-image round trips.</li>
    </ul>
    <div class="chart-container">{chart}</div>
    <div class="stat-grid">
      {stat(f"{time_a:.1f}s", "per-file · read all images")}
      {stat(f"{thr_a:,.0f} img/s", "per-file · rate")}
      {stat(f"{time_b:.1f}s", "Lance · read all images")}
      {stat(f"{thr_b:,.0f} img/s", "Lance · rate")}
      {stat(f"<span class='win'>{speedup:.1f}×</span>", "faster with Lance")}
    </div>
    <div class="note"><b>"Read all images"</b> is the wall-clock to load every
    image once — i.e. what one training epoch's data loading costs; <b>rate</b> is
    images/second. Measured against real <b>{backend}</b>, no downloads or
    simulated latency. A local run reads from disk instead, where the per-file
    penalty is much smaller.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return result


# ----------------------------------------------------------------------------
# Stage 4: a real object detector trained from the Lance stream
# ----------------------------------------------------------------------------
def _build_detector(num_labels: int):
    """Faster R-CNN with a MobileNetV3 backbone — small enough to train quickly
    on a single T4, real enough to produce boxes. Falls back to a randomly
    initialized backbone if pretrained weights can't be downloaded."""
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    try:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone="DEFAULT", num_classes=num_labels
        )
    except Exception:
        model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None, weights_backbone=None, num_classes=num_labels
        )
    return model


def _class_names(source: str, num_labels: int) -> list[str]:
    """Human-readable foreground class names (num_labels includes background)."""
    if source == "cppe5":
        return lds.CPPE5_CLASSES
    return [f"class_{i}" for i in range(num_labels - 1)]


def _load_detector(model_dir_path: str, device):
    """Rebuild the detector from a saved model Dir (weights + config)."""
    import json

    import torch

    with open(os.path.join(model_dir_path, "config.json")) as f:
        config = json.load(f)
    model = _build_detector(num_labels=config["num_labels"])
    model.load_state_dict(
        torch.load(
            os.path.join(model_dir_path, "model.pth"),
            map_location=device,
            weights_only=True,
        )
    )
    model.to(device).eval()
    return model, config


def _draw_boxes(image_tensor, class_names, pred=None, gt=None, score_thr: float = 0.4) -> str:
    """Render an image with ground-truth boxes (blue) and/or predicted boxes
    (green, with a readable class + score label) as a base64 PNG data URI. The
    image is upscaled for legibility and line/label sizes scale with it."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    arr = (image_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB")

    # Upscale small frames so boxes/labels read clearly in the report.
    target = 512
    if max(img.size) < target:
        scale = target / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)))

    draw = ImageDraw.Draw(img)
    scale = img.width / image_tensor.shape[2]  # uniform (aspect ratio preserved)
    lw = max(3, img.width // 160)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(15, img.width // 28))
    except Exception:
        font = ImageFont.load_default()

    if gt is not None:
        for box in gt["boxes"].tolist():
            draw.rectangle([c * scale for c in box], outline=(90, 125, 181), width=lw)
    if pred is not None:
        for box, label, score in zip(
            pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()
        ):
            if score < score_thr:
                continue
            x1, y1, x2, y2 = (c * scale for c in box)
            name = class_names[label - 1] if 0 < label <= len(class_names) else str(label)
            draw.rectangle([x1, y1, x2, y2], outline=(6, 214, 160), width=lw)
            # Label chip with background so text stays readable over any image.
            text = f"{name} {score:.2f}"
            tb = draw.textbbox((0, 0), text, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            ty = max(0, y1 - th - 6)
            draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=(6, 214, 160))
            draw.text((x1 + 4, ty + 2), text, fill=(10, 26, 46), font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@gpu_env.task(report=True)
async def stream_train(
    ds: lance.LanceDataset,
    source: str = "cppe5",
    epochs: int = 30,
    batch_size: int = 8,
    max_train_steps: int = 1500,
    lr: float = 5e-3,
) -> Dir:
    """Train a real object detector (torchvision Faster R-CNN) entirely from the
    Lance stream — reading batches straight from the dataset's object-store URI,
    no download. Runs on GPU (T4). Returns a Dir holding the trained weights +
    config.

    `ds` arrives already open: Flyte decoded the "lance" DataFrame into a
    lance.LanceDataset (URI + credentials resolved), so we just stream from it."""

    import json

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # shuffle=True draws a fresh random row order each epoch via Lance random
    # access — proper SGD shuffling straight off object storage.
    dataset = lds.make_torch_dataset(ds, batch_size=batch_size, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, collate_fn=lds.detection_collate
    )

    num_labels = lds.num_detector_labels(ds)  # fits either dataset
    model = _build_detector(num_labels=num_labels).to(device)
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, momentum=0.9, weight_decay=5e-4,
    )
    # Warmup + cosine anneal in one — keeps Faster R-CNN stable over many steps.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max_train_steps, pct_start=0.1
    )

    loss_curve: list[float] = []
    total_samples = 0
    step = 0
    t_start = time.perf_counter()
    for _epoch in range(epochs):
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

            loss_curve.append(float(loss.item()))
            total_samples += len(images)
            step += 1
            if step >= max_train_steps:
                break
        if step >= max_train_steps:
            break

    total_time = time.perf_counter() - t_start
    overall_thr = total_samples / total_time if total_time else 0.0
    first_loss = loss_curve[0] if loss_curve else 0.0
    final_loss = loss_curve[-1] if loss_curve else 0.0

    # Save weights + config to a Dir so downstream tasks can reload the model.
    save_dir = tempfile.mkdtemp(prefix="model_")
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(
            {
                "num_labels": num_labels,
                "class_names": _class_names(source, num_labels),
                "source": source,
                "steps": step,
                "first_loss": first_loss,
                "final_loss": final_loss,
            },
            f,
        )

    curve_html = line_chart(
        loss_curve, title="Detector training loss", x_label="step", color="#0f3460"
    )
    html = f"""
    <h2>Stage 4 · Real detector trained from the Lance stream</h2>
    <p>A torchvision Faster R-CNN (MobileNetV3 backbone) trains on boxes and
    classes read straight from the Lance dataset — image bytes and labels arrive
    together in each row. Batches are <b>shuffled every epoch via Lance random
    access</b> over object storage (no download, no per-file opens).</p>
    <div class="stat-grid">
      {stat(f"{step}", "training steps")}
      {stat(f"{total_samples:,}", "images to model")}
      {stat(f"{overall_thr:,.1f} img/s", "training throughput")}
      {stat(f"{device.type.upper()}", "device")}
      {stat(f"{first_loss:.2f} → {final_loss:.2f}", "loss (first → last)")}
    </div>
    <div class="chart-container">{curve_html}</div>
    <div class="note">Batches streamed straight from the Lance dataset's
    object-store URI (no download) on <b>{device.type.upper()}</b>. Bounded to
    {max_train_steps} steps so the demo stays quick — lift the cap for higher mAP;
    the data path is unchanged.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return await Dir.from_local(save_dir)


# ----------------------------------------------------------------------------
# Stage 5: evaluate the trained detector on a held-out split (COCO mAP)
# ----------------------------------------------------------------------------
@gpu_env.task(report=True)
async def evaluate(
    model_dir: Dir, val_ds: lance.LanceDataset, batch_size: int = 4
) -> dict:
    """Evaluate the trained detector on the validation Lance dataset with COCO
    mean average precision (torchmetrics), streaming val straight from Lance.

    `val_ds` arrives already open — Flyte decoded the "lance" DataFrame into a
    lance.LanceDataset for us."""

    import torch
    from torchmetrics.detection import MeanAveragePrecision

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = await model_dir.download()  # small weights file
    model, config = _load_detector(model_path, device)
    class_names = config["class_names"]

    dataset = lds.make_torch_dataset(val_ds, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=0, collate_fn=lds.detection_collate
    )

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    n_val = 0
    with torch.no_grad():
        for images, targets in loader:
            preds = model([im.to(device) for im in images])
            preds = [{k: v.cpu() for k, v in p.items()} for p in preds]
            metric.update(preds, targets)
            n_val += len(images)
    res = metric.compute()

    mAP = float(res["map"])
    mAP50 = float(res["map_50"])
    mAP75 = float(res["map_75"])

    # Per-class AP (map_per_class is aligned with the `classes` tensor).
    per_class = {}
    classes = res.get("classes")
    ap_pc = res.get("map_per_class")
    if classes is not None and ap_pc is not None and ap_pc.ndim > 0:
        for cid, ap in zip(classes.tolist(), ap_pc.tolist()):
            name = class_names[cid - 1] if 0 < cid <= len(class_names) else str(cid)
            per_class[name] = max(float(ap), 0.0)

    chart = ""
    if per_class:
        chart = bar_chart(
            labels=list(per_class.keys()),
            values=list(per_class.values()),
            title="Average Precision by class",
            decimals=2,
            colors=["#0f3460"],
        )

    html = f"""
    <h2>Stage 5 · Evaluation on held-out {config['source']} validation</h2>
    <p>COCO mean average precision on {n_val} validation images the detector never
    trained on.</p>
    <div class="stat-grid">
      {stat(f"{mAP:.3f}", "mAP @[.50:.95]")}
      {stat(f"{mAP50:.3f}", "mAP @.50")}
      {stat(f"{mAP75:.3f}", "mAP @.75")}
      {stat(f"{n_val}", "val images")}
    </div>
    <div class="chart-container">{chart}</div>
    <div class="note">mAP is the standard COCO detection metric (0–1, higher is
    better). It scales with how long the detector trains — raise
    <code>max_train_steps</code> for a stronger model; this evaluation stage is
    identical either way.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return {
        "map": round(mAP, 4),
        "map_50": round(mAP50, 4),
        "map_75": round(mAP75, 4),
        "num_val": n_val,
        "per_class_ap": {k: round(v, 4) for k, v in per_class.items()},
    }


# ----------------------------------------------------------------------------
# Stage 6: explore inference — detections drawn on validation frames
# ----------------------------------------------------------------------------
@gpu_env.task(report=True)
async def explore_inference(
    model_dir: Dir,
    val_ds: lance.LanceDataset,
    num_images: int = 8,
    score_thr: float = 0.5,
) -> None:
    """Run the trained detector on a handful of validation frames and render the
    predicted boxes (green) against ground truth (blue). `val_ds` arrives already
    open — Flyte decoded the "lance" DataFrame into a lance.LanceDataset."""

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = await model_dir.download()  # small weights file
    model, config = _load_detector(model_path, device)
    class_names = config["class_names"]

    # Pull the first `num_images` samples off the stream.
    dataset = lds.make_torch_dataset(val_ds, batch_size=num_images)
    images, targets = [], []
    for img, tgt in dataset:
        images.append(img)
        targets.append(tgt)
        if len(images) >= num_images:
            break

    with torch.no_grad():
        preds = model([im.to(device) for im in images])
    preds = [{k: v.cpu() for k, v in p.items()} for p in preds]

    cells = []
    for im, tgt, pred in zip(images, targets, preds):
        uri = _draw_boxes(im, class_names, pred=pred, gt=tgt, score_thr=score_thr)
        n_gt = int(tgt["labels"].shape[0])
        n_pred = int((pred["scores"] > score_thr).sum().item())
        cells.append(
            f'<figure><img src="{uri}"/>'
            f'<figcaption>{n_gt} ground-truth · {n_pred} predicted</figcaption></figure>'
        )

    html = f"""
    <h2>Stage 6 · Explore inference — detections on validation frames</h2>
    <p><span style="color:#5a7db5;font-weight:600">Blue</span> = ground-truth
    boxes streamed from Lance · <span style="color:#06d6a0;font-weight:600">Green</span>
    = the detector's predictions (class + score, threshold {score_thr:g}).</p>
    <div class="gallery">{"".join(cells)}</div>
    <div class="note">The images, boxes, and class labels all came out of the same
    Lance dataset in one streaming scan — no per-file opens anywhere in this path.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
@cpu_env.task(report=True)
async def pipeline(
    source: str = "cppe5",
    max_images: int = 1000,
    val_max_images: int = 100,
    resize_max: int = 480,
    batch_size: int = 64,
    epochs: int = 30,
    train_batch_size: int = 8,
    max_train_steps: int = 1500,
    # synthetic-only knobs (used when source="synthetic"):
    num_groups: int = 2,
    scenes_per_group: int = 20,
    views_per_scene: int = 20,
    img_size: int = 96,
) -> dict:
    """End-to-end: prepare → convert → benchmark → train → evaluate → explore."""
    # --- Train split: prepare → convert → benchmark → train ---
    tiny_dir = await prepare_tiny_files(
        source=source,
        split="train",
        max_images=max_images,
        resize_max=resize_max,
        num_groups=num_groups,
        scenes_per_group=scenes_per_group,
        views_per_scene=views_per_scene,
        img_size=img_size,
    )
    lance_df = await convert_to_lance(tiny_dir)
    bench = await benchmark_loading(tiny_dir, lance_df, batch_size=batch_size)
    model_dir = await stream_train(
        lance_df,
        source=source,
        epochs=epochs,
        batch_size=train_batch_size,
        max_train_steps=max_train_steps,
    )

    # --- Validation split: prepare → convert → evaluate → explore ---
    val_tiny_dir = await prepare_tiny_files(
        source=source,
        split="test",
        max_images=val_max_images,
        resize_max=resize_max,
        num_groups=num_groups,
        scenes_per_group=scenes_per_group,
        views_per_scene=views_per_scene,
        img_size=img_size,
    )
    val_lance_df = await convert_to_lance(val_tiny_dir)
    metrics = await evaluate(model_dir, val_lance_df)
    await explore_inference(model_dir, val_lance_df)

    html = f"""
    <h2>Lance + Flyte · convert once, stream forever</h2>
    <p>The dataset lived as {bench['num_samples']:,} tiny files. After a one-time
    conversion to Lance, streaming the training data ran
    <span class="win">{bench['speedup']:.1f}× faster</span> than opening each file,
    then fed a real detector through train → evaluate → inference.</p>
    <div class="stat-grid">
      {stat(f"{bench['num_samples']:,}", "train samples")}
      {stat(f"<span class='win'>{bench['speedup']:.1f}×</span>", "loading speedup")}
      {stat(f"{bench['lance_img_per_s']:,.0f}", "Lance img/s")}
      {stat(f"{metrics['map']:.3f}", "mAP @[.50:.95]")}
      {stat(f"{metrics['map_50']:.3f}", "mAP @.50")}
      {stat(f"{metrics['num_val']}", "val images")}
    </div>
    <div class="note">Open each task's report tab for the per-stage breakdown:
    raw files · convert · benchmark · train · evaluate · explore. The benchmark
    reads real object storage (no simulation); lift <code>--max_train_steps</code>
    for higher mAP.</div>
    """
    await flyte.report.replace.aio(wrap_report(html), do_flush=True)

    return {"benchmark": bench, "metrics": metrics}


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(pipeline)
    print(run.url)
