"""
RT-DETRv2 Object Detection — Fine-tune on a custom COCO dataset.

Pipeline: download a COCO-format dataset from HuggingFace, fine-tune RT-DETRv2
for object detection, evaluate with COCO mAP, and render an inference demo
with bounding boxes drawn on held-out images.

Usage:
    # Default (RT-DETRv2-R18 on Union swag stickers)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2

    # Remote
    flyte run workflow.py pipeline --epochs 30

    # Swap model
    flyte run workflow.py pipeline --model_name "PekingU/rtdetr_v2_r50vd"
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import shutil
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Report styling — shared CSS for all task reports
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f3460; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  .report tr:nth-child(even) { background: #f8f9fa; }
  .report .highlight { color: #0f3460; font-weight: 700; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .img-pair { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
  .report .img-pair > div { flex: 1; min-width: 300px; }
  .report .img-pair img { width: 100%; border-radius: 6px; border: 1px solid #dee2e6; }
  .report .img-pair .gt-label { color: #5a7db5; font-weight: 600; }
  .report .img-pair .pred-label { color: #06d6a0; font-weight: 600; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d4edda; color: #155724; }
  .report .badge-info { background: #d1ecf1; color: #0c5460; }
  .report .chart-container { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0; }
</style>
"""


def _wrap_report(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# SVG chart helpers — lightweight charts without matplotlib
# ------------------------------------------------------------------

def _make_line_chart(
    data: list[dict],
    x_key: str,
    y_keys: list[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    y_max_cap: float | None = None,
    x_range_override: tuple[float, float] | None = None,
    y_display_names: dict[str, str] | None = None,
) -> str:
    """Generate an SVG line chart from a list of dicts.

    Args:
        data: List of dicts, each with x_key and y_keys values.
        x_key: Key for x-axis values.
        y_keys: Keys for y-axis series to plot.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        colors: Colors for each series (defaults to a built-in palette).
        width: SVG width in pixels.
        height: SVG height in pixels.
        y_max_cap: If set, cap the y-axis at this value (e.g. 1.0 for mAP).
        x_range_override: If set, force the x-axis to this (min, max) range.

    Returns:
        SVG string.
    """

    default_colors = ["#5a7db5", "#0f3460", "#06d6a0", "#ffc107", "#6c757d"]
    colors = colors or default_colors

    # Chart area margins
    ml, mr, mt, mb = 60, 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    x_vals = [d[x_key] for d in data] if data else []
    if x_range_override:
        x_min, x_max = x_range_override
    elif x_vals:
        x_min, x_max = min(x_vals), max(x_vals)
    else:
        x_min, x_max = 0, 1
    x_range = x_max - x_min or 1

    # Compute y range across all series
    all_y = []
    for key in y_keys:
        all_y.extend(d[key] for d in data if key in d)
    y_min = min(all_y) if all_y else 0
    y_max = max(all_y) if all_y else 1
    y_pad = (y_max - y_min) * 0.1 or 0.1
    y_min_plot = max(0, y_min - y_pad)
    y_max_plot = y_max + y_pad
    if y_max_cap is not None:
        y_max_plot = min(y_max_plot, y_max_cap)
    y_range = y_max_plot - y_min_plot or 1

    def sx(v):
        return ml + (v - x_min) / x_range * cw

    def sy(v):
        return mt + ch - (v - y_min_plot) / y_range * ch

    # Build SVG
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        # Background
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid lines (5 horizontal)
    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    # Axes
    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#adb5bd" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#adb5bd" stroke-width="1.5"/>'
    )

    # X-axis ticks
    if x_vals:
        n_x_ticks = min(len(data), 10)
        step = max(1, len(data) // n_x_ticks)
        for i in range(0, len(data), step):
            px = sx(x_vals[i])
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_vals[i]:.0f}</text>'
            )
    else:
        # Empty chart — generate evenly spaced ticks from x range
        for i in range(6):
            x_tick = x_min + x_range * i / 5
            px = sx(x_tick)
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_tick:.0f}</text>'
            )

    # Plot each series
    if not data:
        # Empty chart placeholder
        lines.append(
            f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="13" fill="#adb5bd" font-style="italic">Waiting for data...</text>'
        )
    for si, key in enumerate(y_keys):
        color = colors[si % len(colors)]
        points = [(sx(d[x_key]), sy(d[key])) for d in data if key in d]
        if not points:
            continue
        # Draw line if we have 2+ points (dash odd series for visibility)
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
            for px, py in points[1:]:
                path_d += f" L {px:.1f},{py:.1f}"
            dash = ' stroke-dasharray="6,3"' if si % 2 == 1 else ""
            lines.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"{dash}/>'
            )
        # Always show dots for sparse data (including single points)
        if len(points) <= 30:
            for px, py in points:
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>'
                )

    # Title
    if title:
        lines.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    # Axis labels
    if x_label:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d">{x_label}</text>'
        )
    if y_label:
        lines.append(
            f'<text x="14" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d" '
            f'transform="rotate(-90, 14, {mt + ch / 2})">{y_label}</text>'
        )

    # Legend
    names = y_display_names or {}
    if len(y_keys) > 1:
        lx = ml + 10
        for si, key in enumerate(y_keys):
            color = colors[si % len(colors)]
            ly = mt + 14 + si * 18
            lines.append(
                f'<rect x="{lx}" y="{ly - 6}" width="12" height="12" '
                f'rx="2" fill="{color}"/>'
            )
            label = names.get(key, key)
            lines.append(
                f'<text x="{lx + 16}" y="{ly + 4}" font-size="11" '
                f'fill="#1a1a2e">{label}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def _make_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    y_max_cap: float | None = None,
) -> str:
    """Generate an SVG grouped bar chart.

    Args:
        labels: Category labels for x-axis.
        series: Dict mapping series name to list of values (same length as labels).
        title: Chart title.
        colors: Colors for each series.
        width: SVG width.
        height: SVG height.
        y_max_cap: If set, cap the y-axis at this value (e.g. 1.0 for mAP).

    Returns:
        SVG string.
    """
    if not labels:
        return ""

    default_colors = ["#adb5bd", "#0f3460", "#06d6a0", "#5a7db5"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 60
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for vals in series.values() for v in vals]
    y_max = max(all_vals) if all_vals else 1
    y_max_plot = y_max * 1.15 or 1
    if y_max_cap is not None:
        y_max_plot = min(y_max_plot, y_max_cap) or y_max_cap

    n_groups = len(labels)
    n_series = len(series)
    group_width = cw / n_groups
    bar_width = group_width * 0.7 / max(n_series, 1)
    gap = group_width * 0.15

    def sy(v):
        return mt + ch - (v / y_max_plot) * ch

    lines_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid lines
    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        lines_svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        lines_svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    # Bars
    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            lines_svg.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" '
                f'height="{bh:.1f}" fill="{color}" rx="2"/>'
            )
            # Value label on top of bar
            lines_svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#1a1a2e">'
                f'{val:.3f}</text>'
            )
        # Group label
        lines_svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" '
            f'text-anchor="middle" font-size="11" fill="#6c757d">{label}</text>'
        )

    # Title
    if title:
        lines_svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    # Legend
    lx = ml + cw - len(series) * 100
    for si, name in enumerate(series):
        color = colors[si % len(colors)]
        lines_svg.append(
            f'<rect x="{lx + si * 100}" y="{mt + ch + 35}" width="12" '
            f'height="12" rx="2" fill="{color}"/>'
        )
        lines_svg.append(
            f'<text x="{lx + si * 100 + 16}" y="{mt + ch + 46}" font-size="11" '
            f'fill="#1a1a2e">{name}</text>'
        )

    lines_svg.append("</svg>")
    return "\n".join(lines_svg)


# ------------------------------------------------------------------
# Task 1: Prepare dataset — download COCO JSON + images, split train/val
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_repo: str = "sagecodes/union_swag_coco",
    annotations_path: str = "swag/train.json",
    images_subdir: str = "swag/images",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> flyte.io.Dir:
    """Download a COCO-format dataset from HF and split into train/val."""
    from huggingface_hub import snapshot_download

    log.info(f"Downloading dataset: {dataset_repo}")
    local_repo = snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
    )

    ann_file = os.path.join(local_repo, annotations_path)
    img_root = os.path.join(local_repo, images_subdir)

    with open(ann_file) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    log.info(
        f"Loaded {len(images)} images, {len(annotations)} annotations, "
        f"{len(categories)} categories"
    )
    log.info(f"Raw category ids: {sorted({c['id'] for c in categories})}")
    log.info(
        f"Raw annotation category_ids (unique): "
        f"{sorted({a['category_id'] for a in annotations})}"
    )

    # Remap category ids to contiguous 0..N-1 — required because HF object
    # detection models size their classifier head to len(id2label) and treat
    # class labels as direct indices into that head. Any gap or 1-indexed id
    # causes an IndexKernel OOB inside the focal-loss scatter.
    #
    # Build the remap from the UNION of ids declared in `categories` and ids
    # actually used in `annotations` — some datasets have orphaned annotations
    # referencing categories that aren't declared (this one does).
    declared_ids = {c["id"] for c in categories}
    used_ids = {a["category_id"] for a in annotations}
    orphans = used_ids - declared_ids
    if orphans:
        log.warning(
            f"Annotations reference undeclared category ids {sorted(orphans)} — "
            f"adding stub categories."
        )

    all_cat_ids = sorted(declared_ids | used_ids)
    id_remap = {old: new for new, old in enumerate(all_cat_ids)}
    existing_names = {c["id"]: c["name"] for c in categories}
    categories = [
        {"id": id_remap[old], "name": existing_names.get(old, f"category_{old}")}
        for old in all_cat_ids
    ]
    annotations = [
        {**a, "category_id": id_remap[a["category_id"]]} for a in annotations
    ]
    log.info(f"Remapped category ids: {id_remap}")
    log.info(f"Final categories: {categories}")

    # Split by image id
    rng = random.Random(seed)
    img_ids = [im["id"] for im in images]
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * val_fraction))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    def filter_coco(keep_ids: set) -> dict:
        return {
            "info": coco.get("info", {}),
            "categories": categories,
            "images": [im for im in images if im["id"] in keep_ids],
            "annotations": [a for a in annotations if a["image_id"] in keep_ids],
        }

    train_coco = filter_coco(train_ids)
    val_coco = filter_coco(val_ids)

    log.info(
        f"Split: {len(train_coco['images'])} train / {len(val_coco['images'])} val images"
    )

    # Pack output dir: images/ + train.json + val.json
    out_dir = tempfile.mkdtemp(prefix="coco_split_")
    out_img = os.path.join(out_dir, "images")
    shutil.copytree(img_root, out_img)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(train_coco, f)
    with open(os.path.join(out_dir, "val.json"), "w") as f:
        json.dump(val_coco, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Helpers — torch Dataset wrapping COCO JSON
# ------------------------------------------------------------------

def _build_torch_dataset(coco_path: str, images_root: str, augment: bool):
    """Build a torch Dataset that yields {image, target} for the HF image processor."""
    import albumentations as A
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset

    with open(coco_path) as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    image_ids = list(images_by_id.keys())

    # NOTE: we deliberately don't resize here — the HF image processor handles
    # resize+pad. Augmentation only.
    if augment:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
                A.Rotate(limit=15, border_mode=0, p=0.4),
                A.RandomScale(scale_limit=0.2, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.GaussNoise(p=0.2),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                min_area=4,
                min_visibility=0.1,
                clip=True,
            ),
        )
    else:
        transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
        )

    class CocoDataset(Dataset):
        def __len__(self) -> int:
            return len(image_ids)

        def __getitem__(self, idx: int):
            img_id = image_ids[idx]
            meta = images_by_id[img_id]
            img_path = os.path.join(images_root, os.path.basename(meta["file_name"]))
            if not os.path.exists(img_path):
                img_path = os.path.join(images_root, meta["file_name"])
            image = np.array(Image.open(img_path).convert("RGB"))

            anns = anns_by_image.get(img_id, [])
            bboxes = [a["bbox"] for a in anns]
            categories = [a["category_id"] for a in anns]

            out = transform(image=image, bboxes=bboxes, category=categories)
            image_t = out["image"]
            bboxes_t = out["bboxes"]
            categories_t = out["category"]

            target_anns = []
            for bb, cat in zip(bboxes_t, categories_t):
                x, y, w, h = bb
                target_anns.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

            return {
                "image": image_t,
                "target": {"image_id": img_id, "annotations": target_anns},
            }

    return CocoDataset(), coco["categories"]


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 30,
    lr: float = 5e-5,
    batch_size: int = 4,
    weight_decay: float = 1e-4,
    eval_every_n_epochs: int | None = None,
) -> flyte.io.Dir:
    """Fine-tune RT-DETR (or any HuggingFace object-detection model) on COCO data."""
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    log.info(f"Training: model={model_name}")
    await flyte.report.replace.aio(_wrap_report(
        f"<h2>Loading model...</h2><p>{model_name}</p>"
        f"<p>Preparing dataset and initializing weights...</p>"
    ), do_flush=True)

    # -- Load data --
    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    train_json = os.path.join(data_path, "train.json")

    with open(train_json) as f:
        categories = json.load(f)["categories"]
    id2label = {c["id"]: c["name"] for c in categories}
    label2id = {v: k for k, v in id2label.items()}

    train_ds, _ = _build_torch_dataset(train_json, images_root, augment=True)
    log.info(f"Train examples: {len(train_ds)} | Categories: {id2label}")

    # -- Optionally load val set for periodic mAP evaluation --
    val_json = os.path.join(data_path, "val.json")
    val_images = None
    val_targets = None
    if eval_every_n_epochs and os.path.exists(val_json):
        import torch as _torch
        from PIL import Image

        with open(val_json) as f:
            val_coco = json.load(f)
        images_by_id = {im["id"]: im for im in val_coco["images"]}
        anns_by_image: dict[int, list] = {}
        for a in val_coco["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a)
        val_images = []
        val_targets = []
        for img_id, meta in images_by_id.items():
            path = os.path.join(images_root, os.path.basename(meta["file_name"]))
            if not os.path.exists(path):
                path = os.path.join(images_root, meta["file_name"])
            val_images.append(Image.open(path).convert("RGB"))
            boxes_xyxy = []
            labels = []
            for a in anns_by_image.get(img_id, []):
                x, y, w, h = a["bbox"]
                boxes_xyxy.append([x, y, x + w, y + h])
                labels.append(a["category_id"])
            val_targets.append({
                "boxes": _torch.tensor(boxes_xyxy, dtype=_torch.float32).reshape(-1, 4),
                "labels": _torch.tensor(labels, dtype=_torch.long),
            })
        log.info(f"Val examples for periodic eval: {len(val_images)}")

    # -- Processor + model --
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.1f}%)"
    )

    # -- Collator — runs the image processor on each batch --
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        targets = [b["target"] for b in batch]
        enc = processor(images=images, annotations=targets, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"], "labels": enc["labels"]}

    # -- Sanity check: peek at one batch and verify class_labels fit --
    sample = collate_fn([train_ds[i] for i in range(min(2, len(train_ds)))])
    all_labels = []
    for lbl in sample["labels"]:
        all_labels.extend(lbl["class_labels"].tolist())
    log.info(
        f"Sanity check — class_labels in first batch: {sorted(set(all_labels))} | "
        f"model num_labels: {model.config.num_labels} | "
        f"id2label: {model.config.id2label}"
    )
    if all_labels and max(all_labels) >= model.config.num_labels:
        raise ValueError(
            f"class_label {max(all_labels)} out of range for num_labels="
            f"{model.config.num_labels}. Check category id remapping in prepare_data."
        )

    # -- Collect training metrics and update the report chart live.
    # trainer.train() runs in a background thread (via asyncio.to_thread),
    # so the asyncio event loop stays free. We use run_coroutine_threadsafe
    # to push report updates from the callback thread onto that loop.
    training_log: list[dict] = []
    eval_log: list[dict] = []  # periodic mAP checkpoints (epoch, map, map_50)
    loop = asyncio.get_running_loop()

    cat_badges = " ".join(
        f'<span class="badge badge-info">{name}</span>'
        for name in id2label.values()
    )

    def _build_training_report(max_steps: int) -> str:
        """Build the live training report HTML from current training_log."""
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Examples</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{total_params:,}</div><div class="label">Total Params</div></div>
          <div class="stat"><div class="value">{trainable_params / total_params * 100:.1f}%</div><div class="label">Trainable</div></div>
        </div>
        <p>Categories: {cat_badges}</p>
        """

        charts_html = ""
        if training_log:
            current = training_log[-1]
            progress_pct = current["step"] / max_steps * 100 if max_steps else 0
            charts_html += f"""
            <div class="card">
              <b>Step {current['step']}/{max_steps}</b>
              ({progress_pct:.0f}%) |
              Epoch {current['epoch']:.2f}/{epochs} |
              Loss: <span class="highlight">{current['loss']:.4f}</span>
              <div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#0f3460;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_chart = _make_line_chart(
                data=training_log,
                x_key="epoch",
                y_keys=["loss"],
                title="Training Loss",
                x_label="Epoch",
                y_label="Loss",
                colors=["#5a7db5"],
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

            if eval_every_n_epochs:
                # Match x-axis to the loss chart: start at 0, end at current epoch
                current_max_epoch = current["epoch"]
                map_chart = _make_line_chart(
                    data=eval_log,
                    x_key="epoch",
                    y_keys=["map", "map_50"],
                    title="Validation mAP (periodic)",
                    x_label="Epoch",
                    y_label="mAP",
                    colors=["#0f3460", "#06d6a0"],
                    y_max_cap=1.0,
                    y_display_names={"map": "mAP (0.50:0.95)", "map_50": "mAP@50"},
                    x_range_override=(0, current_max_epoch),
                )
                charts_html += f'<div class="chart-container">{map_chart}</div>'

            if "lr" in training_log[0]:
                lr_chart = _make_line_chart(
                    data=training_log,
                    x_key="epoch",
                    y_keys=["lr"],
                    title="Learning Rate Schedule",
                    x_label="Epoch",
                    y_label="LR",
                    colors=["#0f3460"],
                )
                charts_html += f'<div class="chart-container">{lr_chart}</div>'

        return _wrap_report(stats_html + charts_html)

    class MetricsCallback(TrainerCallback):
        def __init__(self):
            self._last_eval_epoch = 0

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            entry = {
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 2),
                "loss": round(logs["loss"], 4),
            }
            if "learning_rate" in logs:
                entry["lr"] = logs["learning_rate"]
            if "grad_norm" in logs:
                entry["grad_norm"] = round(float(logs["grad_norm"]), 4)
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f} "
                f"loss={entry['loss']:.4f}"
            )

            # Push a live report update onto the asyncio event loop.
            # do_flush=True dispatches the update to the UI immediately.
            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(
                    _build_training_report(state.max_steps),
                    do_flush=True,
                ),
                loop,
            )

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if not eval_every_n_epochs or val_images is None:
                return
            current_epoch = round(state.epoch)
            if current_epoch % eval_every_n_epochs != 0:
                return
            if current_epoch == self._last_eval_epoch:
                return
            self._last_eval_epoch = current_epoch

            log.info(f"Running periodic mAP eval at epoch {current_epoch}...")
            from torchmetrics.detection.mean_ap import MeanAveragePrecision

            device = next(model.parameters()).device
            # _run_inference sets model.eval(); restore train mode after.
            preds = _run_inference(model, processor, val_images, device, threshold=0.3)
            model.train()

            formatted = [
                {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
                for p in preds
            ]
            metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            metric.update(formatted, val_targets)
            result = metric.compute()
            map_val = round(result["map"].item(), 4)
            map_50 = round(result["map_50"].item(), 4)

            eval_log.append({
                "epoch": current_epoch,
                "map": map_val,
                "map_50": map_50,
            })
            log.info(f"Epoch {current_epoch} — mAP: {map_val:.4f}, mAP@50: {map_50:.4f}")

            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(
                    _build_training_report(state.max_steps),
                    do_flush=True,
                ),
                loop,
            )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=5,
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_ratio=0.1,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        callbacks=[MetricsCallback()],
    )

    log.info("Starting training...")
    # Run the sync HF training loop in a thread so the asyncio event loop
    # stays free for Flyte's syncify bridge.
    await asyncio.to_thread(trainer.train)
    log.info("Training complete.")

    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    # -- Build final training report --
    stats_html = f"""
    <h2>Training Complete</h2>
    <h3>{model_name}</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Examples</div></div>
      <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="value">{lr}</div><div class="label">Learning Rate</div></div>
      <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
      <div class="stat"><div class="value">{total_params:,}</div><div class="label">Total Params</div></div>
      <div class="stat"><div class="value">{trainable_params / total_params * 100:.1f}%</div><div class="label">Trainable</div></div>
    </div>
    <p>Categories: {cat_badges}</p>
    """

    charts_html = ""
    if training_log:
        final_loss = training_log[-1]["loss"]
        min_loss = min(d["loss"] for d in training_log)
        initial_loss = training_log[0]["loss"]
        total_steps = training_log[-1]["step"]

        charts_html += f"""
        <div class="card">
          <b>Training Summary:</b>
          Initial loss: {initial_loss:.4f} |
          Final loss: <span class="highlight">{final_loss:.4f}</span> |
          Min loss: {min_loss:.4f} |
          Total steps: {total_steps}
        </div>
        """

        epoch_range = (0, epochs)

        loss_chart = _make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["loss"],
            title="Training Loss",
            x_label="Epoch",
            y_label="Loss",
            colors=["#5a7db5"],
            x_range_override=epoch_range,
        )
        charts_html += f'<div class="chart-container">{loss_chart}</div>'

    if eval_log:
        map_chart = _make_line_chart(
            data=eval_log,
            x_key="epoch",
            y_keys=["map", "map_50"],
            title="Validation mAP (periodic)",
            x_label="Epoch",
            y_label="mAP",
            colors=["#0f3460", "#06d6a0"],
            y_max_cap=1.0,
            x_range_override=(0, epochs),
            y_display_names={"map": "mAP (0.50:0.95)", "map_50": "mAP@50"},
        )
        charts_html += f'<div class="chart-container">{map_chart}</div>'

    if training_log and "lr" in training_log[0]:
        lr_chart = _make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["lr"],
            title="Learning Rate Schedule",
            x_label="Epoch",
            y_label="LR",
            colors=["#0f3460"],
            x_range_override=(0, epochs),
        )
        charts_html += f'<div class="chart-container">{lr_chart}</div>'

    await flyte.report.replace.aio(_wrap_report(stats_html + charts_html), do_flush=True)

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def _run_inference(model, processor, images, device, threshold: float = 0.3):
    """Run object detection on a list of PIL images. Returns list of dicts."""
    import torch

    results = []
    model.eval()
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_size = torch.tensor([img.size[::-1]], device=device)  # (h, w)
        post = processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]
        results.append(
            {
                "scores": post["scores"].cpu(),
                "labels": post["labels"].cpu(),
                "boxes": post["boxes"].cpu(),  # xyxy in original image coords
            }
        )
    return results


def _draw_boxes(image, boxes, labels, scores, id2label, color: str = "lime"):
    """Draw bounding boxes on a PIL image. Returns a new PIL image."""
    from PIL import ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x0, y0, x1, y1 = box
        width = max(2, img.width // 400)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        name = id2label.get(int(label), str(int(label)))
        caption = f"{name} {score:.2f}"
        text_bg = draw.textbbox((x0, y0), caption, font=font)
        draw.rectangle(text_bg, fill=color)
        draw.text((x0, y0), caption, fill="black", font=font)
    return img


def _img_to_data_uri(img, max_dim: int = 800) -> str:
    """PIL image → base64 data URI, downscaled for the report."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------
# Task 3: Evaluate — COCO mAP on fine-tuned model
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    threshold: float = 0.5,
) -> str:
    """Compute COCO mAP for the fine-tuned model on the val split."""
    import torch
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Evaluation</h2><p>Loading val split and scoring model...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    val_json = os.path.join(data_path, "val.json")

    with open(val_json) as f:
        val_coco = json.load(f)

    images_by_id = {im["id"]: im for im in val_coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in val_coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    pil_images = []
    targets = []
    for img_id, meta in images_by_id.items():
        path = os.path.join(images_root, os.path.basename(meta["file_name"]))
        if not os.path.exists(path):
            path = os.path.join(images_root, meta["file_name"])
        pil_images.append(Image.open(path).convert("RGB"))
        boxes_xyxy = []
        labels = []
        for a in anns_by_image.get(img_id, []):
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        targets.append(
            {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ft_path = await finetuned_dir.download()
    log.info(f"Scoring fine-tuned model: {ft_path}")
    processor = AutoImageProcessor.from_pretrained(ft_path)
    model = AutoModelForObjectDetection.from_pretrained(ft_path).to(device)
    preds = _run_inference(model, processor, pil_images, device, threshold=threshold)

    formatted_preds = [
        {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
        for p in preds
    ]

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    metric.update(formatted_preds, targets)

    def to_python(v):
        if hasattr(v, "numel"):
            return v.item() if v.numel() == 1 else v.tolist()
        return v

    ft_metrics = {k: to_python(v) for k, v in metric.compute().items()}
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Fine-tuned mAP: {ft_metrics.get('map', 0):.3f}")

    metric_keys = ["map", "map_50", "map_75", "mar_10"]
    metric_display = {
        "map": "mAP",
        "map_50": "mAP@50",
        "map_75": "mAP@75",
        "mar_10": "mAR@10",
    }

    rows = []
    for key in metric_keys:
        ft_val = ft_metrics.get(key, 0)
        rows.append(
            f"<tr><td><b>{metric_display.get(key, key)}</b></td>"
            f"<td class='highlight'>{ft_val:.3f}</td></tr>"
        )
    table = (
        "<table><tr><th>Metric</th><th>Score</th></tr>"
        + "".join(rows)
        + "</table>"
    )

    bar_chart = _make_bar_chart(
        labels=[metric_display.get(k, k) for k in metric_keys],
        series={"Fine-tuned": [ft_metrics.get(k, 0) for k in metric_keys]},
        title="COCO Evaluation Metrics",
        colors=["#0f3460"],
        y_max_cap=1.0,
    )

    ft_map = ft_metrics.get("map", 0)
    ft_map50 = ft_metrics.get("map_50", 0)

    eval_html = f"""
    <h2>Evaluation — COCO mAP</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(pil_images)}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value">{threshold}</div><div class="label">Threshold</div></div>
      <div class="stat"><div class="value highlight">{ft_map:.3f}</div><div class="label">mAP</div></div>
      <div class="stat"><div class="value highlight">{ft_map50:.3f}</div><div class="label">mAP@50</div></div>
    </div>
    <div class="chart-container">{bar_chart}</div>
    {table}
    <div class="note">
      <b>mAP</b> (mean Average Precision) measures how accurately the model
      detects objects — balancing whether predictions are correct (precision)
      and whether all objects are found (recall). The @50 and @75 variants
      require IoU overlaps of 50% and 75% between predicted and ground-truth
      boxes. <b>mAR</b> (mean Average Recall) measures how many ground-truth
      objects the model finds, with @1 and @10 limiting detections to 1 or 10
      per image.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(eval_html), do_flush=True)

    return json.dumps(
        {
            "finetuned": {k: round(v, 4) for k, v in ft_metrics.items() if isinstance(v, (int, float))},
            "num_val_images": len(pil_images),
        }
    )


# ------------------------------------------------------------------
# Task 4: Inference demo — render bboxes on val images
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def inference_demo(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    threshold: float = 0.5,
    max_images: int = 8,
    metrics_json: str = "{}",
) -> str:
    """Run the fine-tuned model on val images, render bboxes, embed in the report."""
    import torch
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    val_json = os.path.join(data_path, "val.json")

    with open(val_json) as f:
        val_coco = json.load(f)

    id2label = {c["id"]: c["name"] for c in val_coco["categories"]}
    metas = val_coco["images"][:max_images]
    anns_by_image: dict[int, list] = {}
    for a in val_coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    pil_images = []
    gt_per_image = []
    for meta in metas:
        path = os.path.join(images_root, os.path.basename(meta["file_name"]))
        if not os.path.exists(path):
            path = os.path.join(images_root, meta["file_name"])
        pil_images.append(Image.open(path).convert("RGB"))

        boxes_xyxy = []
        labels = []
        for a in anns_by_image.get(meta["id"], []):
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        gt_per_image.append(
            {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
                "scores": torch.ones(len(labels)),
            }
        )

    ft_path = await finetuned_dir.download()
    processor = AutoImageProcessor.from_pretrained(ft_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForObjectDetection.from_pretrained(ft_path).to(device)

    preds = _run_inference(model, processor, pil_images, device, threshold=threshold)

    html_blocks = []
    total_gt = 0
    total_pred = 0
    for i, (img, pred, gt) in enumerate(zip(pil_images, preds, gt_per_image)):
        n_gt = len(gt["labels"])
        n_pred = len(pred["labels"])
        total_gt += n_gt
        total_pred += n_pred

        # Per-image mAP
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(
            [{"boxes": pred["boxes"], "scores": pred["scores"], "labels": pred["labels"]}],
            [{"boxes": gt["boxes"], "labels": gt["labels"]}],
        )
        img_metrics = metric.compute()
        img_map = img_metrics["map"].item()
        img_map_badge = (
            f'<span class="badge badge-success">mAP {img_map:.2f}</span>'
            if img_map >= 0.5
            else f'<span class="badge badge-info">mAP {img_map:.2f}</span>'
        )

        pred_img = _draw_boxes(
            img, pred["boxes"], pred["labels"], pred["scores"],
            id2label, color="lime",
        )
        gt_img = _draw_boxes(
            img, gt["boxes"], gt["labels"], gt["scores"],
            id2label, color="dodgerblue",
        )
        html_blocks.append(f"""
        <div class="card">
          <b>Image {i + 1}</b> {img_map_badge}
          <div class="img-pair">
            <div>
              <p><span class="gt-label">Ground Truth</span>
                 <span class="badge badge-info">{n_gt} boxes</span></p>
              <img src="{_img_to_data_uri(gt_img)}" />
            </div>
            <div>
              <p><span class="pred-label">Predictions</span>
                 <span class="badge badge-success">{n_pred} boxes</span>
                 (threshold={threshold})</p>
              <img src="{_img_to_data_uri(pred_img)}" />
            </div>
          </div>
        </div>""")

    # Parse metrics if provided (from evaluate task)
    metrics = json.loads(metrics_json)
    ft_metrics = metrics.get("finetuned", {})
    ft_map = ft_metrics.get("map", None)
    ft_map50 = ft_metrics.get("map_50", None)

    metrics_stats = ""
    if ft_map is not None:
        metrics_stats = f"""
        <div class="stat"><div class="value highlight">{ft_map:.3f}</div><div class="label">mAP</div></div>
        <div class="stat"><div class="value highlight">{ft_map50:.3f}</div><div class="label">mAP@50</div></div>
        """

    demo_html = f"""
    <h2>Inference Demo</h2>
    <h3>Fine-tuned RT-DETR on validation images</h3>
    <div class="stat-grid">
      {metrics_stats}
      <div class="stat"><div class="value">{len(pil_images)}</div><div class="label">Images Shown</div></div>
      <div class="stat"><div class="value">{total_gt}</div><div class="label">Ground Truth Boxes</div></div>
      <div class="stat"><div class="value">{total_pred}</div><div class="label">Predicted Boxes</div></div>
      <div class="stat"><div class="value">{threshold}</div><div class="label">Confidence Threshold</div></div>
    </div>
    <p><span class="gt-label">Blue = ground truth</span> |
       <span class="pred-label">Green = predictions</span></p>
    {"".join(html_blocks)}
    """

    await flyte.report.replace.aio(_wrap_report(demo_html), do_flush=True)

    return json.dumps(
        {
            "num_images": len(pil_images),
            "predictions_per_image": [len(p["labels"]) for p in preds],
        }
    )


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "PekingU/rtdetr_v2_r18vd",
    dataset_repo: str = "sagecodes/union_swag_coco",
    annotations_path: str = "swag/train.json",
    images_subdir: str = "swag/images",
    epochs: int = 30,
    lr: float = 5e-5,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    threshold: float = 0.5,
    demo_images: int = 8,
    eval_every_n_epochs: int | None = None,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end RT-DETRv2 fine-tuning pipeline.

    Returns the fine-tuned model directory and a JSON summary.

    1. Download COCO dataset from HuggingFace and split train/val
    2. Fine-tune RT-DETRv2 on the train split
    3. Evaluate: COCO mAP comparison (base vs fine-tuned)
    4. Inference demo: render bounding boxes on val images
    """
    log.info(f"Pipeline: {model_name} | dataset={dataset_repo}")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Preparing Data", "Fine-tuning", "Evaluating", "Inference Demo"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#06d6a0;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#e94560;">&#9679;</span>'
            else:
                icon = '<span style="color:#adb5bd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>RT-DETRv2 Object Detection Pipeline</h2>
        <p><b>Model:</b> {model_name} | <b>Dataset:</b> {dataset_repo}</p>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Downloading and splitting dataset...")),
        do_flush=True,
    )

    data_dir = await prepare_data(
        dataset_repo=dataset_repo,
        annotations_path=annotations_path,
        images_subdir=images_subdir,
        val_fraction=val_fraction,
    )

    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Fine-tuning model...")),
        do_flush=True,
    )

    finetuned_dir = await train(
        model_name, data_dir, epochs, lr, batch_size,
        eval_every_n_epochs=eval_every_n_epochs,
    )

    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Running COCO mAP evaluation...")),
        do_flush=True,
    )

    metrics_json = await evaluate(finetuned_dir, data_dir, threshold)
    metrics = json.loads(metrics_json)

    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Rendering bounding box demo...")),
        do_flush=True,
    )

    demo_json = await inference_demo(
        finetuned_dir, data_dir, threshold, demo_images,
        metrics_json=metrics_json,
    )

    ft_map = metrics["finetuned"].get("map", 0)
    ft_map50 = metrics["finetuned"].get("map_50", 0)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <h3>{model_name}</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{metrics['num_val_images']}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value highlight">{ft_map:.3f}</div><div class="label">mAP</div></div>
      <div class="stat"><div class="value highlight">{ft_map50:.3f}</div><div class="label">mAP@50</div></div>
    </div>
    <div class="card">
      <b>Configuration:</b> {epochs} epochs | LR {lr} | Batch size {batch_size} |
      Val fraction {val_fraction} | Threshold {threshold}
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info(f"Pipeline complete. Fine-tuned mAP: {ft_map:.3f}")
    return finetuned_dir, json.dumps({"metrics": metrics, "demo": json.loads(demo_json)})
