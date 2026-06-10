"""
Nuclei Instance Segmentation — Fine-tune Mask R-CNN on PanNuke pathology images.

Pipeline: download the PanNuke dataset (190k nuclei, 5 cell types, 19 tissue types),
fine-tune a Mask R-CNN for multi-class nuclei instance segmentation, evaluate with
COCO mAP (bbox + mask), and render class-colored mask overlays on held-out images.

Each nucleus is colored by its cell type — neoplastic (red), inflammatory (blue),
connective (green), dead (gray), epithelial (purple) — making it immediately
clear which cells the model identifies and classifies.

Usage:
    # Default (Mask R-CNN ResNet-50 FPN v2 on PanNuke)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2

    # Remote
    flyte run workflow.py pipeline --epochs 10

    # Just prepare data (cached)
    flyte run --local --tui workflow.py prepare_data
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
# Nucleus class definitions
# ------------------------------------------------------------------

# PanNuke categories (0-indexed in dataset) → Mask R-CNN classes (1-indexed, 0=background)
CLASS_NAMES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]
NUM_CLASSES = len(CLASS_NAMES)  # 6 (including background)

# Fixed colors per class for consistent visualizations
CLASS_COLORS = {
    1: (239, 68, 68),    # neoplastic — red
    2: (59, 130, 246),   # inflammatory — blue
    3: (34, 197, 94),    # connective — green
    4: (107, 114, 128),  # dead — gray
    5: (168, 85, 247),   # epithelial — purple
}

CLASS_COLORS_HEX = {
    1: "#ef4444",  # neoplastic
    2: "#3b82f6",  # inflammatory
    3: "#22c55e",  # connective
    4: "#6b7280",  # dead
    5: "#a855f7",  # epithelial
}


# ------------------------------------------------------------------
# Report styling — medical imaging teal theme
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0d6e6e; border-bottom: 2px solid #14b8a6; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f766e; margin-top: 20px; }
  .report .card { background: #f0fdfa; border: 1px solid #ccfbf1; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #ccfbf1; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0d6e6e; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0d6e6e; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #ccfbf1; }
  .report tr:nth-child(even) { background: #f0fdfa; }
  .report .highlight { color: #0d6e6e; font-weight: 700; }
  .report .note { background: #f0fdfa; border-left: 4px solid #14b8a6; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #ccfbf1; color: #0d6e6e; }
  .report .chart-container { background: #fff; border: 1px solid #ccfbf1; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; margin: 16px 0; }
  .report .image-card { background: #fff; border: 1px solid #ccfbf1; border-radius: 8px; overflow: hidden; }
  .report .image-card img { width: 100%; aspect-ratio: 1; object-fit: cover; }
  .report .image-card .caption { padding: 8px 12px; font-size: 0.85em; }
  .report .img-pair { display: flex; gap: 12px; margin: 16px 0; flex-wrap: wrap; }
  .report .img-pair > div { flex: 1; min-width: 280px; }
  .report .img-pair img { width: 100%; border-radius: 6px; border: 1px solid #ccfbf1; }
  .report .legend { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
  .report .legend-item { display: flex; align-items: center; gap: 4px; font-size: 0.85em; }
  .report .legend-swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
</style>
"""


def _wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def _class_legend_html() -> str:
    """Render an HTML legend showing class colors."""
    items = ""
    for cls_id in range(1, NUM_CLASSES):
        hex_color = CLASS_COLORS_HEX[cls_id]
        name = CLASS_NAMES[cls_id]
        items += (
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{hex_color};"></span>'
            f'{name}</span>'
        )
    return f'<div class="legend">{items}</div>'


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
    """Generate an SVG line chart from a list of dicts."""
    default_colors = ["#14b8a6", "#0d6e6e", "#06d6a0", "#f59e0b", "#6c757d"]
    colors = colors or default_colors

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

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#ccfbf1" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#99f6e4" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#99f6e4" stroke-width="1.5"/>'
    )

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
        for i in range(6):
            x_tick = x_min + x_range * i / 5
            px = sx(x_tick)
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_tick:.0f}</text>'
            )

    if not data:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="13" fill="#99f6e4" font-style="italic">Waiting for data...</text>'
        )
    for si, key in enumerate(y_keys):
        color = colors[si % len(colors)]
        points = [(sx(d[x_key]), sy(d[key])) for d in data if key in d]
        if not points:
            continue
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
            for px, py in points[1:]:
                path_d += f" L {px:.1f},{py:.1f}"
            dash = ' stroke-dasharray="6,3"' if si % 2 == 1 else ""
            lines.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"{dash}/>'
            )
        if len(points) <= 30:
            for px, py in points:
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>'
                )

    if title:
        lines.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0d6e6e">{title}</text>'
        )

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
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#14b8a6", "#0d6e6e", "#06d6a0", "#99f6e4"]
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

    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        lines_svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#ccfbf1" stroke-width="1"/>'
        )
        lines_svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

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
                f'height="{bh:.1f}" fill="{color}" rx="3"/>'
            )
            lines_svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#0d6e6e">'
                f'{val:.3f}</text>'
            )
        display_label = label if len(label) <= 12 else label[:10] + ".."
        lines_svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" '
            f'text-anchor="middle" font-size="10" fill="#6c757d">{display_label}</text>'
        )

    if title:
        lines_svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0d6e6e">{title}</text>'
        )

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
# Visualization helpers — class-colored instance mask overlays
# ------------------------------------------------------------------

def _overlay_masks_by_class(image, masks, labels, alpha: float = 0.45):
    """Overlay instance masks colored by nucleus class.

    Each nucleus gets the fixed color for its class (neoplastic=red,
    inflammatory=blue, etc.), with white contour outlines on boundaries.
    """
    import numpy as np
    from PIL import Image as PILImage, ImageFilter

    img_array = np.array(image).astype(np.float32)
    n_masks = len(masks)

    if n_masks == 0:
        return image.copy()

    overlay = img_array.copy()

    for mask, label in zip(masks, labels):
        if hasattr(mask, "numpy"):
            mask_np = mask.numpy()
        else:
            mask_np = np.array(mask)
        mask_bool = mask_np > 0.5

        # Get class color (default to white if unknown)
        label_int = int(label) if hasattr(label, "item") else int(label)
        color = CLASS_COLORS.get(label_int, (200, 200, 200))

        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )

    # Draw white contours on mask boundaries
    for mask in masks:
        if hasattr(mask, "numpy"):
            mask_np = mask.numpy()
        else:
            mask_np = np.array(mask)
        mask_bool = mask_np > 0.5

        mask_pil = PILImage.fromarray((mask_bool * 255).astype(np.uint8))
        dilated = mask_pil.filter(ImageFilter.MaxFilter(3))
        boundary = np.array(dilated) > np.array(mask_pil)
        overlay[boundary] = [255, 255, 255]

    result = np.clip(overlay, 0, 255).astype(np.uint8)
    return PILImage.fromarray(result)


def _img_to_data_uri(img, max_dim: int = 600) -> str:
    """PIL image to base64 PNG data URI, downscaled for report embedding."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------
# Task 1: Prepare data — download PanNuke and organize for Mask R-CNN
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "RationAI/PanNuke",
    train_folds: str = "fold1,fold2",
    val_fold: str = "fold3",
    seed: int = 42,
) -> flyte.io.Dir:
    """Download PanNuke from HuggingFace and organize into train/val splits.

    PanNuke has 3 folds; by default fold1+fold2 are used for training, fold3
    for validation. Each sample has an image, a list of instance masks, and
    a list of class labels (0-4) per nucleus.

    Output directory structure:
      images/train/<idx>.png, images/val/<idx>.png
      masks/train/<idx>/*.png (one per instance)
      labels/train/<idx>.json (list of Mask R-CNN class ids, 1-indexed)
      labels/val/<idx>.json
      meta.json
    """
    from collections import Counter

    import numpy as np
    from datasets import concatenate_datasets, load_dataset
    from PIL import Image

    log.info(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # Combine folds for train/val
    train_fold_names = [f.strip() for f in train_folds.split(",")]
    train_splits = [ds[fold] for fold in train_fold_names if fold in ds]
    train_data = concatenate_datasets(train_splits)
    val_data = ds[val_fold]

    log.info(f"Train: {len(train_data)} images ({train_folds}) | "
             f"Val: {len(val_data)} images ({val_fold})")

    # Pack output directory
    out_dir = tempfile.mkdtemp(prefix="pannuke_seg_")

    all_class_counts = Counter()
    total_nuclei = 0
    nuclei_per_image = []

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        img_out = os.path.join(out_dir, "images", split_name)
        mask_out = os.path.join(out_dir, "masks", split_name)
        label_out = os.path.join(out_dir, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(label_out, exist_ok=True)

        for idx in range(len(split_data)):
            sample = split_data[idx]
            img = sample["image"]
            instances = sample["instances"]     # list of PIL mask images
            categories = sample["categories"]   # list of ints (0-4)

            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save image
            img.save(os.path.join(img_out, f"{idx:05d}.png"), "PNG")

            # Save instance masks and labels
            sample_mask_dir = os.path.join(mask_out, f"{idx:05d}")
            os.makedirs(sample_mask_dir, exist_ok=True)

            # Map PanNuke categories (0-4) to Mask R-CNN classes (1-5)
            rcnn_labels = []
            valid_mask_idx = 0
            for mi, (mask_img, cat) in enumerate(zip(instances, categories)):
                mask_np = np.array(mask_img.convert("L"))
                if mask_np.max() < 128:
                    continue  # skip empty masks

                mask_pil = Image.fromarray((mask_np > 128).astype(np.uint8) * 255)
                mask_pil.save(os.path.join(sample_mask_dir, f"mask_{valid_mask_idx:04d}.png"), "PNG")
                rcnn_label = int(cat) + 1  # shift to 1-indexed for Mask R-CNN
                rcnn_labels.append(rcnn_label)
                all_class_counts[CLASS_NAMES[rcnn_label]] += 1
                valid_mask_idx += 1

            with open(os.path.join(label_out, f"{idx:05d}.json"), "w") as f:
                json.dump(rcnn_labels, f)

            total_nuclei += len(rcnn_labels)
            nuclei_per_image.append(len(rcnn_labels))

    # Save metadata
    meta = {
        "dataset_name": dataset_name,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "num_train": len(train_data),
        "num_val": len(val_data),
        "total_nuclei": total_nuclei,
        "avg_nuclei_per_image": float(np.mean(nuclei_per_image)) if nuclei_per_image else 0,
        "class_distribution": dict(all_class_counts),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Data saved: {total_nuclei} nuclei across {len(train_data) + len(val_data)} images")
    log.info(f"Class distribution: {dict(all_class_counts)}")

    # -- Build report with class-colored overlays --
    sample_indices = list(range(min(6, len(train_data))))
    sample_html = '<div class="image-grid">'

    for idx in sample_indices:
        img_path = os.path.join(out_dir, "images", "train", f"{idx:05d}.png")
        mask_dir = os.path.join(out_dir, "masks", "train", f"{idx:05d}")
        label_path = os.path.join(out_dir, "labels", "train", f"{idx:05d}.json")

        img = Image.open(img_path).convert("RGB")
        with open(label_path) as f:
            labels = json.load(f)

        masks = []
        mask_files = sorted(os.listdir(mask_dir)) if os.path.isdir(mask_dir) else []
        for mf in mask_files:
            m = np.array(Image.open(os.path.join(mask_dir, mf)).convert("L"))
            masks.append(m)

        overlay = _overlay_masks_by_class(img, masks, labels, alpha=0.45)

        # Count per class for this image
        class_counts = Counter(labels)
        class_badges = " ".join(
            f'<span style="color:{CLASS_COLORS_HEX.get(c, "#666")};">'
            f'{CLASS_NAMES[c]}:{n}</span>'
            for c, n in sorted(class_counts.items())
        )

        sample_html += f'''
        <div class="image-card">
          <img src="{_img_to_data_uri(overlay, max_dim=400)}" alt="nuclei overlay" />
          <div class="caption">
            <span class="badge badge-info">{len(labels)} nuclei</span>
            <div style="font-size:0.75em;margin-top:4px;">{class_badges}</div>
          </div>
        </div>'''
    sample_html += "</div>"

    # Class distribution bar chart
    dist_labels = [CLASS_NAMES[i] for i in range(1, NUM_CLASSES)]
    dist_values = [float(all_class_counts.get(CLASS_NAMES[i], 0)) for i in range(1, NUM_CLASSES)]

    dist_chart = _make_bar_chart(
        labels=dist_labels,
        series={"Count": dist_values},
        title="Nucleus Class Distribution (All Data)",
        colors=[CLASS_COLORS_HEX[i] for i in range(1, NUM_CLASSES)],
    )

    report_html = f"""
    <h2>Dataset: PanNuke</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{meta['num_train'] + meta['num_val']}</div><div class="label">Total Images</div></div>
      <div class="stat"><div class="value">{meta['num_train']}</div><div class="label">Training</div></div>
      <div class="stat"><div class="value">{meta['num_val']}</div><div class="label">Validation</div></div>
      <div class="stat"><div class="value">{meta['total_nuclei']:,}</div><div class="label">Total Nuclei</div></div>
      <div class="stat"><div class="value">{meta['avg_nuclei_per_image']:.0f}</div><div class="label">Avg Nuclei/Image</div></div>
      <div class="stat"><div class="value">{NUM_CLASSES - 1}</div><div class="label">Nucleus Classes</div></div>
    </div>
    {_class_legend_html()}
    <h3>Sample Images with Class-Colored Masks</h3>
    {sample_html}
    <div class="chart-container">{dist_chart}</div>
    <div class="note">
      <b>PanNuke:</b> 190k nuclei from 19 tissue types, classified into 5 categories —
      <span style="color:#ef4444;">neoplastic</span> (tumor cells),
      <span style="color:#3b82f6;">inflammatory</span> (immune cells),
      <span style="color:#22c55e;">connective</span> (stromal cells),
      <span style="color:#6b7280;">dead</span> (necrotic/apoptotic),
      <span style="color:#a855f7;">epithelial</span> (normal epithelium).
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Dataset class for Mask R-CNN (multi-class)
# ------------------------------------------------------------------

def _build_mask_rcnn_dataset(images_dir: str, masks_dir: str, labels_dir: str, augment: bool = False):
    """Build a torch Dataset returning (image_tensor, target_dict) for Mask R-CNN.

    target_dict contains:
      - boxes: [N, 4] in xyxy format
      - labels: [N] class ids (1-5 for the five nucleus types)
      - masks: [N, H, W] binary masks for each instance
    """
    import numpy as np
    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from torchvision import transforms as T

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    class NucleiDataset(Dataset):
        def __init__(self):
            self.image_files = image_files
            self.augment = augment

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_id = os.path.splitext(img_name)[0]

            img = Image.open(os.path.join(images_dir, img_name)).convert("RGB")
            img_tensor = T.ToTensor()(img)

            # Load class labels
            label_file = os.path.join(labels_dir, f"{img_id}.json")
            if os.path.exists(label_file):
                with open(label_file) as f:
                    class_labels = json.load(f)
            else:
                class_labels = []

            # Load instance masks
            mask_dir = os.path.join(masks_dir, img_id)
            masks_list = []
            boxes_list = []
            valid_labels = []

            if os.path.isdir(mask_dir):
                mask_files = sorted(os.listdir(mask_dir))
                for mi, mf in enumerate(mask_files):
                    m = np.array(Image.open(os.path.join(mask_dir, mf)).convert("L"))
                    m_bool = m > 128

                    if not m_bool.any():
                        continue

                    rows = np.any(m_bool, axis=1)
                    cols = np.any(m_bool, axis=0)
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]

                    if (rmax - rmin) < 2 or (cmax - cmin) < 2:
                        continue

                    boxes_list.append([float(cmin), float(rmin),
                                       float(cmax + 1), float(rmax + 1)])
                    masks_list.append(m_bool.astype(np.uint8))

                    label = class_labels[mi] if mi < len(class_labels) else 1
                    valid_labels.append(label)

            if masks_list:
                masks = torch.tensor(np.stack(masks_list), dtype=torch.uint8)
                boxes = torch.tensor(boxes_list, dtype=torch.float32)
                labels = torch.tensor(valid_labels, dtype=torch.int64)
            else:
                masks = torch.zeros((0, img_tensor.shape[1], img_tensor.shape[2]),
                                    dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)

            # Data augmentation: random horizontal flip
            if self.augment and random.random() > 0.5:
                img_tensor = T.functional.hflip(img_tensor)
                masks = masks.flip(-1)
                w = img_tensor.shape[2]
                if len(boxes) > 0:
                    boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": idx,
            }

            return img_tensor, target

    return NucleiDataset()


# ------------------------------------------------------------------
# Task 2: Train — fine-tune Mask R-CNN for multi-class nuclei segmentation
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    data_dir: flyte.io.Dir,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 5e-3,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
) -> flyte.io.Dir:
    """Fine-tune a pretrained Mask R-CNN (ResNet-50 FPN v2) for multi-class
    nuclei segmentation (5 nucleus types + background).

    Uses a custom PyTorch training loop (torchvision detection models are not
    compatible with HuggingFace Trainer). Reports individual loss components
    (classifier, box regression, mask, objectness, RPN box) live.
    """
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    log.info("Loading Mask R-CNN ResNet-50 FPN v2...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Loading Model...</h2>"
        "<p>Mask R-CNN ResNet-50 FPN v2 (COCO pretrained)</p>"
        "<p>Replacing heads for 5-class nuclei segmentation...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    # Build datasets
    train_ds = _build_mask_rcnn_dataset(
        os.path.join(data_path, "images", "train"),
        os.path.join(data_path, "masks", "train"),
        os.path.join(data_path, "labels", "train"),
        augment=True,
    )
    val_ds = _build_mask_rcnn_dataset(
        os.path.join(data_path, "images", "val"),
        os.path.join(data_path, "masks", "val"),
        os.path.join(data_path, "labels", "val"),
        augment=False,
    )

    log.info(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Load pretrained Mask R-CNN (keep default anchors — they're coupled to
    # the pretrained RPN weights). Lower NMS threshold for dense/touching
    # nuclei and allow more detections per image.
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, NUM_CLASSES
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def collate_fn(batch):
        return tuple(zip(*batch))

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader),
    )

    training_log: list[dict] = []
    loop = asyncio.get_running_loop()
    total_steps = epochs * len(train_loader)

    class_badges = _class_legend_html()

    def _build_training_report() -> str:
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>Mask R-CNN ResNet-50 FPN v2 — Multi-Class Nuclei</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Images</div></div>
          <div class="stat"><div class="value">{len(val_ds)}</div><div class="label">Val Images</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{NUM_CLASSES - 1}</div><div class="label">Nucleus Classes</div></div>
        </div>
        {class_badges}
        """

        charts_html = ""
        if training_log:
            current = training_log[-1]
            progress_pct = current["step"] / total_steps * 100 if total_steps else 0
            charts_html += f"""
            <div class="card">
              <b>Step {current['step']}/{total_steps}</b>
              ({progress_pct:.0f}%) |
              Epoch {current['epoch']}/{epochs} |
              Total Loss: <span class="highlight">{current['total_loss']:.4f}</span>
              <div style="background:#ccfbf1;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#14b8a6;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_chart = _make_line_chart(
                data=training_log,
                x_key="step",
                y_keys=["total_loss"],
                title="Total Training Loss",
                x_label="Step",
                y_label="Loss",
                colors=["#14b8a6"],
                x_range_override=(0, total_steps),
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

            component_keys = ["loss_mask", "loss_classifier", "loss_box_reg"]
            component_names = {
                "loss_mask": "Mask Loss",
                "loss_classifier": "Classifier",
                "loss_box_reg": "Box Reg",
            }
            if all(k in training_log[-1] for k in component_keys):
                comp_chart = _make_line_chart(
                    data=training_log,
                    x_key="step",
                    y_keys=component_keys,
                    title="Loss Components",
                    x_label="Step",
                    y_label="Loss",
                    colors=["#0d6e6e", "#14b8a6", "#06d6a0"],
                    x_range_override=(0, total_steps),
                    y_display_names=component_names,
                )
                charts_html += f'<div class="chart-container">{comp_chart}</div>'

            lr_chart = _make_line_chart(
                data=training_log,
                x_key="step",
                y_keys=["lr"],
                title="Learning Rate Schedule",
                x_label="Step",
                y_label="LR",
                colors=["#0d6e6e"],
                x_range_override=(0, total_steps),
            )
            charts_html += f'<div class="chart-container">{lr_chart}</div>'

        return _wrap_report(stats_html + charts_html)

    def _train_loop():
        global_step = 0
        log_interval = max(1, len(train_loader) // 10)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []

            for batch_idx, (images, targets) in enumerate(train_loader):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) if hasattr(v, "to") else v
                            for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                global_step += 1
                loss_val = losses.item()
                epoch_losses.append(loss_val)

                if global_step % log_interval == 0 or global_step == total_steps:
                    entry = {
                        "step": global_step,
                        "epoch": epoch,
                        "total_loss": round(loss_val, 4),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                    for k, v in loss_dict.items():
                        entry[k] = round(v.item(), 4)

                    training_log.append(entry)
                    log.info(
                        f"step={global_step}/{total_steps} "
                        f"epoch={epoch} loss={loss_val:.4f}"
                    )

                    asyncio.run_coroutine_threadsafe(
                        flyte.report.replace.aio(
                            _build_training_report(), do_flush=True,
                        ),
                        loop,
                    )

            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            log.info(f"Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

    log.info("Starting training...")
    await asyncio.to_thread(_train_loop)
    log.info("Training complete.")

    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_mask_rcnn")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    model_config = {
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    shutil.copy2(os.path.join(data_path, "meta.json"), os.path.join(save_dir, "meta.json"))
    log.info(f"Model saved to {save_dir}")

    # Final training report
    stats_html = f"""
    <h2>Training Complete</h2>
    <h3>Mask R-CNN ResNet-50 FPN v2 — Multi-Class Nuclei</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Images</div></div>
      <div class="stat"><div class="value">{len(val_ds)}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
      <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
      <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Trainable Params</div></div>
    </div>
    {class_badges}
    """

    charts_html = ""
    if training_log:
        initial_loss = training_log[0]["total_loss"]
        final_loss = training_log[-1]["total_loss"]
        min_loss = min(d["total_loss"] for d in training_log)

        charts_html += f"""
        <div class="card">
          <b>Training Summary:</b>
          Initial loss: {initial_loss:.4f} |
          Final loss: <span class="highlight">{final_loss:.4f}</span> |
          Min loss: {min_loss:.4f} |
          Total steps: {total_steps}
        </div>
        """

        loss_chart = _make_line_chart(
            data=training_log,
            x_key="step",
            y_keys=["total_loss"],
            title="Total Training Loss",
            x_label="Step",
            y_label="Loss",
            colors=["#14b8a6"],
            x_range_override=(0, total_steps),
        )
        charts_html += f'<div class="chart-container">{loss_chart}</div>'

        component_keys = ["loss_mask", "loss_classifier", "loss_box_reg"]
        if all(k in training_log[-1] for k in component_keys):
            comp_chart = _make_line_chart(
                data=training_log,
                x_key="step",
                y_keys=component_keys,
                title="Loss Components",
                x_label="Step",
                y_label="Loss",
                colors=["#0d6e6e", "#14b8a6", "#06d6a0"],
                x_range_override=(0, total_steps),
                y_display_names={
                    "loss_mask": "Mask Loss",
                    "loss_classifier": "Classifier",
                    "loss_box_reg": "Box Reg",
                },
            )
            charts_html += f'<div class="chart-container">{comp_chart}</div>'

        lr_chart = _make_line_chart(
            data=training_log,
            x_key="step",
            y_keys=["lr"],
            title="Learning Rate Schedule",
            x_label="Step",
            y_label="LR",
            colors=["#0d6e6e"],
            x_range_override=(0, total_steps),
        )
        charts_html += f'<div class="chart-container">{lr_chart}</div>'

    await flyte.report.replace.aio(_wrap_report(stats_html + charts_html), do_flush=True)

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Inference helper — load model and run on PIL images
# ------------------------------------------------------------------

def _load_mask_rcnn(model_path: str, device):
    """Load fine-tuned Mask R-CNN from saved state dict."""
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    num_classes = config["num_classes"]
    model = maskrcnn_resnet50_fpn_v2(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.load_state_dict(torch.load(
        os.path.join(model_path, "model.pth"),
        map_location=device,
        weights_only=True,
    ))
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
# Task 3: Evaluate — COCO mAP for bbox + segmentation, per-class
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    score_threshold: float = 0.5,
) -> str:
    """Evaluate the fine-tuned Mask R-CNN on the validation set.

    Computes COCO mAP for both bounding boxes and instance masks,
    plus per-class detection accuracy.
    """
    import numpy as np
    import torch
    from collections import Counter
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from torchvision import transforms as T

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Evaluation</h2><p>Loading model and scoring validation images...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_mask_rcnn(ft_path, device)

    val_ds = _build_mask_rcnn_dataset(
        os.path.join(data_path, "images", "val"),
        os.path.join(data_path, "masks", "val"),
        os.path.join(data_path, "labels", "val"),
        augment=False,
    )

    all_targets = []
    all_preds = []
    gt_class_counts = Counter()
    pred_class_counts = Counter()

    for idx in range(len(val_ds)):
        img_tensor, target = val_ds[idx]

        all_targets.append({
            "boxes": target["boxes"],
            "labels": target["labels"],
            "masks": target["masks"],
        })

        for lbl in target["labels"].tolist():
            gt_class_counts[lbl] += 1

        img_t = img_tensor.to(device)
        with torch.no_grad():
            outputs = model([img_t])[0]

        keep = outputs["scores"] >= score_threshold
        pred_masks = (outputs["masks"][keep] > 0.5).squeeze(1).cpu()

        pred = {
            "boxes": outputs["boxes"][keep].cpu(),
            "labels": outputs["labels"][keep].cpu(),
            "scores": outputs["scores"][keep].cpu(),
            "masks": pred_masks,
        }
        all_preds.append(pred)

        for lbl in pred["labels"].tolist():
            pred_class_counts[lbl] += 1

    # COCO mAP — bbox
    bbox_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    bbox_metric.update(
        [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]} for p in all_preds],
        [{"boxes": t["boxes"], "labels": t["labels"]} for t in all_targets],
    )
    bbox_results = bbox_metric.compute()

    # COCO mAP — segm
    segm_metric = MeanAveragePrecision(box_format="xyxy", iou_type="segm")
    segm_metric.update(
        [{"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"], "masks": p["masks"]}
         for p in all_preds],
        [{"boxes": t["boxes"], "labels": t["labels"], "masks": t["masks"]}
         for t in all_targets],
    )
    segm_results = segm_metric.compute()

    def to_python(v):
        if hasattr(v, "numel"):
            return round(v.item(), 4) if v.numel() == 1 else v.tolist()
        return v

    bbox_metrics = {k: to_python(v) for k, v in bbox_results.items()}
    segm_metrics = {k: to_python(v) for k, v in segm_results.items()}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Bbox mAP: {bbox_metrics.get('map', 0):.3f} | "
             f"Segm mAP: {segm_metrics.get('map', 0):.3f}")

    # Build report
    metric_keys = ["map", "map_50", "map_75", "mar_10"]
    metric_display = {
        "map": "mAP", "map_50": "mAP@50", "map_75": "mAP@75", "mar_10": "mAR@10",
    }

    bar_chart = _make_bar_chart(
        labels=[metric_display.get(k, k) for k in metric_keys],
        series={
            "BBox": [bbox_metrics.get(k, 0) for k in metric_keys],
            "Segm": [segm_metrics.get(k, 0) for k in metric_keys],
        },
        title="COCO Evaluation Metrics (BBox vs Segm)",
        colors=["#14b8a6", "#0d6e6e"],
        y_max_cap=1.0,
    )

    table_rows = ""
    for k in metric_keys:
        b_val = bbox_metrics.get(k, 0)
        s_val = segm_metrics.get(k, 0)
        table_rows += (
            f"<tr><td><b>{metric_display.get(k, k)}</b></td>"
            f"<td>{b_val:.3f}</td>"
            f"<td>{s_val:.3f}</td></tr>"
        )

    # Per-class detection counts
    class_table_rows = ""
    for cls_id in range(1, NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id]
        gt_count = gt_class_counts.get(cls_id, 0)
        pred_count = pred_class_counts.get(cls_id, 0)
        hex_color = CLASS_COLORS_HEX[cls_id]
        class_table_rows += (
            f'<tr><td><span style="color:{hex_color};font-weight:600;">'
            f'{cls_name}</span></td>'
            f'<td>{gt_count:,}</td>'
            f'<td>{pred_count:,}</td></tr>'
        )

    eval_html = f"""
    <h2>Evaluation Results</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(val_ds)}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value highlight">{bbox_metrics.get('map', 0):.3f}</div><div class="label">BBox mAP</div></div>
      <div class="stat"><div class="value highlight">{segm_metrics.get('map', 0):.3f}</div><div class="label">Segm mAP</div></div>
      <div class="stat"><div class="value">{score_threshold}</div><div class="label">Score Threshold</div></div>
    </div>
    {_class_legend_html()}
    <div class="chart-container">{bar_chart}</div>
    <table>
      <tr><th>Metric</th><th>BBox</th><th>Segm</th></tr>
      {table_rows}
    </table>
    <h3>Per-Class Detection Counts</h3>
    <table>
      <tr><th>Class</th><th>Ground Truth</th><th>Predicted</th></tr>
      {class_table_rows}
    </table>
    <div class="note">
      <b>BBox mAP</b> evaluates bounding box detection quality.
      <b>Segm mAP</b> evaluates instance mask quality — how well predicted
      masks overlap with ground truth. Both use IoU thresholds from 0.50 to 0.95.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(eval_html), do_flush=True)

    metrics_out = {
        "bbox": {k: v for k, v in bbox_metrics.items() if isinstance(v, (int, float))},
        "segm": {k: v for k, v in segm_metrics.items() if isinstance(v, (int, float))},
        "num_val_images": len(val_ds),
        "gt_class_counts": dict(gt_class_counts),
        "pred_class_counts": dict(pred_class_counts),
    }

    return json.dumps(metrics_out)


# ------------------------------------------------------------------
# Task 4: Inference — class-colored instance mask overlays
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def inference(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    score_threshold: float = 0.5,
    max_images: int = 8,
    metrics_json: str = "{}",
) -> str:
    """Run predictions on validation images and render side-by-side mask overlays.

    Ground truth and predicted masks are both colored by nucleus class type.
    """
    import torch
    from collections import Counter
    from PIL import Image
    from torchvision import transforms as T

    log.info("Starting inference visualization...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Inference</h2><p>Generating class-colored mask overlays...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_mask_rcnn(ft_path, device)

    val_ds = _build_mask_rcnn_dataset(
        os.path.join(data_path, "images", "val"),
        os.path.join(data_path, "masks", "val"),
        os.path.join(data_path, "labels", "val"),
        augment=False,
    )

    rng = random.Random(42)
    indices = list(range(len(val_ds)))
    rng.shuffle(indices)
    indices = indices[:max_images]

    html_blocks = []
    total_gt_nuclei = 0
    total_pred_nuclei = 0

    for idx in indices:
        img_tensor, target = val_ds[idx]
        img_name = val_ds.image_files[idx]
        val_img_dir = os.path.join(data_path, "images", "val")
        img = Image.open(os.path.join(val_img_dir, img_name)).convert("RGB")

        gt_masks = target["masks"]
        gt_labels = target["labels"]
        n_gt = len(gt_masks)
        total_gt_nuclei += n_gt

        img_t = img_tensor.to(device)
        with torch.no_grad():
            outputs = model([img_t])[0]

        keep = outputs["scores"] >= score_threshold
        pred_masks = (outputs["masks"][keep] > 0.5).squeeze(1).cpu()
        pred_labels = outputs["labels"][keep].cpu()
        pred_scores = outputs["scores"][keep].cpu()
        n_pred = len(pred_masks)
        total_pred_nuclei += n_pred

        # Create class-colored overlay visualizations
        gt_overlay = _overlay_masks_by_class(img, gt_masks, gt_labels, alpha=0.45)
        pred_overlay = _overlay_masks_by_class(img, pred_masks, pred_labels, alpha=0.45)

        # Per-class count comparison
        gt_counts = Counter(gt_labels.tolist())
        pred_counts = Counter(pred_labels.tolist())

        class_detail = ""
        for cls_id in range(1, NUM_CLASSES):
            g = gt_counts.get(cls_id, 0)
            p = pred_counts.get(cls_id, 0)
            if g > 0 or p > 0:
                hex_c = CLASS_COLORS_HEX[cls_id]
                class_detail += (
                    f'<span style="color:{hex_c};font-size:0.8em;">'
                    f'{CLASS_NAMES[cls_id]}:{g}/{p} </span>'
                )

        count_diff = abs(n_gt - n_pred)
        count_badge_cls = "badge-success" if count_diff <= 3 else "badge-danger"

        avg_score = float(pred_scores.mean()) if len(pred_scores) > 0 else 0

        html_blocks.append(f"""
        <div class="card">
          <div style="margin-bottom:8px;">
            <b>Image {idx + 1}</b>
            <span class="badge {count_badge_cls}">Count: {n_pred}/{n_gt}</span>
            <span class="badge badge-info">Avg conf: {avg_score:.1%}</span>
            <div style="margin-top:4px;">{class_detail}</div>
          </div>
          <div class="img-pair">
            <div>
              <p style="text-align:center;font-weight:600;color:#0d6e6e;">
                Ground Truth — {n_gt} nuclei</p>
              <img src="{_img_to_data_uri(gt_overlay)}" />
            </div>
            <div>
              <p style="text-align:center;font-weight:600;color:#14b8a6;">
                Predicted — {n_pred} nuclei</p>
              <img src="{_img_to_data_uri(pred_overlay)}" />
            </div>
          </div>
        </div>""")

    metrics = json.loads(metrics_json)
    bbox_map = metrics.get("bbox", {}).get("map")
    segm_map = metrics.get("segm", {}).get("map")

    metric_stats = ""
    if bbox_map is not None:
        metric_stats += f'<div class="stat"><div class="value highlight">{bbox_map:.3f}</div><div class="label">BBox mAP</div></div>'
    if segm_map is not None:
        metric_stats += f'<div class="stat"><div class="value highlight">{segm_map:.3f}</div><div class="label">Segm mAP</div></div>'

    demo_html = f"""
    <h2>Instance Segmentation Results</h2>
    <h3>Multi-Class Nuclei Detection</h3>
    <div class="stat-grid">
      {metric_stats}
      <div class="stat"><div class="value">{len(indices)}</div><div class="label">Images Shown</div></div>
      <div class="stat"><div class="value">{total_gt_nuclei}</div><div class="label">GT Nuclei</div></div>
      <div class="stat"><div class="value">{total_pred_nuclei}</div><div class="label">Pred Nuclei</div></div>
      <div class="stat"><div class="value">{score_threshold}</div><div class="label">Score Threshold</div></div>
    </div>
    {_class_legend_html()}
    <p>Each nucleus is colored by its class. White outlines mark instance boundaries.
    Left: ground truth. Right: model predictions. Count format: predicted/ground truth.</p>
    {"".join(html_blocks)}
    """

    await flyte.report.replace.aio(_wrap_report(demo_html), do_flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json.dumps({
        "num_images": len(indices),
        "total_gt_nuclei": total_gt_nuclei,
        "total_pred_nuclei": total_pred_nuclei,
    })


# ------------------------------------------------------------------
# Pipeline — chains all 4 tasks
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    dataset_name: str = "RationAI/PanNuke",
    train_folds: str = "fold1,fold2",
    val_fold: str = "fold3",
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 5e-3,
    score_threshold: float = 0.5,
    sample_images: int = 8,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end multi-class nuclei instance segmentation pipeline.

    Returns the fine-tuned model directory and a JSON summary of metrics.

    1. Download PanNuke dataset (190k nuclei, 5 cell types) from HuggingFace
    2. Fine-tune Mask R-CNN ResNet-50 FPN v2 for multi-class nuclei segmentation
    3. Evaluate with COCO mAP (bbox + segmentation masks)
    4. Render class-colored instance mask overlays on validation images
    """
    log.info(f"Pipeline: Mask R-CNN | dataset={dataset_name}")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Preparing Data", "Training Mask R-CNN", "Evaluating", "Inference Demo"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#059669;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#14b8a6;">&#9679;</span>'
            else:
                icon = '<span style="color:#99f6e4;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Nuclei Instance Segmentation Pipeline</h2>
        <p><b>Model:</b> Mask R-CNN ResNet-50 FPN v2 | <b>Dataset:</b> PanNuke</p>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Step 1: Prepare data
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Downloading PanNuke from HuggingFace...")),
        do_flush=True,
    )
    data_dir = await prepare_data(
        dataset_name=dataset_name,
        train_folds=train_folds,
        val_fold=val_fold,
    )

    # Step 2: Train
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Fine-tuning Mask R-CNN on 5 nucleus classes...")),
        do_flush=True,
    )
    finetuned_dir = await train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Computing COCO mAP (bbox + segm)...")),
        do_flush=True,
    )
    metrics_json = await evaluate(finetuned_dir, data_dir, score_threshold)
    metrics = json.loads(metrics_json)

    # Step 4: Inference
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating class-colored mask overlays...")),
        do_flush=True,
    )
    inference_json = await inference(
        finetuned_dir, data_dir, score_threshold, sample_images,
        metrics_json=metrics_json,
    )

    # Final pipeline report
    bbox_map = metrics.get("bbox", {}).get("map", 0)
    segm_map = metrics.get("segm", {}).get("map", 0)
    inference_results = json.loads(inference_json)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <h3>Mask R-CNN ResNet-50 FPN v2 on PanNuke</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{metrics.get('num_val_images', 0)}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value highlight">{bbox_map:.3f}</div><div class="label">BBox mAP</div></div>
      <div class="stat"><div class="value highlight">{segm_map:.3f}</div><div class="label">Segm mAP</div></div>
    </div>
    {_class_legend_html()}
    <div class="card">
      <b>Configuration:</b> {epochs} epochs | LR {learning_rate} | Batch size {batch_size} |
      Train: {train_folds} | Val: {val_fold} | Score threshold {score_threshold}
    </div>
    <div class="note">
      Pipeline ran: prepare_data &#8594; train &#8594; evaluate &#8594; inference.
      Check individual task reports for detailed loss curves, metrics charts,
      and side-by-side class-colored mask visualizations.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info(f"Pipeline complete. BBox mAP: {bbox_map:.3f}, Segm mAP: {segm_map:.3f}")
    return finetuned_dir, json.dumps({"metrics": metrics, "inference": inference_results})
