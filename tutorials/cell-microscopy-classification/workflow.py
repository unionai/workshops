"""
Cell Microscopy Image Classification — Fine-tune ViT on blood cell images.

Pipeline: download a HuggingFace image classification dataset, fine-tune a
Vision Transformer (ViT), evaluate with per-class metrics and confusion matrix,
and render an inference demo with confidence visualizations.

Works with any HuggingFace dataset that has "image" and "label" columns.

Usage:
    # Default (ViT-base on blood cell images)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2

    # Remote
    flyte run workflow.py pipeline --epochs 10

    # Swap model or dataset
    flyte run workflow.py pipeline --model_name "google/vit-base-patch16-224-in21k"
    flyte run workflow.py pipeline --dataset_name "Falah/Alzheimer_MRI"
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
# Report styling — biotech purple/violet microscopy theme
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #4c1d95; border-bottom: 2px solid #7c3aed; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #5b21b6; margin-top: 20px; }
  .report .card { background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #ede9fe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #4c1d95; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #4c1d95; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #ede9fe; }
  .report tr:nth-child(even) { background: #f5f3ff; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #ede9fe; color: #4c1d95; }
  .report .chart-container { background: #fff; border: 1px solid #ede9fe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #f5f3ff; border-left: 4px solid #7c3aed; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }
  .report .image-card { background: #fff; border: 1px solid #ede9fe; border-radius: 8px; overflow: hidden; }
  .report .image-card img { width: 100%; aspect-ratio: 1; object-fit: cover; }
  .report .image-card .caption { padding: 8px 12px; font-size: 0.85em; }
  .report .highlight { color: #4c1d95; font-weight: 700; }
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
    """Generate an SVG line chart from a list of dicts."""
    default_colors = ["#7c3aed", "#4c1d95", "#06d6a0", "#f59e0b", "#6c757d"]
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

    # Grid lines
    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#ede9fe" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    # Axes
    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#a78bfa" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#a78bfa" stroke-width="1.5"/>'
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
        for i in range(6):
            x_tick = x_min + x_range * i / 5
            px = sx(x_tick)
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_tick:.0f}</text>'
            )

    # Plot each series
    if not data:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="13" fill="#a78bfa" font-style="italic">Waiting for data...</text>'
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

    # Title
    if title:
        lines.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#4c1d95">{title}</text>'
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
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#7c3aed", "#4c1d95", "#06d6a0", "#a78bfa"]
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
            f'stroke="#ede9fe" stroke-width="1"/>'
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
                f'height="{bh:.1f}" fill="{color}" rx="3"/>'
            )
            lines_svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#4c1d95">'
                f'{val:.3f}</text>'
            )
        # Group label — truncate long names
        display_label = label if len(label) <= 12 else label[:10] + ".."
        lines_svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" '
            f'text-anchor="middle" font-size="10" fill="#6c757d">{display_label}</text>'
        )

    # Title
    if title:
        lines_svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#4c1d95">{title}</text>'
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


def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    color_scale: str = "purple",
    width: int = 500,
    height: int = 500,
) -> str:
    """Generate an SVG heatmap (e.g. for confusion matrix).

    Args:
        matrix: 2D list of values (rows x cols).
        row_labels: Labels for rows (y-axis, true labels).
        col_labels: Labels for columns (x-axis, predicted labels).
        title: Chart title.
        color_scale: "purple" or "blue".
        width: SVG width.
        height: SVG height.

    Returns:
        SVG string.
    """
    if not matrix or not matrix[0]:
        return ""

    n_rows = len(matrix)
    n_cols = len(matrix[0])

    # Margins
    ml, mr, mt, mb = 100, 30, 50, 80
    cw = width - ml - mr
    ch = height - mt - mb
    cell_w = cw / n_cols
    cell_h = ch / n_rows

    max_val = max(max(row) for row in matrix) if matrix else 1
    max_val = max_val or 1

    # Color interpolation
    if color_scale == "purple":
        bg_r, bg_g, bg_b = 245, 243, 255  # #f5f3ff
        fg_r, fg_g, fg_b = 76, 29, 149    # #4c1d95
    else:
        bg_r, bg_g, bg_b = 219, 234, 254
        fg_r, fg_g, fg_b = 30, 58, 138

    def cell_color(val):
        t = val / max_val if max_val > 0 else 0
        r = int(bg_r + (fg_r - bg_r) * t)
        g = int(bg_g + (fg_g - bg_g) * t)
        b = int(bg_b + (fg_b - bg_b) * t)
        return f"rgb({r},{g},{b})"

    def text_color(val):
        t = val / max_val if max_val > 0 else 0
        return "#fff" if t > 0.5 else "#4c1d95"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Title
    if title:
        lines.append(
            f'<text x="{width / 2}" y="28" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#4c1d95">{title}</text>'
        )

    # Cells
    for ri, row in enumerate(matrix):
        for ci, val in enumerate(row):
            x = ml + ci * cell_w
            y = mt + ri * cell_h
            fill = cell_color(val)
            tc = text_color(val)
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                f'height="{cell_h:.1f}" fill="{fill}" stroke="#fff" stroke-width="1.5" rx="2"/>'
            )
            # Show count
            display_val = int(val) if val == int(val) else f"{val:.1f}"
            lines.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" '
                f'text-anchor="middle" font-size="13" font-weight="600" '
                f'fill="{tc}">{display_val}</text>'
            )

    # Row labels (true labels)
    for ri, label in enumerate(row_labels):
        y = mt + ri * cell_h + cell_h / 2 + 4
        display = label if len(label) <= 14 else label[:12] + ".."
        lines.append(
            f'<text x="{ml - 8}" y="{y:.1f}" text-anchor="end" '
            f'font-size="11" fill="#4c1d95">{display}</text>'
        )

    # Column labels (predicted labels)
    for ci, label in enumerate(col_labels):
        x = ml + ci * cell_w + cell_w / 2
        display = label if len(label) <= 14 else label[:12] + ".."
        lines.append(
            f'<text x="{x:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
            f'font-size="11" fill="#4c1d95" '
            f'transform="rotate(-35, {x:.1f}, {mt + ch + 20})">{display}</text>'
        )

    # Axis labels
    lines.append(
        f'<text x="{ml - 60}" y="{mt + ch / 2}" text-anchor="middle" '
        f'font-size="12" fill="#5b21b6" font-weight="600" '
        f'transform="rotate(-90, {ml - 60}, {mt + ch / 2})">True Label</text>'
    )
    lines.append(
        f'<text x="{ml + cw / 2}" y="{height - 10}" text-anchor="middle" '
        f'font-size="12" fill="#5b21b6" font-weight="600">Predicted Label</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


def _img_to_data_uri(img, max_size: int = 400) -> str:
    """PIL image to base64 JPEG data URI, downscaled for report embedding."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------
# Task 1: Prepare data — download HF image classification dataset
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_name: str = "ehottl/blood_dataset",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> flyte.io.Dir:
    """Download an image classification dataset from HuggingFace and split into train/val.

    Expects a dataset with 'image' (PIL) and 'label' (ClassLabel) columns.
    """
    from collections import Counter

    from datasets import ClassLabel, load_dataset
    from PIL import Image

    log.info(f"Downloading dataset: {dataset_name}")
    ds = load_dataset(dataset_name)

    # Handle datasets with pre-defined splits vs single split
    if "train" in ds and "test" in ds:
        train_split = ds["train"]
        val_split = ds["test"]
        log.info("Using pre-defined train/test splits")
    elif "train" in ds and "validation" in ds:
        train_split = ds["train"]
        val_split = ds["validation"]
        log.info("Using pre-defined train/validation splits")
    elif "train" in ds:
        full = ds["train"].train_test_split(test_size=val_fraction, seed=seed)
        train_split = full["train"]
        val_split = full["test"]
        log.info(f"Split 'train' into train/val with val_fraction={val_fraction}")
    else:
        # Single unnamed split
        key = list(ds.keys())[0]
        full = ds[key].train_test_split(test_size=val_fraction, seed=seed)
        train_split = full["train"]
        val_split = full["test"]
        log.info(f"Split '{key}' into train/val with val_fraction={val_fraction}")

    # Resolve label names
    label_feature = train_split.features.get("label") or train_split.features.get("labels")
    if isinstance(label_feature, ClassLabel):
        label_names = label_feature.names
    else:
        # Fallback: infer from unique values
        all_labels = set(train_split["label"]) | set(val_split["label"])
        label_names = [str(l) for l in sorted(all_labels)]

    num_classes = len(label_names)
    log.info(f"Classes ({num_classes}): {label_names}")
    log.info(f"Train: {len(train_split)} | Val: {len(val_split)}")

    # Compute class distribution
    train_labels = train_split["label"]
    label_counts = Counter(train_labels)

    # Pack output directory
    out_dir = tempfile.mkdtemp(prefix="cell_cls_")

    # Save metadata
    meta = {
        "dataset_name": dataset_name,
        "num_classes": num_classes,
        "label_names": label_names,
        "num_train": len(train_split),
        "num_val": len(val_split),
        "class_distribution": {label_names[k]: v for k, v in sorted(label_counts.items())},
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save images organized by split
    for split_name, split_data in [("train", train_split), ("val", val_split)]:
        split_dir = os.path.join(out_dir, split_name)
        for cls_name in label_names:
            os.makedirs(os.path.join(split_dir, cls_name), exist_ok=True)
        for idx, example in enumerate(split_data):
            img = example["image"]
            label_idx = example["label"]
            cls_name = label_names[label_idx]
            save_path = os.path.join(split_dir, cls_name, f"{idx:05d}.jpg")
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(save_path, "JPEG", quality=95)

    log.info(f"Data saved to {out_dir}")

    # Build report: dataset stats, class distribution chart, sample images
    dist_labels = list(meta["class_distribution"].keys())
    dist_values = list(meta["class_distribution"].values())

    bar_chart = _make_bar_chart(
        labels=dist_labels,
        series={"Count": [float(v) for v in dist_values]},
        title="Class Distribution (Training Set)",
        colors=["#7c3aed"],
    )

    # Sample grid: pick images from different classes
    sample_html = '<div class="image-grid">'
    samples_per_class = max(1, 12 // num_classes)
    rng = random.Random(seed)
    sample_indices = []
    for cls_idx in range(num_classes):
        cls_examples = [i for i, l in enumerate(train_labels) if l == cls_idx]
        rng.shuffle(cls_examples)
        sample_indices.extend(cls_examples[:samples_per_class])

    for idx in sample_indices[:12]:
        example = train_split[idx]
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        label_idx = example["label"]
        cls_name = label_names[label_idx]
        data_uri = _img_to_data_uri(img, max_size=300)
        sample_html += f'''
        <div class="image-card">
          <img src="{data_uri}" alt="{cls_name}" />
          <div class="caption">
            <span class="badge badge-info">{cls_name}</span>
          </div>
        </div>'''
    sample_html += "</div>"

    class_badges = " ".join(
        f'<span class="badge badge-info">{name}</span>' for name in label_names
    )

    report_html = f"""
    <h2>Dataset: {dataset_name}</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_split) + len(val_split)}</div><div class="label">Total Images</div></div>
      <div class="stat"><div class="value">{len(train_split)}</div><div class="label">Training</div></div>
      <div class="stat"><div class="value">{len(val_split)}</div><div class="label">Validation</div></div>
      <div class="stat"><div class="value">{num_classes}</div><div class="label">Classes</div></div>
    </div>
    <p>Classes: {class_badges}</p>
    <div class="chart-container">{bar_chart}</div>
    <h3>Sample Images</h3>
    {sample_html}
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Train — fine-tune ViT for image classification
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    data_dir: flyte.io.Dir,
    model_name: str = "google/vit-base-patch16-224",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
) -> flyte.io.Dir:
    """Fine-tune a Vision Transformer for image classification."""
    import torch
    from torchvision import transforms
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    log.info(f"Training: model={model_name}")
    await flyte.report.replace.aio(_wrap_report(
        f"<h2>Loading model...</h2><p>{model_name}</p>"
        f"<p>Preparing dataset and initializing weights...</p>"
    ), do_flush=True)

    # Load data
    data_path = await data_dir.download()
    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    label_names = meta["label_names"]
    num_classes = meta["num_classes"]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    log.info(f"Classes: {label_names}")

    # Load image processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {trainable_params:,} / {total_params:,}")

    # Build torch datasets
    from PIL import Image as PILImage
    from torch.utils.data import Dataset

    img_size = processor.size.get("height", 224)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std,
        ),
    ])

    class ImageClassificationDataset(Dataset):
        def __init__(self, root_dir, transform):
            self.samples = []
            self.transform = transform
            for cls_idx, cls_name in enumerate(label_names):
                cls_dir = os.path.join(root_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(cls_dir, fname), cls_idx))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = PILImage.open(path).convert("RGB")
            pixel_values = self.transform(img)
            return {"pixel_values": pixel_values, "labels": label}

    train_ds = ImageClassificationDataset(
        os.path.join(data_path, "train"), train_transform
    )
    val_ds = ImageClassificationDataset(
        os.path.join(data_path, "val"), val_transform
    )

    log.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Training metrics callback with live report updates
    training_log: list[dict] = []
    loop = asyncio.get_running_loop()

    cat_badges = " ".join(
        f'<span class="badge badge-info">{name}</span>' for name in label_names
    )

    def _build_training_report(max_steps: int) -> str:
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>{model_name}</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Images</div></div>
          <div class="stat"><div class="value">{len(val_ds)}</div><div class="label">Val Images</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
          <div class="stat"><div class="value">{num_classes}</div><div class="label">Classes</div></div>
          <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Trainable Params</div></div>
        </div>
        <p>Classes: {cat_badges}</p>
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
              <div style="background:#ede9fe;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#7c3aed;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
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
                colors=["#7c3aed"],
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

            if "lr" in training_log[0]:
                lr_chart = _make_line_chart(
                    data=training_log,
                    x_key="epoch",
                    y_keys=["lr"],
                    title="Learning Rate Schedule",
                    x_label="Epoch",
                    y_label="LR",
                    colors=["#4c1d95"],
                )
                charts_html += f'<div class="chart-container">{lr_chart}</div>'

        return _wrap_report(stats_html + charts_html)

    class MetricsCallback(TrainerCallback):
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
            training_log.append(entry)
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={entry['epoch']:.2f} loss={entry['loss']:.4f}"
            )
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
        learning_rate=learning_rate,
        weight_decay=1e-4,
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
        callbacks=[MetricsCallback()],
    )

    log.info("Starting training...")
    await asyncio.to_thread(trainer.train)
    log.info("Training complete.")

    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)

    # Copy meta.json for downstream tasks
    shutil.copy2(os.path.join(data_path, "meta.json"), os.path.join(save_dir, "meta.json"))
    log.info(f"Model saved to {save_dir}")

    # Final training report
    stats_html = f"""
    <h2>Training Complete</h2>
    <h3>{model_name}</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train Images</div></div>
      <div class="stat"><div class="value">{len(val_ds)}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
      <div class="stat"><div class="value">{batch_size}</div><div class="label">Batch Size</div></div>
      <div class="stat"><div class="value">{num_classes}</div><div class="label">Classes</div></div>
      <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Trainable Params</div></div>
    </div>
    <p>Classes: {cat_badges}</p>
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
            colors=["#7c3aed"],
            x_range_override=epoch_range,
        )
        charts_html += f'<div class="chart-container">{loss_chart}</div>'

    if training_log and "lr" in training_log[0]:
        lr_chart = _make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["lr"],
            title="Learning Rate Schedule",
            x_label="Epoch",
            y_label="LR",
            colors=["#4c1d95"],
            x_range_override=(0, epochs),
        )
        charts_html += f'<div class="chart-container">{lr_chart}</div>'

    await flyte.report.replace.aio(_wrap_report(stats_html + charts_html), do_flush=True)

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Task 3: Evaluate — per-class metrics and confusion matrix
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
) -> str:
    """Evaluate the fine-tuned model on the validation set."""
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Evaluation</h2><p>Loading model and scoring validation images...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    label_names = meta["label_names"]
    num_classes = meta["num_classes"]

    processor = AutoImageProcessor.from_pretrained(ft_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(ft_path).to(device)
    model.eval()

    img_size = processor.size.get("height", 224)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Collect predictions
    all_preds = []
    all_labels = []
    all_probs = []

    val_dir = os.path.join(data_path, "val")
    for cls_idx, cls_name in enumerate(label_names):
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = PILImage.open(os.path.join(cls_dir, fname)).convert("RGB")
            pixel_values = val_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(cls_idx)
            all_probs.append(probs[0].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total = len(all_labels)

    # Overall accuracy
    accuracy = (all_preds == all_labels).sum() / total if total > 0 else 0

    # Per-class metrics
    per_class = []
    for cls_idx, cls_name in enumerate(label_names):
        mask_true = all_labels == cls_idx
        mask_pred = all_preds == cls_idx

        tp = (mask_true & mask_pred).sum()
        fp = (~mask_true & mask_pred).sum()
        fn = (mask_true & ~mask_pred).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        cls_acc = (mask_true & mask_pred).sum() / mask_true.sum() if mask_true.sum() > 0 else 0.0

        per_class.append({
            "class": cls_name,
            "accuracy": float(cls_acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(mask_true.sum()),
        })

    # Macro F1
    macro_f1 = np.mean([c["f1"] for c in per_class]) if per_class else 0.0

    # Confusion matrix
    conf_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1

    # Build report
    # Stat grid
    stats_html = f"""
    <h2>Evaluation Results</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{total}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value highlight">{accuracy:.1%}</div><div class="label">Overall Accuracy</div></div>
      <div class="stat"><div class="value highlight">{macro_f1:.3f}</div><div class="label">Macro F1</div></div>
      <div class="stat"><div class="value">{num_classes}</div><div class="label">Classes</div></div>
    </div>
    """

    # Per-class accuracy bar chart
    acc_chart = _make_bar_chart(
        labels=[c["class"] for c in per_class],
        series={"Accuracy": [c["accuracy"] for c in per_class]},
        title="Per-Class Accuracy",
        colors=["#7c3aed"],
        y_max_cap=1.0,
    )

    # Confusion matrix heatmap
    heatmap = _make_heatmap(
        matrix=conf_matrix,
        row_labels=label_names,
        col_labels=label_names,
        title="Confusion Matrix",
        color_scale="purple",
        width=max(400, 80 * num_classes + 130),
        height=max(400, 80 * num_classes + 130),
    )

    # Per-class metrics table
    table_rows = ""
    for c in per_class:
        f1_badge = (
            f'<span class="badge badge-success">{c["f1"]:.3f}</span>'
            if c["f1"] >= 0.7
            else f'<span class="badge badge-danger">{c["f1"]:.3f}</span>'
        )
        table_rows += f"""
        <tr>
          <td><span class="badge badge-info">{c['class']}</span></td>
          <td>{c['accuracy']:.1%}</td>
          <td>{c['precision']:.3f}</td>
          <td>{c['recall']:.3f}</td>
          <td>{f1_badge}</td>
          <td>{c['support']}</td>
        </tr>"""

    table_html = f"""
    <table>
      <tr><th>Class</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>
      {table_rows}
    </table>
    """

    eval_html = f"""
    {stats_html}
    <div class="chart-container">{acc_chart}</div>
    <h3>Confusion Matrix</h3>
    <div class="chart-container">{heatmap}</div>
    <h3>Per-Class Metrics</h3>
    {table_html}
    <div class="note">
      <b>Accuracy</b> = correct predictions / total for that class.
      <b>Precision</b> = true positives / (true positives + false positives).
      <b>Recall</b> = true positives / (true positives + false negatives).
      <b>F1</b> = harmonic mean of precision and recall.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(eval_html), do_flush=True)

    metrics_out = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "num_val": total,
        "per_class": per_class,
        "confusion_matrix": conf_matrix,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json.dumps(metrics_out)


# ------------------------------------------------------------------
# Task 4: Inference demo — predictions on sample images
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def inference_demo(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    max_images: int = 12,
    metrics_json: str = "{}",
) -> str:
    """Run predictions on sample validation images and render a visual report."""
    import numpy as np
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    log.info("Starting inference demo...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Inference Demo</h2><p>Running predictions on sample images...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    label_names = meta["label_names"]
    num_classes = meta["num_classes"]

    processor = AutoImageProcessor.from_pretrained(ft_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(ft_path).to(device)
    model.eval()

    img_size = processor.size.get("height", 224)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])

    # Gather sample images from each class
    val_dir = os.path.join(data_path, "val")
    samples = []
    samples_per_class = max(1, max_images // num_classes)

    rng = random.Random(42)
    for cls_idx, cls_name in enumerate(label_names):
        cls_dir = os.path.join(val_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        files = [f for f in sorted(os.listdir(cls_dir))
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        rng.shuffle(files)
        for fname in files[:samples_per_class]:
            samples.append((os.path.join(cls_dir, fname), cls_idx, cls_name))

    rng.shuffle(samples)
    samples = samples[:max_images]

    # Run predictions
    results = []
    for path, true_idx, true_name in samples:
        img = PILImage.open(path).convert("RGB")
        pixel_values = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_name = label_names[pred_idx]
        confidence = float(probs[pred_idx])
        correct = pred_idx == true_idx

        results.append({
            "image": img,
            "true_label": true_name,
            "pred_label": pred_name,
            "confidence": confidence,
            "correct": correct,
            "probs": probs,
        })

    # Build report
    num_correct = sum(1 for r in results if r["correct"])
    demo_accuracy = num_correct / len(results) if results else 0
    confidences = [r["confidence"] for r in results]
    most_confident = max(results, key=lambda r: r["confidence"]) if results else None
    least_confident = min(results, key=lambda r: r["confidence"]) if results else None

    # Parse eval metrics if provided
    eval_metrics = json.loads(metrics_json)
    overall_acc = eval_metrics.get("accuracy", None)
    macro_f1 = eval_metrics.get("macro_f1", None)

    extra_stats = ""
    if overall_acc is not None:
        extra_stats += f'<div class="stat"><div class="value highlight">{overall_acc:.1%}</div><div class="label">Overall Accuracy</div></div>'
    if macro_f1 is not None:
        extra_stats += f'<div class="stat"><div class="value highlight">{macro_f1:.3f}</div><div class="label">Macro F1</div></div>'

    stats_html = f"""
    <h2>Inference Demo</h2>
    <h3>Cell Microscopy Classification Results</h3>
    <div class="stat-grid">
      {extra_stats}
      <div class="stat"><div class="value">{len(results)}</div><div class="label">Images Shown</div></div>
      <div class="stat"><div class="value">{num_correct}/{len(results)}</div><div class="label">Correct</div></div>
      <div class="stat"><div class="value">{demo_accuracy:.0%}</div><div class="label">Demo Accuracy</div></div>
    </div>
    """

    # Image grid with predictions
    grid_html = '<div class="image-grid">'
    for r in results:
        data_uri = _img_to_data_uri(r["image"], max_size=400)
        correct_class = "badge-success" if r["correct"] else "badge-danger"
        correct_text = "Correct" if r["correct"] else "Wrong"
        conf_pct = r["confidence"] * 100

        # Confidence bar (SVG)
        bar_color = "#059669" if r["correct"] else "#dc2626"
        conf_bar = (
            f'<svg width="100%" height="18" style="margin-top:4px;">'
            f'<rect width="100%" height="18" fill="#ede9fe" rx="4"/>'
            f'<rect width="{conf_pct:.1f}%" height="18" fill="{bar_color}" rx="4"/>'
            f'<text x="50%" y="13" text-anchor="middle" font-size="11" '
            f'fill="#1a1a2e" font-weight="600">{conf_pct:.1f}%</text>'
            f'</svg>'
        )

        grid_html += f'''
        <div class="image-card">
          <img src="{data_uri}" alt="{r['true_label']}" />
          <div class="caption">
            <div style="margin-bottom:4px;">
              <span class="badge {correct_class}">{correct_text}</span>
            </div>
            <div style="margin-bottom:2px;">
              <b>True:</b> <span class="badge badge-info">{r['true_label']}</span>
            </div>
            <div style="margin-bottom:4px;">
              <b>Pred:</b> <span class="badge badge-info">{r['pred_label']}</span>
            </div>
            {conf_bar}
          </div>
        </div>'''

    grid_html += "</div>"

    # Summary section
    summary_html = ""
    if most_confident and least_confident:
        mc_badge = "badge-success" if most_confident["correct"] else "badge-danger"
        lc_badge = "badge-success" if least_confident["correct"] else "badge-danger"
        summary_html = f"""
        <div class="card">
          <h3>Prediction Summary</h3>
          <p><b>Most confident:</b> {most_confident['pred_label']}
            ({most_confident['confidence']:.1%})
            <span class="badge {mc_badge}">{'Correct' if most_confident['correct'] else 'Wrong'}</span>
          </p>
          <p><b>Least confident:</b> {least_confident['pred_label']}
            ({least_confident['confidence']:.1%})
            <span class="badge {lc_badge}">{'Correct' if least_confident['correct'] else 'Wrong'}</span>
          </p>
          <p><b>Avg confidence:</b> {np.mean(confidences):.1%}</p>
        </div>
        """

    demo_html = stats_html + grid_html + summary_html

    await flyte.report.replace.aio(_wrap_report(demo_html), do_flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json.dumps({
        "num_images": len(results),
        "num_correct": num_correct,
        "demo_accuracy": float(demo_accuracy),
        "avg_confidence": float(np.mean(confidences)) if confidences else 0,
    })


# ------------------------------------------------------------------
# Pipeline — chains all 4 tasks
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    dataset_name: str = "ehottl/blood_dataset",
    model_name: str = "google/vit-base-patch16-224",
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_fraction: float = 0.2,
    demo_images: int = 12,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end cell microscopy image classification pipeline.

    Returns the fine-tuned model directory and a JSON summary of metrics.

    1. Download dataset from HuggingFace and split train/val
    2. Fine-tune a Vision Transformer (ViT) for classification
    3. Evaluate with per-class metrics and confusion matrix
    4. Inference demo with confidence visualizations
    """
    log.info(f"Pipeline: {model_name} | dataset={dataset_name}")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Preparing Data", "Fine-tuning ViT", "Evaluating", "Inference Demo"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#059669;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#7c3aed;">&#9679;</span>'
            else:
                icon = '<span style="color:#a78bfa;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Cell Microscopy Classification Pipeline</h2>
        <p><b>Model:</b> {model_name} | <b>Dataset:</b> {dataset_name}</p>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Step 1: Prepare data
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Downloading and splitting dataset...")),
        do_flush=True,
    )
    data_dir = await prepare_data(
        dataset_name=dataset_name,
        val_fraction=val_fraction,
    )

    # Step 2: Train
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Fine-tuning Vision Transformer...")),
        do_flush=True,
    )
    finetuned_dir = await train(
        data_dir=data_dir,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Running evaluation on validation set...")),
        do_flush=True,
    )
    metrics_json = await evaluate(finetuned_dir, data_dir)
    metrics = json.loads(metrics_json)

    # Step 4: Inference demo
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating inference visualizations...")),
        do_flush=True,
    )
    demo_json = await inference_demo(
        finetuned_dir, data_dir, demo_images,
        metrics_json=metrics_json,
    )

    # Final pipeline report
    accuracy = metrics.get("accuracy", 0)
    macro_f1 = metrics.get("macro_f1", 0)
    num_val = metrics.get("num_val", 0)
    demo = json.loads(demo_json)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <h3>{model_name} on {dataset_name}</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{num_val}</div><div class="label">Val Images</div></div>
      <div class="stat"><div class="value highlight">{accuracy:.1%}</div><div class="label">Overall Accuracy</div></div>
      <div class="stat"><div class="value highlight">{macro_f1:.3f}</div><div class="label">Macro F1</div></div>
      <div class="stat"><div class="value">{demo.get('avg_confidence', 0):.1%}</div><div class="label">Avg Confidence</div></div>
    </div>
    <div class="card">
      <b>Configuration:</b> {epochs} epochs | LR {learning_rate} | Batch size {batch_size} |
      Val fraction {val_fraction} | {demo_images} demo images
    </div>
    <div class="note">
      Pipeline ran: prepare_data &#8594; train &#8594; evaluate &#8594; inference_demo.
      Check individual task reports for detailed charts, confusion matrix, and prediction visualizations.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info(f"Pipeline complete. Accuracy: {accuracy:.1%}, F1: {macro_f1:.3f}")
    return finetuned_dir, json.dumps({"metrics": metrics, "demo": demo})
