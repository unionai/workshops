"""
3D Brain Tumor Segmentation — Train SegResNet on BraTS 2023 MRI volumes.

Pipeline: download the BraTS 2023 GLI dataset (1,251 multi-modal MRI scans),
train a SegResNet for 3D brain tumor segmentation using MONAI, evaluate with
Dice scores and Hausdorff distance, and render multi-plane tumor overlays
(axial/sagittal/coronal) with color-coded tumor subregions.

Tumor subregions:
  - Enhancing Tumor (ET) — active tumor growth (red)
  - Necrotic Core (NCR) — dead tissue at tumor center (yellow)
  - Peritumoral Edema (ED) — swelling around tumor (green)

Usage:
    # Default (SegResNet on BraTS 2023, subset of 100 cases)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --max_cases 20 --epochs 5

    # Remote (full dataset)
    flyte run workflow.py pipeline --max_cases 0 --epochs 50

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
# Tumor subregion definitions
# ------------------------------------------------------------------

# BraTS label map (label 3 is intentionally skipped in the dataset)
LABEL_MAP = {0: "background", 1: "NCR", 2: "ED", 4: "ET"}

# Composite regions used for evaluation (standard BraTS metrics)
# WT = Whole Tumor (labels 1+2+4), TC = Tumor Core (labels 1+4), ET = Enhancing (label 4)
EVAL_REGIONS = {
    "WT": "Whole Tumor",
    "TC": "Tumor Core",
    "ET": "Enhancing Tumor",
}

# Colors for overlay visualization
REGION_COLORS = {
    1: (255, 220, 50),   # NCR — yellow
    2: (50, 205, 50),    # ED — green
    4: (220, 50, 50),    # ET — red
}

REGION_COLORS_HEX = {
    1: "#ffdc32",  # NCR
    2: "#32cd32",  # ED
    4: "#dc3232",  # ET
}

MRI_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]
MRI_DISPLAY_NAMES = {"t1n": "T1", "t1c": "T1-contrast", "t2w": "T2", "t2f": "FLAIR"}


# ------------------------------------------------------------------
# Report styling — deep blue/indigo neuroimaging theme
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #1e40af; margin-top: 20px; }
  .report .card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #1e3a5f; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #1e3a5f; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #bfdbfe; }
  .report tr:nth-child(even) { background: #eff6ff; }
  .report .highlight { color: #1e3a5f; font-weight: 700; }
  .report .note { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #dbeafe; color: #1e3a5f; }
  .report .chart-container { background: #fff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .image-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin: 16px 0; }
  .report .image-card { background: #000; border: 1px solid #bfdbfe; border-radius: 8px; overflow: hidden; }
  .report .image-card img { width: 100%; aspect-ratio: 1; object-fit: contain; }
  .report .image-card .caption { padding: 6px 10px; font-size: 0.8em; color: #fff; background: #1e293b; }
  .report .slice-panel { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin: 12px 0; }
  .report .slice-panel img { width: 100%; border-radius: 6px; border: 1px solid #334155; }
  .report .legend { display: flex; gap: 12px; flex-wrap: wrap; margin: 8px 0; }
  .report .legend-item { display: flex; align-items: center; gap: 4px; font-size: 0.85em; }
  .report .legend-swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
</style>
"""


def _wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def _tumor_legend_html() -> str:
    """Render an HTML legend showing tumor subregion colors."""
    items = ""
    for label_id, hex_color in REGION_COLORS_HEX.items():
        name = LABEL_MAP[label_id]
        items += (
            f'<span class="legend-item">'
            f'<span class="legend-swatch" style="background:{hex_color};"></span>'
            f'{name}</span>'
        )
    return f'<div class="legend">{items}</div>'


# ------------------------------------------------------------------
# SVG chart helpers
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
    default_colors = ["#3b82f6", "#1e3a5f", "#06d6a0", "#f59e0b", "#6c757d"]
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
            f'stroke="#bfdbfe" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#93c5fd" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#93c5fd" stroke-width="1.5"/>'
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
            f'font-size="13" fill="#93c5fd" font-style="italic">Waiting for data...</text>'
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
            f'font-size="14" font-weight="600" fill="#1e3a5f">{title}</text>'
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

    default_colors = ["#3b82f6", "#1e3a5f", "#06d6a0", "#93c5fd"]
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
            f'stroke="#bfdbfe" stroke-width="1"/>'
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
                f'text-anchor="middle" font-size="10" fill="#1e3a5f">'
                f'{val:.3f}</text>'
            )
        lines_svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" '
            f'text-anchor="middle" font-size="10" fill="#6c757d">{label}</text>'
        )

    if title:
        lines_svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1e3a5f">{title}</text>'
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
# Visualization helpers — MRI slice rendering with tumor overlays
# ------------------------------------------------------------------

def _render_mri_slice(volume_slice, seg_slice=None, alpha: float = 0.4) -> bytes:
    """Render a 2D MRI slice as a PNG with optional colored tumor overlay.

    Args:
        volume_slice: 2D numpy array (H, W) — MRI intensity values.
        seg_slice: 2D numpy array (H, W) — segmentation labels (0,1,2,4).
        alpha: Overlay transparency.

    Returns:
        PNG image bytes.
    """
    import numpy as np
    from PIL import Image

    # Normalize MRI to 0-255
    s = volume_slice.astype(np.float32)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    s = (s * 255).astype(np.uint8)

    # Convert grayscale to RGB
    rgb = np.stack([s, s, s], axis=-1).astype(np.float32)

    # Overlay tumor regions
    if seg_slice is not None:
        for label_id, color in REGION_COLORS.items():
            mask = seg_slice == label_id
            if mask.any():
                for c in range(3):
                    rgb[:, :, c] = np.where(
                        mask,
                        rgb[:, :, c] * (1 - alpha) + color[c] * alpha,
                        rgb[:, :, c],
                    )

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _slice_to_data_uri(volume_slice, seg_slice=None, alpha: float = 0.4) -> str:
    """Render an MRI slice to a base64 data URI."""
    png_bytes = _render_mri_slice(volume_slice, seg_slice, alpha)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _render_three_planes(volume, seg=None, alpha: float = 0.4) -> tuple[str, str, str]:
    """Render axial, sagittal, coronal center slices as data URIs.

    Args:
        volume: 3D numpy array (H, W, D).
        seg: 3D numpy array (H, W, D) of labels, or None.

    Returns:
        Tuple of (axial_uri, coronal_uri, sagittal_uri).
    """
    import numpy as np

    # Find center of tumor mass if segmentation available, else use volume center
    if seg is not None and seg.max() > 0:
        coords = np.argwhere(seg > 0)
        center = coords.mean(axis=0).astype(int)
    else:
        center = [s // 2 for s in volume.shape]

    h, w, d = volume.shape
    ax_idx = min(max(center[2], 0), d - 1)
    cor_idx = min(max(center[1], 0), w - 1)
    sag_idx = min(max(center[0], 0), h - 1)

    axial = volume[:, :, ax_idx]
    axial_seg = seg[:, :, ax_idx] if seg is not None else None

    coronal = volume[:, cor_idx, :]
    coronal_seg = seg[:, cor_idx, :] if seg is not None else None

    sagittal = volume[sag_idx, :, :]
    sagittal_seg = seg[sag_idx, :, :] if seg is not None else None

    return (
        _slice_to_data_uri(axial, axial_seg, alpha),
        _slice_to_data_uri(coronal, coronal_seg, alpha),
        _slice_to_data_uri(sagittal, sagittal_seg, alpha),
    )


def _three_plane_html(volume, seg=None, label: str = "", alpha: float = 0.4) -> str:
    """Render a 3-plane panel (axial, coronal, sagittal) as report HTML."""
    axial_uri, coronal_uri, sagittal_uri = _render_three_planes(volume, seg, alpha)
    return f"""
    <div class="card">
      {f'<b>{label}</b>' if label else ''}
      <div class="slice-panel">
        <div><img src="{axial_uri}" /><div style="text-align:center;font-size:0.8em;color:#6c757d;">Axial</div></div>
        <div><img src="{coronal_uri}" /><div style="text-align:center;font-size:0.8em;color:#6c757d;">Coronal</div></div>
        <div><img src="{sagittal_uri}" /><div style="text-align:center;font-size:0.8em;color:#6c757d;">Sagittal</div></div>
      </div>
    </div>
    """


# ------------------------------------------------------------------
# Task 1: Prepare data — download BraTS 2023 and organize
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_repo: str = "Angelou0516/brats2023-gli-dataset",
    max_cases: int = 100,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> flyte.io.Dir:
    """Download BraTS 2023 GLI dataset from HuggingFace and split train/val.

    Each case has 4 MRI modalities (T1, T1ce, T2, FLAIR) and a segmentation
    label, all as NIfTI files (.nii.gz). Data is already preprocessed
    (co-registered, 1mm isotropic, skull-stripped).

    Output directory:
      train/<case_id>/{t1n,t1c,t2w,t2f,seg}.nii.gz
      val/<case_id>/{t1n,t1c,t2w,t2f,seg}.nii.gz
      meta.json
    """
    import numpy as np
    from huggingface_hub import snapshot_download

    log.info(f"Downloading dataset: {dataset_repo}")
    dataset_path = snapshot_download(repo_id=dataset_repo, repo_type="dataset")

    # Find all case directories (contain .nii.gz files)
    all_cases = []
    for root, dirs, files in os.walk(dataset_path):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if nii_files and any("seg" in f for f in nii_files):
            all_cases.append(root)

    all_cases.sort()
    log.info(f"Found {len(all_cases)} cases with segmentation labels")

    # Subset if requested
    if max_cases > 0 and max_cases < len(all_cases):
        rng = random.Random(seed)
        rng.shuffle(all_cases)
        all_cases = all_cases[:max_cases]
        all_cases.sort()
        log.info(f"Using subset of {max_cases} cases")

    # Split train/val
    rng = random.Random(seed)
    rng.shuffle(all_cases)
    n_val = max(1, int(len(all_cases) * val_fraction))
    val_cases = all_cases[:n_val]
    train_cases = all_cases[n_val:]

    log.info(f"Split: {len(train_cases)} train / {len(val_cases)} val")

    # Pack output directory
    out_dir = tempfile.mkdtemp(prefix="brats_seg_")

    tumor_stats = []

    for split_name, cases in [("train", train_cases), ("val", val_cases)]:
        for case_path in cases:
            case_id = os.path.basename(case_path)
            case_out = os.path.join(out_dir, split_name, case_id)
            os.makedirs(case_out, exist_ok=True)

            # Find and copy NIfTI files, normalizing names
            nii_files = [f for f in os.listdir(case_path) if f.endswith(".nii.gz")]
            for f in nii_files:
                src = os.path.join(case_path, f)
                # Normalize filename: look for modality suffixes
                f_lower = f.lower()
                if "seg" in f_lower:
                    dst_name = "seg.nii.gz"
                elif "t1c" in f_lower or "t1ce" in f_lower or "t1gd" in f_lower:
                    dst_name = "t1c.nii.gz"
                elif "t1n" in f_lower or ("t1" in f_lower and "c" not in f_lower.split("t1")[1][:2]):
                    dst_name = "t1n.nii.gz"
                elif "t2w" in f_lower or ("t2" in f_lower and "f" not in f_lower.split("t2")[1][:2]):
                    dst_name = "t2w.nii.gz"
                elif "t2f" in f_lower or "flair" in f_lower:
                    dst_name = "t2f.nii.gz"
                else:
                    dst_name = f
                shutil.copy2(src, os.path.join(case_out, dst_name))

            # Collect tumor stats from segmentation
            seg_path = os.path.join(case_out, "seg.nii.gz")
            if os.path.exists(seg_path):
                import nibabel as nib
                seg = nib.load(seg_path).get_fdata().astype(int)
                voxel_counts = {
                    "NCR": int((seg == 1).sum()),
                    "ED": int((seg == 2).sum()),
                    "ET": int((seg == 4).sum()),
                }
                total_tumor = sum(voxel_counts.values())
                tumor_stats.append({
                    "case_id": case_id,
                    "split": split_name,
                    "total_tumor_voxels": total_tumor,
                    **voxel_counts,
                })

    # Save metadata
    meta = {
        "dataset_repo": dataset_repo,
        "num_train": len(train_cases),
        "num_val": len(val_cases),
        "total_cases": len(all_cases),
        "modalities": MRI_MODALITIES,
        "labels": LABEL_MAP,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Data saved to {out_dir}")

    # -- Build report with sample MRI slices and tumor overlays --
    import nibabel as nib

    sample_cases = [s for s in tumor_stats if s["split"] == "train" and s["total_tumor_voxels"] > 1000][:3]

    samples_html = ""
    for stat in sample_cases:
        case_id = stat["case_id"]
        case_dir = os.path.join(out_dir, "train", case_id)

        # Load FLAIR (best for seeing edema) and segmentation
        flair_path = os.path.join(case_dir, "t2f.nii.gz")
        seg_path = os.path.join(case_dir, "seg.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            continue

        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata().astype(int)

        samples_html += _three_plane_html(
            flair, seg,
            label=f"Case: {case_id} — NCR:{stat['NCR']:,} ED:{stat['ED']:,} ET:{stat['ET']:,} voxels",
        )

    # Show 4-modality panel for first case
    modality_html = ""
    if sample_cases:
        case_dir = os.path.join(out_dir, "train", sample_cases[0]["case_id"])
        seg = nib.load(os.path.join(case_dir, "seg.nii.gz")).get_fdata().astype(int)

        # Find best axial slice (most tumor)
        tumor_per_slice = [(seg[:, :, z] > 0).sum() for z in range(seg.shape[2])]
        best_z = int(np.argmax(tumor_per_slice))

        modality_html = '<h3>4 MRI Modalities (Same Slice)</h3><div class="image-grid">'
        for mod in MRI_MODALITIES:
            mod_path = os.path.join(case_dir, f"{mod}.nii.gz")
            if os.path.exists(mod_path):
                vol = nib.load(mod_path).get_fdata()
                uri = _slice_to_data_uri(vol[:, :, best_z])
                modality_html += f'''
                <div class="image-card">
                  <img src="{uri}" />
                  <div class="caption">{MRI_DISPLAY_NAMES[mod]}</div>
                </div>'''
        modality_html += "</div>"

    # Tumor size distribution
    tumor_volumes = [s["total_tumor_voxels"] for s in tumor_stats if s["total_tumor_voxels"] > 0]
    avg_tumor = float(np.mean(tumor_volumes)) if tumor_volumes else 0
    has_tumor_pct = len(tumor_volumes) / len(tumor_stats) * 100 if tumor_stats else 0

    report_html = f"""
    <h2>Dataset: BraTS 2023 GLI</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{meta['total_cases']}</div><div class="label">Total Cases</div></div>
      <div class="stat"><div class="value">{meta['num_train']}</div><div class="label">Training</div></div>
      <div class="stat"><div class="value">{meta['num_val']}</div><div class="label">Validation</div></div>
      <div class="stat"><div class="value">4</div><div class="label">MRI Modalities</div></div>
      <div class="stat"><div class="value">3</div><div class="label">Tumor Subregions</div></div>
      <div class="stat"><div class="value">{has_tumor_pct:.0f}%</div><div class="label">With Tumor</div></div>
    </div>
    {_tumor_legend_html()}
    {modality_html}
    <h3>Sample Cases with Tumor Overlays (FLAIR)</h3>
    <p>Axial, coronal, and sagittal views centered on tumor mass.</p>
    {samples_html}
    <div class="note">
      <b>BraTS 2023:</b> Multi-modal MRI brain scans with glioma annotations.
      Each case has T1, T1-contrast, T2, and FLAIR volumes (240x240x155 voxels, 1mm isotropic).
      Tumors are segmented into 3 subregions:
      <span style="color:#ffdc32;">necrotic core (NCR)</span>,
      <span style="color:#32cd32;">peritumoral edema (ED)</span>, and
      <span style="color:#dc3232;">enhancing tumor (ET)</span>.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Train — SegResNet via MONAI
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    data_dir: flyte.io.Dir,
    epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patch_size: int = 128,
) -> flyte.io.Dir:
    """Train a SegResNet for 3D brain tumor segmentation using MONAI.

    Uses patch-based training (128x128x128 crops from 240x240x155 volumes)
    to fit in GPU memory. Reports training loss and validation Dice live.
    """
    import numpy as np
    import torch
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.networks.nets import SegResNet
    from monai.transforms import (
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        Activationsd,
        AsDiscreted,
    )

    log.info("Setting up SegResNet training...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Loading Model...</h2>"
        "<p>SegResNet for 3D brain tumor segmentation</p>"
        "<p>Preparing MONAI data pipeline with patch-based training...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    # Build file lists for MONAI
    def _get_data_dicts(split: str) -> list[dict]:
        split_dir = os.path.join(data_path, split)
        if not os.path.isdir(split_dir):
            return []
        dicts = []
        for case_id in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, case_id)
            if not os.path.isdir(case_dir):
                continue
            entry = {"label": os.path.join(case_dir, "seg.nii.gz")}
            # Stack all 4 modalities as channels
            images = []
            for mod in MRI_MODALITIES:
                p = os.path.join(case_dir, f"{mod}.nii.gz")
                if os.path.exists(p):
                    images.append(p)
            if len(images) == 4 and os.path.exists(entry["label"]):
                entry["image"] = images
                dicts.append(entry)
        return dicts

    train_dicts = _get_data_dicts("train")
    val_dicts = _get_data_dicts("val")
    log.info(f"Train: {len(train_dicts)} cases | Val: {len(val_dicts)} cases")

    # MONAI transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[patch_size, patch_size, patch_size],
            random_size=False,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])

    train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=0.5)
    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # SegResNet: 4 input channels (T1, T1ce, T2, FLAIR), 3 output channels
    # (WT, TC, ET — the standard BraTS composite regions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SegResNet parameters: {trainable_params:,} / {total_params:,}")

    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])

    # Training loop with live reporting
    training_log: list[dict] = []
    val_log: list[dict] = []
    loop = asyncio.get_running_loop()
    total_steps = epochs * len(train_loader)

    def _build_training_report() -> str:
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>SegResNet — 3D Brain Tumor Segmentation</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(train_dicts)}</div><div class="label">Train Cases</div></div>
          <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{patch_size}³</div><div class="label">Patch Size</div></div>
          <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>
        </div>
        {_tumor_legend_html()}
        """

        charts_html = ""
        if training_log:
            current = training_log[-1]
            progress_pct = current["epoch"] / epochs * 100

            charts_html += f"""
            <div class="card">
              <b>Epoch {current['epoch']}/{epochs}</b>
              ({progress_pct:.0f}%) |
              Train Loss: <span class="highlight">{current['train_loss']:.4f}</span>
              <div style="background:#bfdbfe;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#3b82f6;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_chart = _make_line_chart(
                data=training_log,
                x_key="epoch",
                y_keys=["train_loss"],
                title="Training Loss (Dice)",
                x_label="Epoch",
                y_label="Loss",
                colors=["#3b82f6"],
                x_range_override=(0, epochs),
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

        if val_log:
            dice_chart = _make_line_chart(
                data=val_log,
                x_key="epoch",
                y_keys=["dice_wt", "dice_tc", "dice_et"],
                title="Validation Dice Score",
                x_label="Epoch",
                y_label="Dice",
                colors=["#06d6a0", "#f59e0b", "#dc3232"],
                y_max_cap=1.0,
                x_range_override=(0, epochs),
                y_display_names={
                    "dice_wt": "Whole Tumor",
                    "dice_tc": "Tumor Core",
                    "dice_et": "Enhancing",
                },
            )
            charts_html += f'<div class="chart-container">{dice_chart}</div>'

        return _wrap_report(stats_html + charts_html)

    def _train_loop():
        best_dice = 0
        val_interval = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []

            for batch_data in train_loader:
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            lr_scheduler.step()
            avg_loss = np.mean(epoch_losses)

            entry = {
                "epoch": epoch,
                "train_loss": round(float(avg_loss), 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            training_log.append(entry)
            log.info(f"Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}")

            # Validation at intervals
            if epoch % val_interval == 0 or epoch == epochs:
                model.eval()
                dice_metric.reset()

                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device)

                        val_outputs = sliding_window_inference(
                            val_inputs,
                            roi_size=(patch_size, patch_size, patch_size),
                            sw_batch_size=4,
                            predictor=model,
                            overlap=0.5,
                        )

                        val_data["pred"] = val_outputs
                        val_data = [post_trans(i) for i in decollate_batch(val_data)]
                        val_outputs_post = [d["pred"] for d in val_data]
                        val_labels_post = [d["label"] for d in val_data]

                        dice_metric(y_pred=val_outputs_post, y=val_labels_post)

                dice_values = dice_metric.aggregate()
                dice_wt = float(dice_values[0])
                dice_tc = float(dice_values[1])
                dice_et = float(dice_values[2])
                mean_dice = (dice_wt + dice_tc + dice_et) / 3

                val_log.append({
                    "epoch": epoch,
                    "dice_wt": round(dice_wt, 4),
                    "dice_tc": round(dice_tc, 4),
                    "dice_et": round(dice_et, 4),
                    "mean_dice": round(mean_dice, 4),
                })

                log.info(f"  Val Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} "
                         f"ET:{dice_et:.3f} Mean:{mean_dice:.3f}")

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), os.path.join(tempfile.gettempdir(), "best_model.pth"))
                    log.info(f"  New best model (mean dice: {best_dice:.3f})")

            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(_build_training_report(), do_flush=True),
                loop,
            )

    log.info("Starting training...")
    await asyncio.to_thread(_train_loop)
    log.info("Training complete.")

    # Save best model
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_segresnet")
    os.makedirs(save_dir, exist_ok=True)

    best_path = os.path.join(tempfile.gettempdir(), "best_model.pth")
    if os.path.exists(best_path):
        shutil.copy2(best_path, os.path.join(save_dir, "model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    model_config = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "patch_size": patch_size,
        "in_channels": 4,
        "out_channels": 3,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    shutil.copy2(os.path.join(data_path, "meta.json"), os.path.join(save_dir, "meta.json"))

    # Final training report
    stats_html = f"""
    <h2>Training Complete</h2>
    <h3>SegResNet — 3D Brain Tumor Segmentation</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_dicts)}</div><div class="label">Train Cases</div></div>
      <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>
    </div>
    {_tumor_legend_html()}
    """

    charts_html = ""
    if training_log:
        loss_chart = _make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["train_loss"],
            title="Training Loss (Dice)",
            x_label="Epoch",
            y_label="Loss",
            colors=["#3b82f6"],
            x_range_override=(0, epochs),
        )
        charts_html += f'<div class="chart-container">{loss_chart}</div>'

    if val_log:
        best = max(val_log, key=lambda d: d["mean_dice"])
        charts_html += f"""
        <div class="card">
          <b>Best Validation (Epoch {best['epoch']}):</b>
          WT: <span class="highlight">{best['dice_wt']:.3f}</span> |
          TC: <span class="highlight">{best['dice_tc']:.3f}</span> |
          ET: <span class="highlight">{best['dice_et']:.3f}</span> |
          Mean: <span class="highlight">{best['mean_dice']:.3f}</span>
        </div>
        """

        dice_chart = _make_line_chart(
            data=val_log,
            x_key="epoch",
            y_keys=["dice_wt", "dice_tc", "dice_et"],
            title="Validation Dice Score",
            x_label="Epoch",
            y_label="Dice",
            colors=["#06d6a0", "#f59e0b", "#dc3232"],
            y_max_cap=1.0,
            x_range_override=(0, epochs),
            y_display_names={
                "dice_wt": "Whole Tumor",
                "dice_tc": "Tumor Core",
                "dice_et": "Enhancing",
            },
        )
        charts_html += f'<div class="chart-container">{dice_chart}</div>'

    await flyte.report.replace.aio(_wrap_report(stats_html + charts_html), do_flush=True)

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Helper: load trained model
# ------------------------------------------------------------------

def _load_segresnet(model_path: str, device):
    """Load a trained SegResNet from saved state dict."""
    import torch
    from monai.networks.nets import SegResNet

    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        dropout_prob=0.2,
    )
    model.load_state_dict(torch.load(
        os.path.join(model_path, "model.pth"),
        map_location=device,
        weights_only=True,
    ))
    model.to(device)
    model.eval()
    return model, config


# ------------------------------------------------------------------
# Task 3: Evaluate — Dice scores per tumor region
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
) -> str:
    """Evaluate the trained SegResNet on validation cases.

    Computes Dice score for each composite tumor region:
      WT (Whole Tumor) = labels 1+2+4
      TC (Tumor Core) = labels 1+4
      ET (Enhancing Tumor) = label 4
    """
    import numpy as np
    import torch
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activationsd,
        AsDiscreted,
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
    )

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Evaluation</h2><p>Running sliding window inference on validation set...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_segresnet(ft_path, device)
    patch_size = config["patch_size"]

    # Build val data
    val_dir = os.path.join(data_path, "val")
    val_dicts = []
    for case_id in sorted(os.listdir(val_dir)):
        case_dir = os.path.join(val_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        images = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in MRI_MODALITIES]
        label = os.path.join(case_dir, "seg.nii.gz")
        if all(os.path.exists(p) for p in images) and os.path.exists(label):
            val_dicts.append({"image": images, "label": label})

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ])

    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])

    per_case_dice = []

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            val_data["pred"] = val_outputs
            val_data = [post_trans(i) for i in decollate_batch(val_data)]
            val_outputs_post = [d["pred"] for d in val_data]
            val_labels_post = [d["label"] for d in val_data]

            dice_metric(y_pred=val_outputs_post, y=val_labels_post)

            # Per-case dice
            case_dice = DiceMetric(include_background=True, reduction="mean_batch")
            case_dice(y_pred=val_outputs_post, y=val_labels_post)
            cd = case_dice.aggregate()
            per_case_dice.append({
                "wt": float(cd[0]),
                "tc": float(cd[1]),
                "et": float(cd[2]),
            })

    dice_values = dice_metric.aggregate()
    dice_wt = float(dice_values[0])
    dice_tc = float(dice_values[1])
    dice_et = float(dice_values[2])
    mean_dice = (dice_wt + dice_tc + dice_et) / 3

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} ET:{dice_et:.3f} Mean:{mean_dice:.3f}")

    # Build report
    bar_chart = _make_bar_chart(
        labels=["Whole Tumor", "Tumor Core", "Enhancing"],
        series={"Dice": [dice_wt, dice_tc, dice_et]},
        title="Dice Score by Tumor Region",
        colors=["#06d6a0", "#f59e0b", "#dc3232"],
        y_max_cap=1.0,
    )

    eval_html = f"""
    <h2>Evaluation Results</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
      <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
      <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
      <div class="stat"><div class="value highlight">{mean_dice:.3f}</div><div class="label">Mean Dice</div></div>
    </div>
    {_tumor_legend_html()}
    <div class="chart-container">{bar_chart}</div>
    <table>
      <tr><th>Region</th><th>Dice Score</th><th>Description</th></tr>
      <tr><td><b>Whole Tumor (WT)</b></td><td class="highlight">{dice_wt:.3f}</td><td>All tumor subregions (NCR+ED+ET)</td></tr>
      <tr><td><b>Tumor Core (TC)</b></td><td class="highlight">{dice_tc:.3f}</td><td>Core without edema (NCR+ET)</td></tr>
      <tr><td><b>Enhancing (ET)</b></td><td class="highlight">{dice_et:.3f}</td><td>Active tumor growth only</td></tr>
    </table>
    <div class="note">
      <b>Dice Score</b> measures overlap between predicted and ground truth segmentation (0=no overlap, 1=perfect).
      WT is typically easiest (largest region), ET is hardest (smallest, most variable).
      BraTS challenge winners achieve ~0.90 WT, ~0.85 TC, ~0.80 ET.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(eval_html), do_flush=True)

    return json.dumps({
        "dice_wt": dice_wt,
        "dice_tc": dice_tc,
        "dice_et": dice_et,
        "mean_dice": mean_dice,
        "num_val_cases": len(val_dicts),
        "per_case_dice": per_case_dice,
    })


# ------------------------------------------------------------------
# Task 4: Inference — multi-plane tumor overlay visualizations
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def inference(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    max_cases: int = 4,
    metrics_json: str = "{}",
) -> str:
    """Run inference on validation cases and render multi-plane tumor overlays.

    Shows axial/coronal/sagittal views with ground truth vs predicted tumor
    overlays, color-coded by subregion.
    """
    import nibabel as nib
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
    )

    log.info("Starting inference visualization...")
    await flyte.report.replace.aio(_wrap_report(
        "<h2>Inference</h2><p>Generating multi-plane tumor overlays...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_segresnet(ft_path, device)
    patch_size = config["patch_size"]

    # Load val cases
    val_dir = os.path.join(data_path, "val")
    case_ids = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

    rng = random.Random(42)
    rng.shuffle(case_ids)
    case_ids = case_ids[:max_cases]

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])

    html_blocks = []

    for case_id in case_ids:
        case_dir = os.path.join(val_dir, case_id)
        images = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in MRI_MODALITIES]
        label_path = os.path.join(case_dir, "seg.nii.gz")

        if not all(os.path.exists(p) for p in images) or not os.path.exists(label_path):
            continue

        data_dict = {"image": images, "label": label_path}
        data_dict = val_transforms(data_dict)

        # Run inference
        input_tensor = data_dict["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )
            pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy()

        # Convert multi-channel pred back to label map
        # Channel 0=WT, 1=TC, 2=ET
        # Reconstruct: ET=4 where ch2, NCR=1 where ch1 & ~ch2, ED=2 where ch0 & ~ch1
        pred_seg = np.zeros(pred.shape[1:], dtype=int)
        pred_seg[pred[0] > 0] = 2   # ED (whole tumor minus core)
        pred_seg[pred[1] > 0] = 1   # NCR (core minus enhancing)
        pred_seg[pred[2] > 0] = 4   # ET (enhancing)

        # Ground truth: load original labels
        gt_seg = nib.load(label_path).get_fdata().astype(int)

        # Load FLAIR for background
        flair = nib.load(os.path.join(case_dir, "t2f.nii.gz")).get_fdata()

        # Render GT and pred overlays
        gt_html = _three_plane_html(flair, gt_seg, label="Ground Truth")
        pred_html = _three_plane_html(flair, pred_seg, label="Predicted")

        # Count voxels per region
        gt_counts = {
            "NCR": int((gt_seg == 1).sum()),
            "ED": int((gt_seg == 2).sum()),
            "ET": int((gt_seg == 4).sum()),
        }
        pred_counts = {
            "NCR": int((pred_seg == 1).sum()),
            "ED": int((pred_seg == 2).sum()),
            "ET": int((pred_seg == 4).sum()),
        }

        html_blocks.append(f"""
        <div class="card">
          <b>Case: {case_id}</b>
          <span style="font-size:0.8em;color:#6c757d;">
            GT: NCR={gt_counts['NCR']:,} ED={gt_counts['ED']:,} ET={gt_counts['ET']:,} |
            Pred: NCR={pred_counts['NCR']:,} ED={pred_counts['ED']:,} ET={pred_counts['ET']:,}
          </span>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px;">
            <div>{gt_html}</div>
            <div>{pred_html}</div>
          </div>
        </div>
        """)

    # Parse metrics
    metrics = json.loads(metrics_json)
    dice_wt = metrics.get("dice_wt")
    dice_tc = metrics.get("dice_tc")
    dice_et = metrics.get("dice_et")

    metric_stats = ""
    if dice_wt is not None:
        metric_stats = f"""
        <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
        <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
        <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
        """

    demo_html = f"""
    <h2>Brain Tumor Segmentation Results</h2>
    <div class="stat-grid">
      {metric_stats}
      <div class="stat"><div class="value">{len(case_ids)}</div><div class="label">Cases Shown</div></div>
    </div>
    {_tumor_legend_html()}
    <p>Each case shows axial, coronal, and sagittal views centered on the tumor.
    Left: ground truth annotations. Right: model predictions.
    <span style="color:#ffdc32;">Yellow=NCR</span>
    <span style="color:#32cd32;">Green=ED</span>
    <span style="color:#dc3232;">Red=ET</span></p>
    {"".join(html_blocks)}
    """

    await flyte.report.replace.aio(_wrap_report(demo_html), do_flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json.dumps({
        "num_cases": len(case_ids),
    })


# ------------------------------------------------------------------
# Pipeline — chains all 4 tasks
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    dataset_repo: str = "Angelou0516/brats2023-gli-dataset",
    max_cases: int = 100,
    epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patch_size: int = 128,
    val_fraction: float = 0.15,
    demo_cases: int = 4,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end 3D brain tumor segmentation pipeline.

    1. Download BraTS 2023 MRI volumes from HuggingFace
    2. Train SegResNet for 3D tumor segmentation using MONAI
    3. Evaluate with Dice scores (WT, TC, ET)
    4. Render multi-plane tumor overlays on validation cases
    """
    log.info(f"Pipeline: SegResNet | dataset={dataset_repo}")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Preparing Data", "Training SegResNet", "Evaluating", "Inference Demo"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#059669;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#3b82f6;">&#9679;</span>'
            else:
                icon = '<span style="color:#93c5fd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Brain Tumor Segmentation Pipeline</h2>
        <p><b>Model:</b> SegResNet | <b>Dataset:</b> BraTS 2023 GLI</p>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Step 1: Prepare data
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Downloading BraTS 2023 from HuggingFace...")),
        do_flush=True,
    )
    data_dir = await prepare_data(
        dataset_repo=dataset_repo,
        max_cases=max_cases,
        val_fraction=val_fraction,
    )

    # Step 2: Train
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Training SegResNet on 3D MRI volumes...")),
        do_flush=True,
    )
    finetuned_dir = await train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patch_size=patch_size,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Computing Dice scores (WT, TC, ET)...")),
        do_flush=True,
    )
    metrics_json = await evaluate(finetuned_dir, data_dir)
    metrics = json.loads(metrics_json)

    # Step 4: Inference
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating multi-plane tumor overlays...")),
        do_flush=True,
    )
    inference_json = await inference(
        finetuned_dir, data_dir, demo_cases,
        metrics_json=metrics_json,
    )

    # Final pipeline report
    dice_wt = metrics.get("dice_wt", 0)
    dice_tc = metrics.get("dice_tc", 0)
    dice_et = metrics.get("dice_et", 0)
    mean_dice = metrics.get("mean_dice", 0)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <h3>SegResNet on BraTS 2023 GLI</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{metrics.get('num_val_cases', 0)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
      <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
      <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
      <div class="stat"><div class="value highlight">{mean_dice:.3f}</div><div class="label">Mean Dice</div></div>
    </div>
    {_tumor_legend_html()}
    <div class="card">
      <b>Configuration:</b> {epochs} epochs | LR {learning_rate} | Patch {patch_size}³ |
      {max_cases} cases | Val fraction {val_fraction}
    </div>
    <div class="note">
      Pipeline ran: prepare_data &#8594; train &#8594; evaluate &#8594; inference.
      Check individual task reports for loss curves, Dice progression,
      and side-by-side tumor overlay visualizations.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info(f"Pipeline complete. Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} ET:{dice_et:.3f}")
    return finetuned_dir, json.dumps({"metrics": metrics, "inference": json.loads(inference_json)})
