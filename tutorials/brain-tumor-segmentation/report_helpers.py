"""
Report visualization helpers for brain tumor segmentation.

Contains: CSS theming, SVG chart generators, MRI slice rendering,
tumor overlay visualization, and interactive slice slider/player.
"""

import base64
import io
import uuid


# ------------------------------------------------------------------
# Tumor subregion definitions
# ------------------------------------------------------------------

LABEL_MAP = {0: "background", 1: "NCR", 2: "ED", 4: "ET"}

EVAL_REGIONS = {
    "WT": "Whole Tumor",
    "TC": "Tumor Core",
    "ET": "Enhancing Tumor",
}

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
  .report .slice-viewer { background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 12px; margin: 12px 0; }
  .report .slice-viewer img { width: 100%; border-radius: 4px; }
  .report .slice-controls { display: flex; align-items: center; gap: 8px; margin-top: 8px; }
  .report .slice-controls input[type="range"] { flex: 1; }
  .report .slice-controls button { background: #3b82f6; color: #fff; border: none; border-radius: 4px; padding: 4px 12px; cursor: pointer; font-size: 0.85em; }
  .report .slice-controls button:hover { background: #2563eb; }
  .report .slice-controls .slice-label { color: #94a3b8; font-size: 0.8em; min-width: 80px; text-align: center; }
</style>
"""


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def tumor_legend_html() -> str:
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

def make_line_chart(
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


def make_bar_chart(
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
# MRI slice rendering
# ------------------------------------------------------------------

def render_mri_slice(volume_slice, seg_slice=None, alpha: float = 0.4) -> bytes:
    """Render a 2D MRI slice as PNG bytes with optional colored tumor overlay."""
    import numpy as np
    from PIL import Image

    s = volume_slice.astype(np.float32)
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    s = (s * 255).astype(np.uint8)

    rgb = np.stack([s, s, s], axis=-1).astype(np.float32)

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


def slice_to_data_uri(volume_slice, seg_slice=None, alpha: float = 0.4) -> str:
    """Render an MRI slice to a base64 data URI."""
    png_bytes = render_mri_slice(volume_slice, seg_slice, alpha)
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def render_three_planes(volume, seg=None, alpha: float = 0.4) -> tuple[str, str, str]:
    """Render axial, coronal, sagittal center slices as data URIs."""
    import numpy as np

    if seg is not None and seg.max() > 0:
        coords = np.argwhere(seg > 0)
        center = coords.mean(axis=0).astype(int)
    else:
        center = [s // 2 for s in volume.shape]

    h, w, d = volume.shape
    ax_idx = min(max(center[2], 0), d - 1)
    cor_idx = min(max(center[1], 0), w - 1)
    sag_idx = min(max(center[0], 0), h - 1)

    return (
        slice_to_data_uri(volume[:, :, ax_idx], seg[:, :, ax_idx] if seg is not None else None, alpha),
        slice_to_data_uri(volume[:, cor_idx, :], seg[:, cor_idx, :] if seg is not None else None, alpha),
        slice_to_data_uri(volume[sag_idx, :, :], seg[sag_idx, :, :] if seg is not None else None, alpha),
    )


def three_plane_html(volume, seg=None, label: str = "", alpha: float = 0.4) -> str:
    """Render a 3-plane panel (axial, coronal, sagittal) as report HTML."""
    axial_uri, coronal_uri, sagittal_uri = render_three_planes(volume, seg, alpha)
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
# Interactive slice slider + play/pause control
# ------------------------------------------------------------------

def slice_viewer_html(
    volume,
    seg=None,
    axis: str = "axial",
    n_frames: int = 24,
    label: str = "",
    alpha: float = 0.4,
    fps: int = 6,
) -> str:
    """Render an interactive slice viewer with slider + play/pause button.

    Pre-renders N evenly spaced slices along the given axis as base64 PNGs,
    then embeds them in HTML/JS with:
      - Range slider to scrub through slices
      - Play/Pause button for auto-animation
      - Slice index label

    Args:
        volume: 3D numpy array (H, W, D).
        seg: 3D numpy array of labels, or None.
        axis: "axial" (z), "coronal" (y), or "sagittal" (x).
        n_frames: Number of slices to render.
        label: Title text above the viewer.
        alpha: Overlay transparency.
        fps: Playback speed (frames per second).

    Returns:
        HTML string with embedded JS viewer.
    """
    import numpy as np

    # Determine axis and slice range
    if axis == "axial":
        n_slices = volume.shape[2]
        get_slice = lambda i: (volume[:, :, i], seg[:, :, i] if seg is not None else None)
    elif axis == "coronal":
        n_slices = volume.shape[1]
        get_slice = lambda i: (volume[:, i, :], seg[:, i, :] if seg is not None else None)
    else:  # sagittal
        n_slices = volume.shape[0]
        get_slice = lambda i: (volume[i, :, :], seg[i, :, :] if seg is not None else None)

    # Find slice range with content (skip empty borders)
    if seg is not None:
        if axis == "axial":
            has_tumor = [int((seg[:, :, z] > 0).any()) for z in range(n_slices)]
        elif axis == "coronal":
            has_tumor = [int((seg[:, y, :] > 0).any()) for y in range(n_slices)]
        else:
            has_tumor = [int((seg[x, :, :] > 0).any()) for x in range(n_slices)]

        tumor_indices = [i for i, h in enumerate(has_tumor) if h]
        if tumor_indices:
            # Pad range around tumor region
            pad = max(5, n_slices // 10)
            start = max(0, tumor_indices[0] - pad)
            end = min(n_slices - 1, tumor_indices[-1] + pad)
        else:
            start, end = 0, n_slices - 1
    else:
        start, end = 0, n_slices - 1

    # Sample N evenly spaced slices
    indices = np.linspace(start, end, n_frames, dtype=int)

    # Render frames as base64
    frame_uris = []
    for idx in indices:
        vol_slice, seg_slice = get_slice(int(idx))
        frame_uris.append(slice_to_data_uri(vol_slice, seg_slice, alpha))

    # Generate unique ID for this viewer instance
    viewer_id = f"sv_{uuid.uuid4().hex[:8]}"

    # Build JS frame array
    frames_js = ",\n".join(f'"{uri}"' for uri in frame_uris)

    axis_label = axis.capitalize()

    return f"""
    <div class="slice-viewer">
      {f'<div style="color:#e2e8f0;font-weight:600;margin-bottom:8px;">{label}</div>' if label else ''}
      <img id="{viewer_id}_img" src="{frame_uris[0]}" style="width:100%;border-radius:4px;" />
      <div class="slice-controls">
        <button id="{viewer_id}_btn" onclick="
          var s = document.getElementById('{viewer_id}_state');
          if (s.dataset.playing === 'true') {{
            clearInterval(parseInt(s.dataset.timer));
            s.dataset.playing = 'false';
            document.getElementById('{viewer_id}_btn').textContent = 'Play';
          }} else {{
            s.dataset.playing = 'true';
            document.getElementById('{viewer_id}_btn').textContent = 'Pause';
            var t = setInterval(function() {{
              var slider = document.getElementById('{viewer_id}_slider');
              var v = parseInt(slider.value);
              v = (v + 1) % {n_frames};
              slider.value = v;
              document.getElementById('{viewer_id}_img').src = {viewer_id}_frames[v];
              document.getElementById('{viewer_id}_label').textContent = '{axis_label} slice ' + (v+1) + '/{n_frames}';
            }}, {1000 // fps});
            s.dataset.timer = t;
          }}
        ">Play</button>
        <input type="range" id="{viewer_id}_slider" min="0" max="{n_frames - 1}" value="0"
          oninput="
            document.getElementById('{viewer_id}_img').src = {viewer_id}_frames[this.value];
            document.getElementById('{viewer_id}_label').textContent = '{axis_label} slice ' + (parseInt(this.value)+1) + '/{n_frames}';
          "
        />
        <span id="{viewer_id}_label" class="slice-label">{axis_label} slice 1/{n_frames}</span>
      </div>
      <span id="{viewer_id}_state" data-playing="false" data-timer="0" style="display:none;"></span>
    </div>
    <script>
      var {viewer_id}_frames = [{frames_js}];
    </script>
    """


def dual_slice_viewer_html(
    volume,
    gt_seg,
    pred_seg,
    axis: str = "axial",
    n_frames: int = 24,
    label: str = "",
    alpha: float = 0.4,
    fps: int = 6,
) -> str:
    """Render side-by-side GT vs predicted slice viewers synced to one slider.

    Both viewers advance together when sliding or playing.
    """
    import numpy as np

    if axis == "axial":
        n_slices = volume.shape[2]
        get_vol = lambda i: volume[:, :, i]
        get_gt = lambda i: gt_seg[:, :, i]
        get_pred = lambda i: pred_seg[:, :, i]
    elif axis == "coronal":
        n_slices = volume.shape[1]
        get_vol = lambda i: volume[:, i, :]
        get_gt = lambda i: gt_seg[:, i, :]
        get_pred = lambda i: pred_seg[:, i, :]
    else:
        n_slices = volume.shape[0]
        get_vol = lambda i: volume[i, :, :]
        get_gt = lambda i: gt_seg[i, :, :]
        get_pred = lambda i: pred_seg[i, :, :]

    # Find tumor range from either seg
    combined = np.maximum(gt_seg, pred_seg)
    if axis == "axial":
        has_content = [int((combined[:, :, z] > 0).any()) for z in range(n_slices)]
    elif axis == "coronal":
        has_content = [int((combined[:, y, :] > 0).any()) for y in range(n_slices)]
    else:
        has_content = [int((combined[x, :, :] > 0).any()) for x in range(n_slices)]

    content_indices = [i for i, h in enumerate(has_content) if h]
    if content_indices:
        pad = max(5, n_slices // 10)
        start = max(0, content_indices[0] - pad)
        end = min(n_slices - 1, content_indices[-1] + pad)
    else:
        start, end = 0, n_slices - 1

    indices = np.linspace(start, end, n_frames, dtype=int)

    gt_uris = []
    pred_uris = []
    for idx in indices:
        i = int(idx)
        gt_uris.append(slice_to_data_uri(get_vol(i), get_gt(i), alpha))
        pred_uris.append(slice_to_data_uri(get_vol(i), get_pred(i), alpha))

    vid = f"dsv_{uuid.uuid4().hex[:8]}"
    gt_js = ",\n".join(f'"{u}"' for u in gt_uris)
    pred_js = ",\n".join(f'"{u}"' for u in pred_uris)
    axis_label = axis.capitalize()

    return f"""
    <div class="card">
      {f'<div style="font-weight:600;margin-bottom:8px;">{label}</div>' if label else ''}
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
        <div class="slice-viewer">
          <div style="color:#e2e8f0;font-size:0.85em;text-align:center;margin-bottom:4px;">Ground Truth</div>
          <img id="{vid}_gt" src="{gt_uris[0]}" style="width:100%;border-radius:4px;" />
        </div>
        <div class="slice-viewer">
          <div style="color:#e2e8f0;font-size:0.85em;text-align:center;margin-bottom:4px;">Predicted</div>
          <img id="{vid}_pred" src="{pred_uris[0]}" style="width:100%;border-radius:4px;" />
        </div>
      </div>
      <div class="slice-controls" style="margin-top:8px;">
        <button id="{vid}_btn" onclick="
          var s = document.getElementById('{vid}_state');
          if (s.dataset.playing === 'true') {{
            clearInterval(parseInt(s.dataset.timer));
            s.dataset.playing = 'false';
            document.getElementById('{vid}_btn').textContent = 'Play';
          }} else {{
            s.dataset.playing = 'true';
            document.getElementById('{vid}_btn').textContent = 'Pause';
            var t = setInterval(function() {{
              var sl = document.getElementById('{vid}_slider');
              var v = (parseInt(sl.value) + 1) % {n_frames};
              sl.value = v;
              document.getElementById('{vid}_gt').src = {vid}_gt_frames[v];
              document.getElementById('{vid}_pred').src = {vid}_pred_frames[v];
              document.getElementById('{vid}_label').textContent = '{axis_label} slice ' + (v+1) + '/{n_frames}';
            }}, {1000 // fps});
            s.dataset.timer = t;
          }}
        ">Play</button>
        <input type="range" id="{vid}_slider" min="0" max="{n_frames - 1}" value="0"
          oninput="
            document.getElementById('{vid}_gt').src = {vid}_gt_frames[this.value];
            document.getElementById('{vid}_pred').src = {vid}_pred_frames[this.value];
            document.getElementById('{vid}_label').textContent = '{axis_label} slice ' + (parseInt(this.value)+1) + '/{n_frames}';
          "
        />
        <span id="{vid}_label" class="slice-label">{axis_label} slice 1/{n_frames}</span>
      </div>
      <span id="{vid}_state" data-playing="false" data-timer="0" style="display:none;"></span>
    </div>
    <script>
      var {vid}_gt_frames = [{gt_js}];
      var {vid}_pred_frames = [{pred_js}];
    </script>
    """


# ------------------------------------------------------------------
# Animated GIF generation
# ------------------------------------------------------------------

def render_slice_gif(
    volume,
    seg=None,
    axis: str = "axial",
    n_frames: int = 30,
    alpha: float = 0.4,
    duration_ms: int = 150,
) -> bytes:
    """Render an animated GIF sweeping through MRI slices with tumor overlay.

    Returns GIF bytes that can be saved to a file.
    """
    import numpy as np
    from PIL import Image

    if axis == "axial":
        n_slices = volume.shape[2]
        get_slice = lambda i: (volume[:, :, i], seg[:, :, i] if seg is not None else None)
    elif axis == "coronal":
        n_slices = volume.shape[1]
        get_slice = lambda i: (volume[:, i, :], seg[:, i, :] if seg is not None else None)
    else:
        n_slices = volume.shape[0]
        get_slice = lambda i: (volume[i, :, :], seg[i, :, :] if seg is not None else None)

    # Find tumor range
    if seg is not None:
        if axis == "axial":
            has_tumor = [int((seg[:, :, z] > 0).any()) for z in range(n_slices)]
        elif axis == "coronal":
            has_tumor = [int((seg[:, y, :] > 0).any()) for y in range(n_slices)]
        else:
            has_tumor = [int((seg[x, :, :] > 0).any()) for x in range(n_slices)]

        tumor_indices = [i for i, h in enumerate(has_tumor) if h]
        if tumor_indices:
            pad = max(5, n_slices // 10)
            start = max(0, tumor_indices[0] - pad)
            end = min(n_slices - 1, tumor_indices[-1] + pad)
        else:
            start, end = 0, n_slices - 1
    else:
        start, end = 0, n_slices - 1

    indices = np.linspace(start, end, n_frames, dtype=int)

    frames = []
    for idx in indices:
        png_bytes = render_mri_slice(*get_slice(int(idx)), alpha=alpha)
        frames.append(Image.open(io.BytesIO(png_bytes)))

    buf = io.BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    return buf.getvalue()
