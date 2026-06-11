"""
Report visualization helpers for nuclei instance segmentation.

Contains: CSS theming, SVG chart generators, class-colored mask overlays,
confusion matrix, confidence heatmap, and interactive per-class toggle viewer.
"""

import base64
import io


# ------------------------------------------------------------------
# Nucleus class definitions
# ------------------------------------------------------------------

CLASS_NAMES = ["background", "neoplastic", "inflammatory", "connective", "dead", "epithelial"]
NUM_CLASSES = len(CLASS_NAMES)  # 6 (including background)

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


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def class_legend_html() -> str:
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

    def sx(v): return ml + (v - x_min) / x_range * cw
    def sy(v): return mt + ch - (v - y_min_plot) / y_range * ch

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#ccfbf1" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:.3f}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#99f6e4" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#99f6e4" stroke-width="1.5"/>')

    if x_vals:
        n_x_ticks = min(len(data), 10)
        step = max(1, len(data) // n_x_ticks)
        for i in range(0, len(data), step):
            px = sx(x_vals[i])
            lines.append(f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" font-size="11" fill="#6c757d">{x_vals[i]:.0f}</text>')
    else:
        for i in range(6):
            x_tick = x_min + x_range * i / 5
            px = sx(x_tick)
            lines.append(f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" font-size="11" fill="#6c757d">{x_tick:.0f}</text>')

    if not data:
        lines.append(f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" font-size="13" fill="#99f6e4" font-style="italic">Waiting for data...</text>')

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
            lines.append(f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"{dash}/>')
        if len(points) <= 30:
            for px, py in points:
                lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>')

    if title:
        lines.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#0d6e6e">{title}</text>')
    if x_label:
        lines.append(f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#6c757d">{x_label}</text>')
    if y_label:
        lines.append(f'<text x="14" y="{mt + ch / 2}" text-anchor="middle" font-size="12" fill="#6c757d" transform="rotate(-90, 14, {mt + ch / 2})">{y_label}</text>')

    names = y_display_names or {}
    if len(y_keys) > 1:
        lx = ml + 10
        for si, key in enumerate(y_keys):
            color = colors[si % len(colors)]
            ly = mt + 14 + si * 18
            lines.append(f'<rect x="{lx}" y="{ly - 6}" width="12" height="12" rx="2" fill="{color}"/>')
            label = names.get(key, key)
            lines.append(f'<text x="{lx + 16}" y="{ly + 4}" font-size="11" fill="#1a1a2e">{label}</text>')

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

    def sy(v): return mt + ch - (v / y_max_plot) * ch

    lines_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        lines_svg.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#ccfbf1" stroke-width="1"/>')
        lines_svg.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:.3f}</text>')

    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            lines_svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" height="{bh:.1f}" fill="{color}" rx="3"/>')
            lines_svg.append(f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="10" fill="#0d6e6e">{val:.3f}</text>')
        display_label = label if len(label) <= 12 else label[:10] + ".."
        lines_svg.append(f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" text-anchor="middle" font-size="10" fill="#6c757d">{display_label}</text>')

    if title:
        lines_svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#0d6e6e">{title}</text>')

    lx = ml + cw - len(series) * 100
    for si, name in enumerate(series):
        color = colors[si % len(colors)]
        lines_svg.append(f'<rect x="{lx + si * 100}" y="{mt + ch + 35}" width="12" height="12" rx="2" fill="{color}"/>')
        lines_svg.append(f'<text x="{lx + si * 100 + 16}" y="{mt + ch + 46}" font-size="11" fill="#1a1a2e">{name}</text>')

    lines_svg.append("</svg>")
    return "\n".join(lines_svg)


# ------------------------------------------------------------------
# Mask overlay rendering
# ------------------------------------------------------------------

def overlay_masks_by_class(image, masks, labels, alpha: float = 0.45):
    """Overlay instance masks colored by nucleus class with white contour outlines."""
    import numpy as np
    from PIL import Image as PILImage, ImageFilter

    img_array = np.array(image).astype(np.float32)
    if len(masks) == 0:
        return image.copy()

    overlay = img_array.copy()
    for mask, label in zip(masks, labels):
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
        mask_bool = mask_np > 0.5
        label_int = int(label) if hasattr(label, "item") else int(label)
        color = CLASS_COLORS.get(label_int, (200, 200, 200))
        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, overlay[:, :, c] * (1 - alpha) + color[c] * alpha, overlay[:, :, c])

    for mask in masks:
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
        mask_bool = mask_np > 0.5
        mask_pil = PILImage.fromarray((mask_bool * 255).astype(np.uint8))
        dilated = mask_pil.filter(ImageFilter.MaxFilter(3))
        boundary = np.array(dilated) > np.array(mask_pil)
        overlay[boundary] = [255, 255, 255]

    return PILImage.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


def img_to_data_uri(img, max_dim: int = 600) -> str:
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
# Confidence heatmap
# ------------------------------------------------------------------

def render_confidence_heatmap(image, masks, scores, max_dim: int = 600):
    """Render confidence heatmap overlay (fire colormap: black → red → yellow → white)."""
    import numpy as np
    from PIL import Image as PILImage

    img_array = np.array(image).astype(np.float32)
    h, w = img_array.shape[:2]
    conf_map = np.zeros((h, w), dtype=np.float32)

    for mask, score in zip(masks, scores):
        mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
        mask_bool = mask_np > 0.5
        score_val = float(score) if hasattr(score, "item") else float(score)
        conf_map = np.where(mask_bool, np.maximum(conf_map, score_val), conf_map)

    rgb = img_array.copy() * 0.3
    if conf_map.max() > 0:
        r = np.clip(conf_map * 3, 0, 1)
        g = np.clip(conf_map * 3 - 1, 0, 1)
        b = np.clip(conf_map * 3 - 2, 0, 1)
        a = (conf_map > 0).astype(np.float32) * 0.7
        for c, channel in enumerate([r, g, b]):
            rgb[:, :, c] = rgb[:, :, c] * (1 - a) + channel * 255 * a

    return PILImage.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


# ------------------------------------------------------------------
# Confusion matrix SVG
# ------------------------------------------------------------------

def make_confusion_matrix_svg(
    matrix: list[list[int]],
    labels: list[str],
    colors: list[str],
    title: str = "Nucleus Type Confusion Matrix",
    width: int = 500,
    height: int = 500,
) -> str:
    """Render a confusion matrix as an SVG heatmap with teal color scale."""
    if not matrix or not matrix[0]:
        return ""

    n = len(matrix)
    ml, mr, mt, mb = 100, 30, 50, 80
    cw = width - ml - mr
    ch = height - mt - mb
    cell_w = cw / n
    cell_h = ch / n

    max_val = max(max(row) for row in matrix) if matrix else 1
    max_val = max_val or 1

    bg_r, bg_g, bg_b = 240, 253, 250
    fg_r, fg_g, fg_b = 13, 110, 110

    def cell_color(val):
        t = val / max_val if max_val > 0 else 0
        return f"rgb({int(bg_r + (fg_r - bg_r) * t)},{int(bg_g + (fg_g - bg_g) * t)},{int(bg_b + (fg_b - bg_b) * t)})"

    def text_color(val):
        return "#fff" if val / max_val > 0.5 else "#0d6e6e"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        lines.append(f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="14" font-weight="600" fill="#0d6e6e">{title}</text>')

    for ri, row in enumerate(matrix):
        for ci, val in enumerate(row):
            x = ml + ci * cell_w
            y = mt + ri * cell_h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{cell_color(val)}" stroke="#fff" stroke-width="1.5" rx="2"/>')
            if val > 0:
                lines.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-size="13" font-weight="600" fill="{text_color(val)}">{val}</text>')

    for ri, label in enumerate(labels):
        y = mt + ri * cell_h + cell_h / 2 + 4
        color = colors[ri] if ri < len(colors) else "#0d6e6e"
        lines.append(f'<text x="{ml - 8}" y="{y:.1f}" text-anchor="end" font-size="11" fill="{color}" font-weight="600">{label}</text>')

    for ci, label in enumerate(labels):
        x = ml + ci * cell_w + cell_w / 2
        color = colors[ci] if ci < len(colors) else "#0d6e6e"
        lines.append(f'<text x="{x:.1f}" y="{mt + ch + 20}" text-anchor="middle" font-size="11" fill="{color}" font-weight="600" transform="rotate(-35, {x:.1f}, {mt + ch + 20})">{label}</text>')

    lines.append(f'<text x="{ml - 60}" y="{mt + ch / 2}" text-anchor="middle" font-size="12" fill="#0f766e" font-weight="600" transform="rotate(-90, {ml - 60}, {mt + ch / 2})">True Class</text>')
    lines.append(f'<text x="{ml + cw / 2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#0f766e" font-weight="600">Predicted Class</text>')

    lines.append("</svg>")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Interactive per-class toggle overlay viewer
# ------------------------------------------------------------------

def interactive_overlay_html(
    image,
    pred_masks,
    pred_labels,
    pred_scores,
    gt_overlay_uri: str,
    viewer_id: str,
) -> str:
    """Render an interactive overlay viewer with per-class toggles and heatmap mode."""
    import numpy as np
    from PIL import Image as PILImage

    base_uri = img_to_data_uri(image, max_dim=512)

    class_uris = {}
    for cls_id in range(1, NUM_CLASSES):
        cls_mask = np.zeros((image.size[1], image.size[0]), dtype=np.float32)
        for mask, label in zip(pred_masks, pred_labels):
            label_int = int(label) if hasattr(label, "item") else int(label)
            if label_int != cls_id:
                continue
            mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
            cls_mask = np.maximum(cls_mask, (mask_np > 0.5).astype(np.float32))

        if cls_mask.max() > 0:
            color = CLASS_COLORS[cls_id]
            rgba = np.zeros((image.size[1], image.size[0], 4), dtype=np.uint8)
            mask_bool = cls_mask > 0.5
            rgba[mask_bool, 0] = color[0]
            rgba[mask_bool, 1] = color[1]
            rgba[mask_bool, 2] = color[2]
            rgba[mask_bool, 3] = 140

            overlay_img = PILImage.fromarray(rgba, "RGBA")
            buf = io.BytesIO()
            overlay_img.save(buf, format="PNG")
            class_uris[cls_id] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    heatmap_img = render_confidence_heatmap(image, pred_masks, pred_scores)
    heatmap_uri = img_to_data_uri(heatmap_img, max_dim=512)

    layers_html = ""
    for cls_id, uri in class_uris.items():
        layers_html += f'<img id="{viewer_id}_cls{cls_id}" src="{uri}" style="position:absolute;top:0;left:0;width:100%;height:100%;" />'
    layers_html += f'<img id="{viewer_id}_heatmap" src="{heatmap_uri}" style="position:absolute;top:0;left:0;width:100%;height:100%;display:none;" />'

    checkboxes_html = ""
    for cls_id in range(1, NUM_CLASSES):
        hex_c = CLASS_COLORS_HEX[cls_id]
        name = CLASS_NAMES[cls_id]
        has_layer = cls_id in class_uris
        checked = "checked" if has_layer else ""
        disabled = "" if has_layer else "disabled"
        checkboxes_html += (
            f'<label style="display:inline-flex;align-items:center;gap:3px;margin-right:8px;font-size:0.8em;">'
            f'<input type="checkbox" {checked} {disabled} onchange="'
            f"document.getElementById('{viewer_id}_cls{cls_id}').style.display="
            f"this.checked?'block':'none';"
            f'" />'
            f'<span style="color:{hex_c};font-weight:600;">{name}</span></label>'
        )

    checkboxes_html += (
        f'<label style="display:inline-flex;align-items:center;gap:3px;margin-left:12px;font-size:0.8em;">'
        f'<input type="checkbox" onchange="'
        f"var hm=document.getElementById('{viewer_id}_heatmap');"
        f"var base=document.getElementById('{viewer_id}_base');"
        f"if(this.checked){{hm.style.display='block';"
    )
    for cls_id in class_uris:
        checkboxes_html += f"document.getElementById('{viewer_id}_cls{cls_id}').style.display='none';"
    checkboxes_html += f"}}else{{hm.style.display='none';"
    for cls_id in class_uris:
        checkboxes_html += (
            f"document.getElementById('{viewer_id}_cls{cls_id}').style.display="
            f"document.querySelector('input[onchange*=\\\"cls{cls_id}\\\"]').checked?'block':'none';"
        )
    checkboxes_html += (
        f"}}"
        f'" />'
        f'<span style="color:#f59e0b;font-weight:600;">Confidence Heatmap</span></label>'
    )

    return f"""
    <div style="position:relative;display:inline-block;width:100%;max-width:512px;">
      <img id="{viewer_id}_base" src="{base_uri}" style="width:100%;display:block;border-radius:6px;" />
      {layers_html}
    </div>
    <div style="margin-top:6px;">{checkboxes_html}</div>
    """
