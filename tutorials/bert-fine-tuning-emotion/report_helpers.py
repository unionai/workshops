"""Flyte report helpers — CSS, SVG charts, attention heatmaps, and HTML wrappers."""

import html as html_module

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f3460; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  .report tr:nth-child(even) { background: #f8f9fa; }
  .report .highlight { color: #0f3460; font-weight: 700; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d4edda; color: #155724; }
  .report .badge-info { background: #d1ecf1; color: #0c5460; }
  .report .badge-danger { background: #f8d7da; color: #721c24; }
  .report .badge-warning { background: #fff3cd; color: #856404; }
  .report .chart-container { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .attention-text { line-height: 2.2; font-size: 1.05em; margin: 8px 0; }
  .report .attention-text span { padding: 2px 4px; border-radius: 3px; margin: 1px; }
  .report .confidence-bar { display: flex; align-items: center; margin: 4px 0; }
  .report .confidence-bar .bar-label { width: 80px; font-size: 0.85em; text-align: right; padding-right: 8px; }
  .report .confidence-bar .bar-track { flex: 1; background: #e9ecef; height: 20px; border-radius: 4px; overflow: hidden; }
  .report .confidence-bar .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .report .confidence-bar .bar-value { width: 50px; font-size: 0.85em; padding-left: 8px; }
</style>
"""


def wrap_report(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# Line chart (same as GRPO tutorial)
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
    y_display_names: dict[str, str] | None = None,
) -> str:
    """Generate an SVG line chart from a list of dicts."""
    default_colors = ["#5a7db5", "#0f3460", "#06d6a0", "#ffc107", "#6c757d"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    x_vals = [d[x_key] for d in data] if data else []
    if x_vals:
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
    y_min_plot = y_min - y_pad
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
        lines.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        lines.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:.3f}</text>')

    lines.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" stroke="#adb5bd" stroke-width="1.5"/>')
    lines.append(f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" stroke="#adb5bd" stroke-width="1.5"/>')

    if x_vals:
        n_x_ticks = min(len(data), 10)
        step = max(1, len(data) // n_x_ticks)
        for i in range(0, len(data), step):
            px = sx(x_vals[i])
            lines.append(f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" font-size="11" fill="#6c757d">{x_vals[i]:.1f}</text>')

    if not data:
        lines.append(f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" font-size="13" fill="#adb5bd" font-style="italic">Waiting for data...</text>')

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
        lines.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')
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


# ------------------------------------------------------------------
# Bar chart
# ------------------------------------------------------------------

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

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        svg.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:.1f}</text>')

    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" height="{bh:.1f}" fill="{color}" rx="2"/>')
            svg.append(f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="10" fill="#1a1a2e">{val:.1f}%</text>')
        svg.append(f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" text-anchor="middle" font-size="11" fill="#6c757d">{label}</text>')

    if title:
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    lx = ml + cw - len(series) * 100
    for si, name in enumerate(series):
        color = colors[si % len(colors)]
        svg.append(f'<rect x="{lx + si * 100}" y="{mt + ch + 35}" width="12" height="12" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{lx + si * 100 + 16}" y="{mt + ch + 46}" font-size="11" fill="#1a1a2e">{name}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Confusion matrix heatmap (SVG)
# ------------------------------------------------------------------

def make_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    title: str = "Confusion Matrix",
    width: int = 500,
    height: int = 500,
) -> str:
    """Generate an SVG confusion matrix heatmap."""
    n = len(labels)
    ml, mt = 90, 50
    cell_size = min((width - ml - 20) / n, (height - mt - 40) / n)
    grid_w = cell_size * n
    grid_h = cell_size * n

    max_val = max(max(row) for row in matrix) if matrix else 1

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ml + grid_w + 30} {mt + grid_h + 50}" '
        f'style="width:100%;max-width:{int(ml + grid_w + 30)}px;height:auto;">',
        f'<rect width="{ml + grid_w + 30}" height="{mt + grid_h + 50}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{(ml + grid_w + 30) / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    # Axis labels
    svg.append(f'<text x="{ml + grid_w / 2}" y="{mt + grid_h + 35}" text-anchor="middle" font-size="12" fill="#6c757d">Predicted</text>')
    svg.append(f'<text x="12" y="{mt + grid_h / 2}" text-anchor="middle" font-size="12" fill="#6c757d" transform="rotate(-90, 12, {mt + grid_h / 2})">Actual</text>')

    for i in range(n):
        for j in range(n):
            x = ml + j * cell_size
            y = mt + i * cell_size
            val = matrix[i][j]
            intensity = val / max_val if max_val > 0 else 0

            if i == j:
                r = int(15 + (15 - 15) * (1 - intensity))
                g = int(52 + (52 - 52) * (1 - intensity))
                b = int(96 + (96 - 96) * (1 - intensity))
                opacity = 0.15 + intensity * 0.85
                color = f"rgba(15, 52, 96, {opacity:.2f})"
            else:
                opacity = 0.05 + intensity * 0.6
                color = f"rgba(220, 53, 69, {opacity:.2f})"

            text_color = "#fff" if intensity > 0.5 else "#1a1a2e"

            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_size:.1f}" height="{cell_size:.1f}" fill="{color}" stroke="#dee2e6" stroke-width="1"/>')
            svg.append(f'<text x="{x + cell_size / 2:.1f}" y="{y + cell_size / 2 + 5:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="{text_color}">{val}</text>')

    # Row labels (actual)
    for i, label in enumerate(labels):
        y = mt + i * cell_size + cell_size / 2 + 4
        svg.append(f'<text x="{ml - 6}" y="{y:.1f}" text-anchor="end" font-size="11" fill="#1a1a2e">{label}</text>')

    # Column labels (predicted)
    for j, label in enumerate(labels):
        x = ml + j * cell_size + cell_size / 2
        svg.append(f'<text x="{x:.1f}" y="{mt - 8}" text-anchor="middle" font-size="11" fill="#1a1a2e">{label}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Colored text for attention / token importance
# ------------------------------------------------------------------

EMOTION_COLORS = {
    "sadness": "#4a6fa5",
    "joy": "#f4a261",
    "love": "#e76f51",
    "anger": "#e63946",
    "fear": "#6c567b",
    "surprise": "#2a9d8f",
}


def make_attention_text(
    tokens: list[str],
    weights: list[float],
    color: str = "#0f3460",
    title: str = "",
) -> str:
    """Render tokens with background color intensity proportional to attention weight.

    Args:
        tokens: List of tokens (words/subwords).
        weights: Attention weights per token (0-1 normalized).
        color: Base color for highlighting (CSS color string).
        title: Optional title above the text.
    """
    max_w = max(weights) if weights else 1
    min_w = min(weights) if weights else 0
    w_range = max_w - min_w or 1

    spans = []
    for token, w in zip(tokens, weights):
        norm_w = (w - min_w) / w_range
        opacity = 0.05 + norm_w * 0.85
        text_color = "#fff" if norm_w > 0.6 else "#1a1a2e"
        clean_token = token.replace("##", "").replace("Ġ", "").replace("▁", "")
        safe_token = html_module.escape(clean_token)
        if not safe_token.strip():
            continue
        spans.append(
            f'<span style="background:rgba(15,52,96,{opacity:.2f});color:{text_color};'
            f'padding:2px 4px;border-radius:3px;margin:1px;">{safe_token}</span>'
        )

    title_html = f"<p style='font-size:0.85em;color:#6c757d;margin:4px 0;'>{title}</p>" if title else ""
    return f'{title_html}<div class="attention-text">{"".join(spans)}</div>'


def make_token_importance_text(
    tokens: list[str],
    importance: list[float],
    title: str = "",
) -> str:
    """Render tokens colored by gradient-based importance (green = positive, red = negative).

    For classification, higher importance means the token had more influence on the prediction.
    """
    max_imp = max(abs(v) for v in importance) if importance else 1

    spans = []
    for token, imp in zip(tokens, importance):
        norm = imp / max_imp if max_imp > 0 else 0
        abs_norm = abs(norm)
        opacity = 0.05 + abs_norm * 0.85

        if norm >= 0:
            bg = f"rgba(6, 214, 160, {opacity:.2f})"
        else:
            bg = f"rgba(230, 57, 70, {opacity:.2f})"

        text_color = "#fff" if abs_norm > 0.6 else "#1a1a2e"
        clean_token = token.replace("##", "").replace("Ġ", "").replace("▁", "")
        safe_token = html_module.escape(clean_token)
        if not safe_token.strip():
            continue
        spans.append(
            f'<span style="background:{bg};color:{text_color};'
            f'padding:2px 4px;border-radius:3px;margin:1px;">{safe_token}</span>'
        )

    title_html = f"<p style='font-size:0.85em;color:#6c757d;margin:4px 0;'>{title}</p>" if title else ""
    return f'{title_html}<div class="attention-text">{"".join(spans)}</div>'


# ------------------------------------------------------------------
# Confidence bars (horizontal)
# ------------------------------------------------------------------

def make_confidence_bars(
    labels: list[str],
    probabilities: list[float],
    predicted_idx: int = -1,
    true_idx: int = -1,
) -> str:
    """Render horizontal confidence bars for each class."""
    bars = []
    for i, (label, prob) in enumerate(zip(labels, probabilities)):
        pct = prob * 100
        color = EMOTION_COLORS.get(label, "#0f3460")

        label_suffix = ""
        if i == predicted_idx and i == true_idx:
            label_suffix = ' <span class="badge badge-success">correct</span>'
        elif i == predicted_idx:
            label_suffix = ' <span class="badge badge-danger">predicted</span>'
        elif i == true_idx:
            label_suffix = ' <span class="badge badge-info">true</span>'

        bars.append(
            f'<div class="confidence-bar">'
            f'  <div class="bar-label">{label}{label_suffix}</div>'
            f'  <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
            f'  <div class="bar-value">{pct:.1f}%</div>'
            f'</div>'
        )
    return "\n".join(bars)


# ------------------------------------------------------------------
# Pipeline step indicator
# ------------------------------------------------------------------

def pipeline_step_indicator(current: int, steps: list[str]) -> str:
    """Build a visual step indicator (checkmark/dot/circle)."""
    html = '<div style="display:flex;gap:24px;margin:16px 0;">'
    for i, label in enumerate(steps):
        if i < current:
            icon = '<span style="color:#155724;font-size:1.2em;">&#10003;</span>'
            color = "#155724"
        elif i == current:
            icon = '<span style="color:#0f3460;font-size:1.2em;">&#9679;</span>'
            color = "#0f3460"
        else:
            icon = '<span style="color:#adb5bd;font-size:1.2em;">&#9675;</span>'
            color = "#adb5bd"
        html += f'<div style="text-align:center;"><div>{icon}</div><div style="font-size:0.85em;color:{color};">{label}</div></div>'
    html += '</div>'
    return html
