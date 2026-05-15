"""
Drug Molecule Screening — Virtual screening pipeline for drug-likeness.

Pipeline: parse a library of drug molecules, compute physicochemical properties,
screen candidates against a target profile (Lipinski's Rule of Five + custom
filters), and rank them by drug-likeness score.

Usage:
    # Default library (~15 well-known drugs)
    flyte run --local --tui workflow.py pipeline

    # Custom target profile (stricter MW filter)
    flyte run --local --tui workflow.py pipeline \\
        --target_profile '{"mw": [100, 400], "logp": [-0.5, 4.0]}'

    # Remote
    flyte run workflow.py pipeline
"""

import base64
import io
import json
import logging
import math
import os
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Default molecule library — real SMILES for well-known drugs
# ------------------------------------------------------------------

DEFAULT_MOLECULES = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin G": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O",
    "Diazepam": "ClC1=CC2=C(C=C1)N(C(=O)CN=C2C3=CC=CC=C3)C",
    "Omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
    "Atorvastatin": "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "Methotrexate": "CN(CC1=CN=C2N=C(N=C(N)C2=N1)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O",
    "Doxorubicin": "CC1C(C(CC(O1)OC2CC(CC3=C2C(=C4C(=C3O)C(=O)C5=C(C4=O)C(=CC=C5)OC)O)(C(=O)CO)O)N)O",
    "Tamoxifen": "CCC(=C(C1=CC=CC=C1)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3",
    "Lopinavir": "CC1=C(C(=CC=C1)C)OCC(=O)NC(CC2=CC=CC=C2)C(CC(CC3=CC=CC=C3)NC(=O)C(C(C)C)N4CCCNC4=O)O",
    "Remdesivir": "CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4",
    "Erlotinib": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
}


# ------------------------------------------------------------------
# Report styling — pharma blue/cyan theme
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0e4f6e; border-bottom: 2px solid #0891b2; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #155e75; margin-top: 20px; }
  .report .card { background: #ecfeff; border: 1px solid #a5f3fc; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #cffafe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0e4f6e; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0e4f6e; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #cffafe; }
  .report tr:nth-child(even) { background: #ecfeff; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #cffafe; color: #155e75; }
  .report .chart-container { background: #fff; border: 1px solid #cffafe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #ecfeff; border-left: 4px solid #0891b2; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .molecule-card { background: #fff; border: 1px solid #cffafe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .molecule-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }
  .report .funnel { text-align: center; margin: 24px 0; }
</style>
"""


def _wrap_report(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# SVG chart helpers
# ------------------------------------------------------------------

def _mol_to_data_uri(mol, size: tuple[int, int] = (300, 300)) -> str:
    """Convert an RDKit molecule to a PNG base64 data URI."""
    from rdkit.Chem import Draw

    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def _make_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 340,
    y_max_cap: float | None = None,
    horizontal: bool = False,
    value_fmt: str = ".1f",
) -> str:
    """Generate an SVG grouped bar chart.

    Args:
        labels: Category labels.
        series: Dict mapping series name to list of values.
        title: Chart title.
        colors: Colors for each series.
        width/height: SVG dimensions.
        y_max_cap: Cap the y-axis at this value.
        horizontal: If True, draw horizontal bars.
        value_fmt: Format string for value labels.

    Returns:
        SVG string.
    """
    if not labels:
        return ""

    default_colors = ["#0891b2", "#0e4f6e", "#06d6a0", "#a5f3fc", "#155e75"]
    colors = colors or default_colors

    if horizontal:
        return _make_horizontal_bar_chart(labels, series, title, colors, width, height, value_fmt)

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
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid lines
    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e0f2fe" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:{value_fmt}}</text>'
        )

    # Axes
    svg.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>'
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
            svg.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" '
                f'height="{bh:.1f}" fill="{color}" rx="2"/>'
            )
            svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="9" fill="#1a1a2e">'
                f'{val:{value_fmt}}</text>'
            )
        # Truncate long labels
        disp_label = label if len(label) <= 12 else label[:10] + ".."
        svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 16}" '
            f'text-anchor="middle" font-size="10" fill="#6c757d" '
            f'transform="rotate(-35, {gx + n_series * bar_width / 2:.1f}, {mt + ch + 16})">'
            f'{disp_label}</text>'
        )

    # Title
    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0e4f6e">{title}</text>'
        )

    # Legend
    if n_series > 1:
        lx = ml + cw - len(series) * 100
        for si, name in enumerate(series):
            color = colors[si % len(colors)]
            svg.append(
                f'<rect x="{lx + si * 100}" y="{mt + ch + 40}" width="12" '
                f'height="12" rx="2" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{lx + si * 100 + 16}" y="{mt + ch + 51}" font-size="11" '
                f'fill="#1a1a2e">{name}</text>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_horizontal_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 400,
    value_fmt: str = ".1f",
) -> str:
    """Generate an SVG horizontal bar chart (sorted)."""
    default_colors = ["#0891b2", "#0e4f6e", "#06d6a0"]
    colors = colors or default_colors

    n = len(labels)
    row_height = max(22, min(35, (height - 80) // max(n, 1)))
    actual_height = max(height, 80 + n * row_height)
    ml, mr, mt, mb = 120, 60, 40, 20
    cw = width - ml - mr
    ch = actual_height - mt - mb

    # Use first series
    first_key = list(series.keys())[0]
    vals = series[first_key]
    x_max = max(vals) * 1.15 if vals else 1

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {actual_height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{actual_height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0e4f6e">{title}</text>'
        )

    bar_h = row_height * 0.65
    for i, (label, val) in enumerate(zip(labels, vals)):
        y = mt + i * row_height
        bw = (val / x_max) * cw if x_max else 0
        color = colors[i % len(colors)]
        # Label
        disp = label if len(label) <= 14 else label[:12] + ".."
        svg.append(
            f'<text x="{ml - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#1a1a2e">{disp}</text>'
        )
        # Bar
        svg.append(
            f'<rect x="{ml}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" rx="3"/>'
        )
        # Value
        svg.append(
            f'<text x="{ml + bw + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
            f'font-size="11" fill="#0e4f6e" font-weight="600">{val:{value_fmt}}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    color_scale: str = "cyan",
    width: int = 700,
    height: int = 500,
    value_fmt: str = ".2f",
) -> str:
    """Generate an SVG heatmap.

    Args:
        matrix: 2D list of values (rows x cols).
        row_labels: Labels for rows.
        col_labels: Labels for columns.
        title: Chart title.
        color_scale: Color scheme ("cyan", "red", "green").
        width/height: SVG dimensions.
        value_fmt: Format string for cell values.

    Returns:
        SVG string.
    """
    if not matrix or not matrix[0]:
        return ""

    n_rows = len(matrix)
    n_cols = len(matrix[0])

    ml, mr, mt, mb = 110, 20, 70, 20
    cw = width - ml - mr
    ch = height - mt - mb
    cell_w = cw / n_cols
    cell_h = ch / n_rows

    # Flatten to find range
    flat = [v for row in matrix for v in row]
    v_min = min(flat)
    v_max = max(flat)
    v_range = v_max - v_min or 1

    def color_for(v):
        t = (v - v_min) / v_range
        if color_scale == "cyan":
            # White to deep teal
            r = int(255 - t * (255 - 14))
            g = int(255 - t * (255 - 79))
            b = int(255 - t * (255 - 110))
        elif color_scale == "red":
            r = int(255 - t * 50)
            g = int(255 - t * 200)
            b = int(255 - t * 200)
        else:  # green
            r = int(255 - t * 200)
            g = int(255 - t * 50)
            b = int(255 - t * 200)
        return f"rgb({r},{g},{b})"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0e4f6e">{title}</text>'
        )

    # Column labels (rotated)
    for ci, label in enumerate(col_labels):
        x = ml + ci * cell_w + cell_w / 2
        disp = label if len(label) <= 12 else label[:10] + ".."
        svg.append(
            f'<text x="{x:.1f}" y="{mt - 8}" text-anchor="end" font-size="10" '
            f'fill="#1a1a2e" transform="rotate(-45, {x:.1f}, {mt - 8})">{disp}</text>'
        )

    # Row labels + cells
    for ri, (row_label, row_vals) in enumerate(zip(row_labels, matrix)):
        y = mt + ri * cell_h
        disp = row_label if len(row_label) <= 14 else row_label[:12] + ".."
        svg.append(
            f'<text x="{ml - 8}" y="{y + cell_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#1a1a2e">{disp}</text>'
        )
        for ci, val in enumerate(row_vals):
            x = ml + ci * cell_w
            fill = color_for(val)
            svg.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" '
                f'height="{cell_h:.1f}" fill="{fill}" stroke="#fff" stroke-width="1"/>'
            )
            # Text color: dark on light, light on dark
            t = (val - v_min) / v_range
            txt_color = "#fff" if t > 0.55 else "#1a1a2e"
            # Only show text if cells are large enough
            if cell_w > 30 and cell_h > 18:
                svg.append(
                    f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 4:.1f}" '
                    f'text-anchor="middle" font-size="9" fill="{txt_color}">'
                    f'{val:{value_fmt}}</text>'
                )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_scatter_plot(
    points: list[dict],
    x_label: str = "MW",
    y_label: str = "LogP",
    title: str = "",
    reference_lines: list[dict] | None = None,
    width: int = 700,
    height: int = 400,
) -> str:
    """Generate an SVG scatter plot.

    Args:
        points: List of dicts with "x", "y", "label" keys.
        x_label/y_label: Axis labels.
        title: Chart title.
        reference_lines: List of dicts with "axis" ("x"/"y"), "value", "label".
        width/height: SVG dimensions.

    Returns:
        SVG string.
    """
    if not points:
        return ""

    ml, mr, mt, mb = 60, 30, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    x_vals = [p["x"] for p in points]
    y_vals = [p["y"] for p in points]
    x_min, x_max = min(x_vals) * 0.9, max(x_vals) * 1.1
    y_min, y_max = min(y_vals) - 1, max(y_vals) + 1

    # Extend ranges to include reference lines
    if reference_lines:
        for rl in reference_lines:
            if rl["axis"] == "x":
                x_max = max(x_max, rl["value"] * 1.1)
            else:
                y_max = max(y_max, rl["value"] * 1.1)

    x_range = x_max - x_min or 1
    y_range = y_max - y_min or 1

    def sx(v):
        return ml + (v - x_min) / x_range * cw

    def sy(v):
        return mt + ch - (v - y_min) / y_range * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid
    for i in range(6):
        y_tick = y_min + y_range * i / 5
        py = sy(y_tick)
        svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e0f2fe" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.1f}</text>'
        )

    for i in range(6):
        x_tick = x_min + x_range * i / 5
        px = sx(x_tick)
        svg.append(
            f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
            f'font-size="11" fill="#6c757d">{x_tick:.0f}</text>'
        )

    # Axes
    svg.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>'
    )
    svg.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#94a3b8" stroke-width="1.5"/>'
    )

    # Reference lines (Lipinski boundaries)
    if reference_lines:
        for rl in reference_lines:
            if rl["axis"] == "x":
                px = sx(rl["value"])
                svg.append(
                    f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt + ch}" '
                    f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6,4"/>'
                )
                svg.append(
                    f'<text x="{px + 4:.1f}" y="{mt + 14}" font-size="10" '
                    f'fill="#ef4444" font-weight="600">{rl["label"]}</text>'
                )
            else:
                py = sy(rl["value"])
                svg.append(
                    f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
                    f'stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6,4"/>'
                )
                svg.append(
                    f'<text x="{ml + cw - 4:.1f}" y="{py - 6:.1f}" text-anchor="end" '
                    f'font-size="10" fill="#ef4444" font-weight="600">{rl["label"]}</text>'
                )

    # Drug-like zone shading (MW<=500 and LogP<=5 quadrant)
    if reference_lines:
        mw_line = next((rl for rl in reference_lines if rl["axis"] == "x"), None)
        logp_line = next((rl for rl in reference_lines if rl["axis"] == "y"), None)
        if mw_line and logp_line:
            zx1 = sx(x_min)
            zx2 = sx(min(mw_line["value"], x_max))
            zy1 = sy(min(logp_line["value"], y_max))
            zy2 = sy(y_min)
            svg.append(
                f'<rect x="{zx1:.1f}" y="{zy1:.1f}" '
                f'width="{zx2 - zx1:.1f}" height="{zy2 - zy1:.1f}" '
                f'fill="#0891b2" opacity="0.06" rx="4"/>'
            )
            svg.append(
                f'<text x="{zx1 + 8:.1f}" y="{zy2 - 8:.1f}" font-size="11" '
                f'fill="#0891b2" font-weight="600" opacity="0.6">Drug-like Zone</text>'
            )

    # Points
    point_colors = ["#0891b2", "#0e4f6e", "#06d6a0", "#155e75", "#0284c7",
                    "#059669", "#0d9488", "#0369a1", "#047857", "#115e59",
                    "#0c4a6e", "#064e3b", "#1e3a5f", "#134e4a", "#075985"]
    for i, pt in enumerate(points):
        px, py = sx(pt["x"]), sy(pt["y"])
        color = point_colors[i % len(point_colors)]
        svg.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" '
            f'stroke="#fff" stroke-width="1.5" opacity="0.85"/>'
        )
        # Label offset to avoid overlap
        offset_x = 8
        offset_y = -8 if i % 2 == 0 else 14
        label = pt["label"] if len(pt["label"]) <= 12 else pt["label"][:10] + ".."
        svg.append(
            f'<text x="{px + offset_x:.1f}" y="{py + offset_y:.1f}" '
            f'font-size="9" fill="#1a1a2e">{label}</text>'
        )

    # Title
    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#0e4f6e">{title}</text>'
        )

    # Axis labels
    if x_label:
        svg.append(
            f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d">{x_label}</text>'
        )
    if y_label:
        svg.append(
            f'<text x="14" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d" '
            f'transform="rotate(-90, 14, {mt + ch / 2})">{y_label}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_funnel(
    stages: list[dict],
    title: str = "",
    width: int = 600,
    height: int = 400,
) -> str:
    """Generate an SVG funnel visualization.

    Args:
        stages: List of dicts with "label", "count", "total" keys.
        title: Chart title.
        width/height: SVG dimensions.

    Returns:
        SVG string.
    """
    if not stages:
        return ""

    n = len(stages)
    mt = 50
    mb = 20
    available_h = height - mt - mb
    stage_h = available_h / n
    cx = width / 2

    # Color gradient from light cyan to deep teal
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        r = int(207 - t * (207 - 14))
        g = int(250 - t * (250 - 79))
        b = int(254 - t * (254 - 110))
        colors.append(f"rgb({r},{g},{b})")

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(
            f'<text x="{cx}" y="28" text-anchor="middle" '
            f'font-size="16" font-weight="700" fill="#0e4f6e">{title}</text>'
        )

    max_count = stages[0]["count"] if stages else 1
    max_width = width * 0.75

    for i, stage in enumerate(stages):
        y_top = mt + i * stage_h
        y_bot = y_top + stage_h

        # Width proportional to count
        w_top = max_width * (stage["count"] / max_count) if i == 0 else prev_w_bot
        if i < n - 1:
            w_bot = max_width * (stages[i + 1]["count"] / max_count)
        else:
            w_bot = max_width * (stage["count"] / max_count) * 0.7

        prev_w_bot = w_bot

        # Trapezoid
        x1_top = cx - w_top / 2
        x2_top = cx + w_top / 2
        x1_bot = cx - w_bot / 2
        x2_bot = cx + w_bot / 2

        svg.append(
            f'<polygon points="{x1_top:.1f},{y_top:.1f} {x2_top:.1f},{y_top:.1f} '
            f'{x2_bot:.1f},{y_bot:.1f} {x1_bot:.1f},{y_bot:.1f}" '
            f'fill="{colors[i]}" stroke="#fff" stroke-width="2"/>'
        )

        # Text: dark on light, white on dark
        t = i / max(n - 1, 1)
        txt_color = "#0e4f6e" if t < 0.5 else "#fff"
        y_mid = (y_top + y_bot) / 2

        svg.append(
            f'<text x="{cx}" y="{y_mid - 4:.1f}" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="{txt_color}">{stage["label"]}</text>'
        )
        svg.append(
            f'<text x="{cx}" y="{y_mid + 14:.1f}" text-anchor="middle" '
            f'font-size="12" fill="{txt_color}" opacity="0.85">'
            f'{stage["count"]} / {stage["total"]}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Task 1: Load and validate molecules
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def load_molecules(
    molecules_json: str = "",
) -> flyte.io.Dir:
    """Parse SMILES strings, validate with RDKit, generate 2D depictions.

    Args:
        molecules_json: JSON string mapping molecule names to SMILES.
            Defaults to a curated library of ~15 well-known drugs.

    Returns:
        Directory containing molecule data (JSON + PNG depictions).
    """
    from rdkit import Chem
    from rdkit.Chem import Draw

    if molecules_json.strip():
        molecules = json.loads(molecules_json)
    else:
        molecules = DEFAULT_MOLECULES

    out_dir = tempfile.mkdtemp(prefix="mol_library_")
    results = []
    valid_count = 0
    invalid_count = 0

    log.info(f"Parsing {len(molecules)} molecules...")

    for name, smiles in molecules.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"  [INVALID] {name}: {smiles}")
            invalid_count += 1
            continue

        valid_count += 1

        # Generate 2D depiction as PNG
        img = Draw.MolToImage(mol, size=(300, 300))
        img_path = os.path.join(out_dir, f"{name.replace(' ', '_')}.png")
        img.save(img_path)

        results.append({
            "name": name,
            "smiles": smiles,
            "valid": True,
            "image_file": os.path.basename(img_path),
        })

    # Save molecule manifest
    manifest = {
        "total": len(molecules),
        "valid": valid_count,
        "invalid": invalid_count,
        "molecules": results,
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log.info(f"Loaded {valid_count} valid molecules ({invalid_count} invalid)")

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Compute physicochemical properties
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def compute_properties(
    molecule_dir: flyte.io.Dir,
) -> str:
    """Compute drug-likeness properties for all molecules.

    Computes MW, LogP, HBD, HBA, TPSA, rotatable bonds, formal charge,
    ring count, QED, and Lipinski Rule of Five compliance.

    Returns:
        JSON string with all computed properties.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem.QED import qed

    # --- Loading report ---
    await flyte.report.replace.aio(
        _wrap_report("<h2>Computing Molecular Properties...</h2>"
                      "<p>Analyzing physicochemical descriptors for all molecules.</p>"),
        do_flush=True,
    )

    mol_dir = await molecule_dir.download()
    with open(os.path.join(mol_dir, "manifest.json")) as f:
        manifest = json.load(f)

    molecules_data = []
    lipinski_pass = 0

    for mol_info in manifest["molecules"]:
        mol = Chem.MolFromSmiles(mol_info["smiles"])
        if mol is None:
            continue

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        formal_charge = Chem.GetFormalCharge(mol)
        num_rings = Lipinski.RingCount(mol)
        qed_score = qed(mol)

        # Lipinski Rule of Five
        lipinski = {
            "mw_ok": mw <= 500,
            "logp_ok": logp <= 5,
            "hbd_ok": hbd <= 5,
            "hba_ok": hba <= 10,
        }
        lipinski_all = all(lipinski.values())
        if lipinski_all:
            lipinski_pass += 1

        # Read image for data URI
        img_path = os.path.join(mol_dir, mol_info["image_file"])
        data_uri = ""
        if os.path.exists(img_path):
            with open(img_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode()
                data_uri = f"data:image/png;base64,{b64}"

        molecules_data.append({
            "name": mol_info["name"],
            "smiles": mol_info["smiles"],
            "mw": round(mw, 2),
            "logp": round(logp, 2),
            "hbd": hbd,
            "hba": hba,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable,
            "formal_charge": formal_charge,
            "num_rings": num_rings,
            "qed": round(qed_score, 4),
            "lipinski": lipinski,
            "lipinski_pass": lipinski_all,
            "image_data_uri": data_uri,
        })

    total = len(molecules_data)
    avg_mw = sum(m["mw"] for m in molecules_data) / total if total else 0
    avg_logp = sum(m["logp"] for m in molecules_data) / total if total else 0
    lipinski_rate = lipinski_pass / total * 100 if total else 0

    # ---- Build report ----
    html_parts = []

    # Header
    html_parts.append("<h2>Molecular Properties Analysis</h2>")

    # Stat grid
    html_parts.append('<div class="stat-grid">')
    for val, label in [
        (str(total), "Total Molecules"),
        (f"{lipinski_rate:.0f}%", "Lipinski Pass Rate"),
        (f"{avg_mw:.1f}", "Avg. MW (Da)"),
        (f"{avg_logp:.2f}", "Avg. LogP"),
    ]:
        html_parts.append(
            f'<div class="stat"><div class="value">{val}</div>'
            f'<div class="label">{label}</div></div>'
        )
    html_parts.append("</div>")

    # Molecule gallery
    html_parts.append("<h3>Molecule Library</h3>")
    html_parts.append('<div class="molecule-grid">')
    for m in molecules_data:
        if m["image_data_uri"]:
            badge_class = "badge-success" if m["lipinski_pass"] else "badge-danger"
            badge_text = "Lipinski Pass" if m["lipinski_pass"] else "Lipinski Fail"
            html_parts.append(
                f'<div class="molecule-card" style="text-align:center;">'
                f'<img src="{m["image_data_uri"]}" style="width:160px;height:160px;object-fit:contain;"/>'
                f'<div style="font-weight:600;margin-top:6px;color:#0e4f6e;">{m["name"]}</div>'
                f'<div style="font-size:0.8em;color:#6c757d;">MW: {m["mw"]:.1f} | LogP: {m["logp"]:.2f}</div>'
                f'<div><span class="badge {badge_class}">{badge_text}</span></div>'
                f'</div>'
            )
    html_parts.append("</div>")

    # MW bar chart (horizontal, sorted)
    sorted_by_mw = sorted(molecules_data, key=lambda m: m["mw"], reverse=True)
    mw_labels = [m["name"] for m in sorted_by_mw]
    mw_vals = [m["mw"] for m in sorted_by_mw]
    mw_chart = _make_bar_chart(
        mw_labels, {"MW (Da)": mw_vals},
        title="Molecular Weight Distribution",
        horizontal=True,
        width=700, height=max(300, len(mw_labels) * 30 + 80),
        value_fmt=".1f",
    )
    html_parts.append("<h3>Molecular Weight</h3>")
    html_parts.append(f'<div class="chart-container">{mw_chart}</div>')

    # LogP vs MW scatter plot
    scatter_points = [
        {"x": m["mw"], "y": m["logp"], "label": m["name"]}
        for m in molecules_data
    ]
    scatter_chart = _make_scatter_plot(
        scatter_points,
        x_label="Molecular Weight (Da)",
        y_label="LogP",
        title="LogP vs. Molecular Weight (Lipinski Boundaries)",
        reference_lines=[
            {"axis": "x", "value": 500, "label": "MW = 500"},
            {"axis": "y", "value": 5, "label": "LogP = 5"},
        ],
        width=700,
        height=420,
    )
    html_parts.append("<h3>Lipinski Space</h3>")
    html_parts.append(f'<div class="chart-container">{scatter_chart}</div>')

    # Property heatmap (molecules x properties)
    prop_names = ["MW", "LogP", "HBD", "HBA", "TPSA", "Rot. Bonds"]
    # Normalize each property to 0-1 for heatmap
    raw_matrix = []
    for m in molecules_data:
        raw_matrix.append([m["mw"], m["logp"], m["hbd"], m["hba"], m["tpsa"], m["rotatable_bonds"]])

    # Normalize per column
    n_props = len(prop_names)
    col_min = [min(row[c] for row in raw_matrix) for c in range(n_props)]
    col_max = [max(row[c] for row in raw_matrix) for c in range(n_props)]
    norm_matrix = []
    for row in raw_matrix:
        norm_row = []
        for c in range(n_props):
            rng = col_max[c] - col_min[c]
            norm_row.append((row[c] - col_min[c]) / rng if rng else 0.5)
        norm_matrix.append(norm_row)

    heatmap_labels = [m["name"] for m in molecules_data]
    heatmap = _make_heatmap(
        norm_matrix, heatmap_labels, prop_names,
        title="Normalized Property Heatmap",
        color_scale="cyan",
        width=700,
        height=max(400, len(heatmap_labels) * 28 + 100),
    )
    html_parts.append("<h3>Property Heatmap</h3>")
    html_parts.append(f'<div class="chart-container">{heatmap}</div>')

    # Lipinski compliance table
    html_parts.append("<h3>Lipinski Rule of Five Compliance</h3>")
    html_parts.append("<table><tr><th>Molecule</th><th>MW &le; 500</th>"
                      "<th>LogP &le; 5</th><th>HBD &le; 5</th>"
                      "<th>HBA &le; 10</th><th>Overall</th></tr>")
    for m in molecules_data:
        lip = m["lipinski"]

        def _badge(ok):
            if ok:
                return '<span class="badge badge-success">Pass</span>'
            return '<span class="badge badge-danger">Fail</span>'

        overall_badge = _badge(m["lipinski_pass"])
        html_parts.append(
            f'<tr><td><strong>{m["name"]}</strong></td>'
            f'<td>{_badge(lip["mw_ok"])}</td>'
            f'<td>{_badge(lip["logp_ok"])}</td>'
            f'<td>{_badge(lip["hbd_ok"])}</td>'
            f'<td>{_badge(lip["hba_ok"])}</td>'
            f'<td>{overall_badge}</td></tr>'
        )
    html_parts.append("</table>")

    # QED bar chart
    sorted_by_qed = sorted(molecules_data, key=lambda m: m["qed"], reverse=True)
    qed_labels = [m["name"] for m in sorted_by_qed]
    qed_vals = [m["qed"] for m in sorted_by_qed]
    qed_chart = _make_bar_chart(
        qed_labels, {"QED Score": qed_vals},
        title="Drug-likeness (QED Score)",
        horizontal=True,
        width=700, height=max(300, len(qed_labels) * 30 + 80),
        value_fmt=".3f",
        colors=["#06d6a0"],
    )
    html_parts.append("<h3>Drug-likeness (QED)</h3>")
    html_parts.append(f'<div class="chart-container">{qed_chart}</div>')

    # Flush full report
    await flyte.report.replace.aio(
        _wrap_report("\n".join(html_parts)),
        do_flush=True,
    )

    # Return properties as JSON (strip image data URIs to reduce size)
    output = {
        "total": total,
        "lipinski_pass_count": lipinski_pass,
        "lipinski_pass_rate": round(lipinski_rate, 2),
        "avg_mw": round(avg_mw, 2),
        "avg_logp": round(avg_logp, 2),
        "molecules": [
            {k: v for k, v in m.items() if k != "image_data_uri"}
            for m in molecules_data
        ],
    }
    return json.dumps(output)


# ------------------------------------------------------------------
# Task 3: Screen candidates against target profile
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def screen_candidates(
    properties_json: str,
    target_profile: str = "",
) -> str:
    """Screen molecules against a target drug profile and rank candidates.

    Scores each molecule on how well it matches the target profile, computes
    pairwise Tanimoto similarity, and produces a ranked list.

    Args:
        properties_json: JSON from compute_properties.
        target_profile: JSON string with desired property ranges.

    Returns:
        JSON string with ranked molecules and screening scores.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    await flyte.report.replace.aio(
        _wrap_report("<h2>Screening Candidates...</h2>"
                      "<p>Evaluating molecules against the target drug profile.</p>"),
        do_flush=True,
    )

    props = json.loads(properties_json)
    molecules = props["molecules"]

    # Default target profile
    if target_profile.strip():
        profile = json.loads(target_profile)
    else:
        profile = {
            "mw": [150, 500],
            "logp": [-0.5, 5.0],
            "hbd": [0, 5],
            "hba": [0, 10],
            "tpsa": [20, 140],
        }

    # --- Screening ---
    funnel_total = len(molecules)
    pass_mw = 0
    pass_logp = 0
    pass_lipinski = 0
    final_candidates = 0

    scored = []
    for m in molecules:
        score = 0
        max_score = 0
        criteria = {}

        # Check each profile criterion
        checks = [
            ("mw", m["mw"]),
            ("logp", m["logp"]),
            ("hbd", m["hbd"]),
            ("hba", m["hba"]),
            ("tpsa", m["tpsa"]),
        ]

        for key, val in checks:
            if key in profile:
                lo, hi = profile[key]
                max_score += 1
                in_range = lo <= val <= hi
                criteria[key] = in_range
                if in_range:
                    score += 1
                    # Bonus: closer to midpoint = higher score
                    mid = (lo + hi) / 2
                    rng = (hi - lo) / 2
                    dist = abs(val - mid) / rng if rng else 0
                    score += max(0, 0.5 * (1 - dist))

        # QED bonus
        score += m["qed"] * 2
        max_score += 2

        # Lipinski bonus
        if m["lipinski_pass"]:
            score += 1
        max_score += 1

        normalized_score = score / max_score if max_score else 0

        # Funnel tracking — cascading filter (each stage requires passing the previous)
        mw_ok = criteria.get("mw", True)
        logp_ok = criteria.get("logp", True)
        if mw_ok:
            pass_mw += 1
            if logp_ok:
                pass_logp += 1
                if m["lipinski_pass"]:
                    pass_lipinski += 1
                    if all(criteria.values()):
                        final_candidates += 1

        scored.append({
            **m,
            "screening_score": round(normalized_score, 4),
            "criteria_met": criteria,
            "all_criteria_met": all(criteria.values()),
        })

    # Sort by score descending
    scored.sort(key=lambda m: m["screening_score"], reverse=True)

    # --- Tanimoto similarity matrix ---
    fps = []
    valid_names = []
    for m in scored:
        mol = Chem.MolFromSmiles(m["smiles"])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_names.append(m["name"])

    similarity_matrix = []
    for i in range(len(fps)):
        row = []
        for j in range(len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            row.append(round(sim, 3))
        similarity_matrix.append(row)

    # ---- Build report ----
    html_parts = []
    html_parts.append("<h2>Candidate Screening Results</h2>")

    # Stat grid
    html_parts.append('<div class="stat-grid">')
    for val, label in [
        (str(funnel_total), "Total Screened"),
        (str(pass_lipinski), "Lipinski Passes"),
        (str(final_candidates), "All Criteria Met"),
        (f"{scored[0]['screening_score']:.3f}" if scored else "N/A", "Top Score"),
    ]:
        html_parts.append(
            f'<div class="stat"><div class="value">{val}</div>'
            f'<div class="label">{label}</div></div>'
        )
    html_parts.append("</div>")

    # Screening funnel
    funnel_stages = [
        {"label": "Total Molecules", "count": funnel_total, "total": funnel_total},
        {"label": "Pass MW Filter", "count": pass_mw, "total": funnel_total},
        {"label": "Pass LogP Filter", "count": pass_logp, "total": funnel_total},
        {"label": "Lipinski Compliant", "count": pass_lipinski, "total": funnel_total},
        {"label": "All Criteria Met", "count": final_candidates, "total": funnel_total},
    ]
    funnel_svg = _make_funnel(
        funnel_stages,
        title="Screening Funnel",
        width=600,
        height=380,
    )
    html_parts.append("<h3>Screening Funnel</h3>")
    html_parts.append(f'<div class="chart-container" style="text-align:center;">{funnel_svg}</div>')

    # Ranked candidates table
    html_parts.append("<h3>Ranked Candidates</h3>")
    html_parts.append(
        "<table><tr><th>Rank</th><th>Molecule</th><th>Score</th>"
        "<th>MW</th><th>LogP</th><th>QED</th><th>Lipinski</th><th>All Criteria</th></tr>"
    )
    for rank, m in enumerate(scored, 1):
        lip_badge = ('<span class="badge badge-success">Pass</span>'
                     if m["lipinski_pass"]
                     else '<span class="badge badge-danger">Fail</span>')
        crit_badge = ('<span class="badge badge-success">Pass</span>'
                      if m["all_criteria_met"]
                      else '<span class="badge badge-danger">Fail</span>')
        # Highlight top 3
        row_style = ' style="background:#ecfeff;font-weight:600;"' if rank <= 3 else ""
        html_parts.append(
            f"<tr{row_style}><td>{rank}</td><td>{m['name']}</td>"
            f"<td>{m['screening_score']:.3f}</td>"
            f"<td>{m['mw']:.1f}</td><td>{m['logp']:.2f}</td>"
            f"<td>{m['qed']:.3f}</td><td>{lip_badge}</td><td>{crit_badge}</td></tr>"
        )
    html_parts.append("</table>")

    # Top 5 candidate cards with structures
    html_parts.append("<h3>Top 5 Candidates</h3>")
    html_parts.append('<div class="molecule-grid">')
    for m in scored[:5]:
        mol = Chem.MolFromSmiles(m["smiles"])
        img_uri = _mol_to_data_uri(mol, size=(250, 250)) if mol else ""
        badge_class = "badge-success" if m["all_criteria_met"] else "badge-info"
        badge_text = "All Criteria Met" if m["all_criteria_met"] else "Partial Match"
        html_parts.append(
            f'<div class="molecule-card" style="text-align:center;">'
            f'<img src="{img_uri}" style="width:140px;height:140px;object-fit:contain;"/>'
            f'<div style="font-weight:700;margin-top:6px;color:#0e4f6e;font-size:1.05em;">{m["name"]}</div>'
            f'<div style="font-size:0.85em;color:#155e75;margin:4px 0;">Score: {m["screening_score"]:.3f}</div>'
            f'<div style="font-size:0.8em;color:#6c757d;">MW: {m["mw"]:.1f} | LogP: {m["logp"]:.2f} | QED: {m["qed"]:.3f}</div>'
            f'<div style="margin-top:4px;"><span class="badge {badge_class}">{badge_text}</span></div>'
            f'</div>'
        )
    html_parts.append("</div>")

    # Tanimoto similarity heatmap
    if similarity_matrix:
        sim_heatmap = _make_heatmap(
            similarity_matrix, valid_names, valid_names,
            title="Pairwise Tanimoto Similarity (Morgan Fingerprints)",
            color_scale="cyan",
            width=700,
            height=max(500, len(valid_names) * 32 + 100),
        )
        html_parts.append("<h3>Chemical Similarity</h3>")
        html_parts.append(f'<div class="chart-container">{sim_heatmap}</div>')

    await flyte.report.replace.aio(
        _wrap_report("\n".join(html_parts)),
        do_flush=True,
    )

    output = {
        "ranked_molecules": scored,
        "similarity_matrix": similarity_matrix,
        "similarity_labels": valid_names,
        "funnel": funnel_stages,
        "target_profile": profile,
    }
    return json.dumps(output)


# ------------------------------------------------------------------
# Task 4: Generate final comprehensive report
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def generate_report(
    molecule_dir: flyte.io.Dir,
    properties_json: str,
    screening_json: str,
) -> str:
    """Generate a comprehensive drug screening report.

    Produces an executive summary, top candidate spotlight cards, property
    distributions, chemical diversity analysis, and final recommendation.

    Returns:
        JSON summary of screening results.
    """
    from rdkit import Chem

    await flyte.report.replace.aio(
        _wrap_report("<h2>Generating Final Report...</h2>"),
        do_flush=True,
    )

    props = json.loads(properties_json)
    screening = json.loads(screening_json)
    ranked = screening["ranked_molecules"]
    sim_matrix = screening["similarity_matrix"]
    sim_labels = screening["similarity_labels"]

    total = props["total"]
    lipinski_pass = props["lipinski_pass_count"]
    all_criteria = sum(1 for m in ranked if m["all_criteria_met"])
    top = ranked[0] if ranked else None

    html_parts = []

    # --- Executive Summary ---
    html_parts.append("<h2>Drug Molecule Screening Report</h2>")
    top_name = top["name"] if top else "N/A"
    top_score = f'{top["screening_score"]:.3f}' if top else "N/A"
    html_parts.append(
        f'<div class="card">'
        f'<h3 style="margin-top:0;color:#0e4f6e;">Executive Summary</h3>'
        f'<p style="font-size:1.05em;">'
        f'<strong>{total}</strong> molecules were screened against the target drug profile. '
        f'<strong>{lipinski_pass}</strong> passed Lipinski\'s Rule of Five, and '
        f'<strong>{all_criteria}</strong> met all screening criteria. '
        f'The top candidate is <strong style="color:#0891b2;">{top_name}</strong> '
        f'with a screening score of <strong>{top_score}</strong>.</p>'
        f'</div>'
    )

    # Stat grid
    html_parts.append('<div class="stat-grid">')
    for val, label in [
        (str(total), "Molecules Screened"),
        (str(lipinski_pass), "Lipinski Passes"),
        (str(all_criteria), "All Criteria Met"),
        (top_score, "Top Score"),
        (f'{props["avg_mw"]:.0f} Da', "Avg. Molecular Weight"),
        (f'{props["avg_logp"]:.2f}', "Avg. LogP"),
    ]:
        html_parts.append(
            f'<div class="stat"><div class="value">{val}</div>'
            f'<div class="label">{label}</div></div>'
        )
    html_parts.append("</div>")

    # --- Top 3 Candidate Spotlights ---
    html_parts.append("<h2>Top Candidate Spotlights</h2>")

    for rank, m in enumerate(ranked[:3], 1):
        mol = Chem.MolFromSmiles(m["smiles"])
        img_uri = _mol_to_data_uri(mol, size=(300, 300)) if mol else ""

        medal = ["gold", "silver", "#cd7f32"][rank - 1]
        medal_emoji = ["1st", "2nd", "3rd"][rank - 1]

        lip_badges = ""
        for rule, key in [("MW", "mw_ok"), ("LogP", "logp_ok"),
                          ("HBD", "hbd_ok"), ("HBA", "hba_ok")]:
            ok = m["lipinski"].get(key, False)
            cls = "badge-success" if ok else "badge-danger"
            lip_badges += f'<span class="badge {cls}" style="margin:2px;">{rule}</span> '

        html_parts.append(
            f'<div class="molecule-card" style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap;">'
            f'<div style="text-align:center;min-width:180px;">'
            f'<div style="font-size:1.6em;font-weight:800;color:{medal};">{medal_emoji}</div>'
            f'<img src="{img_uri}" style="width:200px;height:200px;object-fit:contain;border-radius:8px;'
            f'border:2px solid #a5f3fc;"/>'
            f'<div style="font-weight:700;font-size:1.1em;color:#0e4f6e;margin-top:8px;">{m["name"]}</div>'
            f'</div>'
            f'<div style="flex:1;min-width:280px;">'
            f'<table style="margin:0;">'
            f'<tr><td><strong>SMILES</strong></td><td style="font-family:monospace;font-size:0.8em;word-break:break-all;">{m["smiles"]}</td></tr>'
            f'<tr><td><strong>Screening Score</strong></td><td style="font-weight:700;color:#0891b2;font-size:1.1em;">{m["screening_score"]:.3f}</td></tr>'
            f'<tr><td><strong>Molecular Weight</strong></td><td>{m["mw"]:.1f} Da</td></tr>'
            f'<tr><td><strong>LogP</strong></td><td>{m["logp"]:.2f}</td></tr>'
            f'<tr><td><strong>H-Bond Donors</strong></td><td>{m["hbd"]}</td></tr>'
            f'<tr><td><strong>H-Bond Acceptors</strong></td><td>{m["hba"]}</td></tr>'
            f'<tr><td><strong>TPSA</strong></td><td>{m["tpsa"]:.1f} A&sup2;</td></tr>'
            f'<tr><td><strong>Rotatable Bonds</strong></td><td>{m["rotatable_bonds"]}</td></tr>'
            f'<tr><td><strong>QED</strong></td><td>{m["qed"]:.4f}</td></tr>'
            f'<tr><td><strong>Lipinski Compliance</strong></td><td>{lip_badges}</td></tr>'
            f'</table>'
            f'</div>'
            f'</div>'
        )

    # --- Property Distribution (box-plot style as bars with min/max/median) ---
    html_parts.append("<h2>Property Distributions</h2>")

    prop_keys = [("mw", "Molecular Weight (Da)"), ("logp", "LogP"),
                 ("tpsa", "TPSA"), ("qed", "QED Score")]
    for key, label in prop_keys:
        vals = sorted([m[key] for m in ranked])
        n = len(vals)
        if n == 0:
            continue
        v_min = vals[0]
        v_max = vals[-1]
        median = vals[n // 2] if n % 2 == 1 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        q1 = vals[n // 4] if n >= 4 else v_min
        q3 = vals[3 * n // 4] if n >= 4 else v_max

        # Simple horizontal box-plot as SVG
        box_w = 500
        box_h = 50
        margin_l = 10
        v_range = v_max - v_min or 1

        def sx(v):
            return margin_l + ((v - v_min) / v_range) * (box_w - 2 * margin_l)

        box_svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {box_w} {box_h}" '
            f'style="width:100%;max-width:{box_w}px;height:auto;">'
            f'<rect width="{box_w}" height="{box_h}" fill="#fff" rx="4"/>'
            # Whisker line
            f'<line x1="{sx(v_min):.1f}" y1="25" x2="{sx(v_max):.1f}" y2="25" '
            f'stroke="#94a3b8" stroke-width="1.5"/>'
            # Min whisker
            f'<line x1="{sx(v_min):.1f}" y1="18" x2="{sx(v_min):.1f}" y2="32" '
            f'stroke="#94a3b8" stroke-width="1.5"/>'
            # Max whisker
            f'<line x1="{sx(v_max):.1f}" y1="18" x2="{sx(v_max):.1f}" y2="32" '
            f'stroke="#94a3b8" stroke-width="1.5"/>'
            # IQR box
            f'<rect x="{sx(q1):.1f}" y="14" width="{sx(q3) - sx(q1):.1f}" height="22" '
            f'fill="#a5f3fc" stroke="#0891b2" stroke-width="1.5" rx="3"/>'
            # Median line
            f'<line x1="{sx(median):.1f}" y1="12" x2="{sx(median):.1f}" y2="38" '
            f'stroke="#0e4f6e" stroke-width="2"/>'
            # Labels
            f'<text x="{sx(v_min):.1f}" y="46" text-anchor="middle" font-size="9" fill="#6c757d">{v_min:.1f}</text>'
            f'<text x="{sx(median):.1f}" y="10" text-anchor="middle" font-size="9" fill="#0e4f6e" font-weight="600">{median:.1f}</text>'
            f'<text x="{sx(v_max):.1f}" y="46" text-anchor="middle" font-size="9" fill="#6c757d">{v_max:.1f}</text>'
            f'</svg>'
        )
        html_parts.append(
            f'<div style="margin:8px 0;"><strong style="color:#155e75;">{label}</strong>'
            f'<div class="chart-container" style="padding:8px;">{box_svg}</div></div>'
        )

    # --- Chemical Diversity ---
    html_parts.append("<h2>Chemical Diversity Analysis</h2>")

    if sim_matrix and len(sim_matrix) > 1:
        # Compute average pairwise similarity (off-diagonal)
        n_mols = len(sim_matrix)
        off_diag = []
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                off_diag.append(sim_matrix[i][j])

        avg_sim = sum(off_diag) / len(off_diag) if off_diag else 0
        max_sim = max(off_diag) if off_diag else 0
        min_sim = min(off_diag) if off_diag else 0

        # Find most similar pair
        best_i, best_j = 0, 1
        best_val = 0
        for i in range(n_mols):
            for j in range(i + 1, n_mols):
                if sim_matrix[i][j] > best_val:
                    best_val = sim_matrix[i][j]
                    best_i, best_j = i, j

        html_parts.append('<div class="stat-grid">')
        html_parts.append(
            f'<div class="stat"><div class="value">{avg_sim:.3f}</div>'
            f'<div class="label">Avg. Pairwise Similarity</div></div>'
        )
        html_parts.append(
            f'<div class="stat"><div class="value">{min_sim:.3f}</div>'
            f'<div class="label">Min Similarity</div></div>'
        )
        html_parts.append(
            f'<div class="stat"><div class="value">{max_sim:.3f}</div>'
            f'<div class="label">Max Similarity</div></div>'
        )
        html_parts.append("</div>")

        diversity_text = "highly diverse" if avg_sim < 0.3 else "moderately diverse" if avg_sim < 0.5 else "relatively similar"
        html_parts.append(
            f'<div class="note">'
            f'The library is <strong>{diversity_text}</strong> (avg. Tanimoto = {avg_sim:.3f}). '
            f'The most similar pair is <strong>{sim_labels[best_i]}</strong> and '
            f'<strong>{sim_labels[best_j]}</strong> (similarity = {best_val:.3f}).</div>'
        )

    # --- Recommendation ---
    html_parts.append("<h2>Recommendation</h2>")
    if top:
        html_parts.append(
            f'<div class="card">'
            f'<h3 style="margin-top:0;color:#0891b2;">Top Candidate: {top["name"]}</h3>'
            f'<p>Based on the virtual screening analysis, <strong>{top["name"]}</strong> '
            f'achieved the highest composite screening score of <strong>{top["screening_score"]:.3f}</strong>. '
        )

        reasons = []
        if top["lipinski_pass"]:
            reasons.append("full Lipinski Rule of Five compliance")
        if top["qed"] > 0.5:
            reasons.append(f"high drug-likeness (QED = {top['qed']:.3f})")
        if top.get("all_criteria_met"):
            reasons.append("all target profile criteria met")
        if top["mw"] <= 500:
            reasons.append(f"favorable molecular weight ({top['mw']:.1f} Da)")

        if reasons:
            html_parts.append(
                f'This candidate stands out due to: {", ".join(reasons)}.</p>'
            )
        else:
            html_parts.append("</p>")

        # Runner-up mentions
        if len(ranked) >= 2:
            html_parts.append(
                f'<p style="font-size:0.9em;color:#6c757d;">Runner-up candidates: '
            )
            runners = []
            for m in ranked[1:4]:
                runners.append(f'{m["name"]} (score: {m["screening_score"]:.3f})')
            html_parts.append(", ".join(runners) + ".</p>")

        html_parts.append("</div>")

    # Final note
    html_parts.append(
        '<div class="note">'
        "This is a virtual screening analysis. All candidates should undergo "
        "further computational validation (molecular dynamics, docking) and "
        "experimental testing before advancing to clinical trials.</div>"
    )

    await flyte.report.replace.aio(
        _wrap_report("\n".join(html_parts)),
        do_flush=True,
    )

    # JSON summary
    summary = {
        "total_screened": total,
        "lipinski_passes": lipinski_pass,
        "all_criteria_met": all_criteria,
        "top_candidate": top["name"] if top else None,
        "top_score": top["screening_score"] if top else None,
        "top_3": [
            {"name": m["name"], "score": m["screening_score"]}
            for m in ranked[:3]
        ],
    }
    return json.dumps(summary)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    molecules_json: str = "",
    target_profile: str = "",
) -> str:
    """Virtual drug molecule screening pipeline.

    Parses a molecular library, computes physicochemical properties,
    screens candidates against a target drug profile, and generates
    a comprehensive visual report with ranked candidates.

    Args:
        molecules_json: JSON mapping molecule names to SMILES strings.
            Defaults to a curated library of ~15 well-known drugs.
        target_profile: JSON with desired property ranges
            (e.g. {"mw": [150, 500], "logp": [-0.5, 5]}).
            Defaults to standard drug-like criteria.

    Returns:
        JSON summary of screening results.
    """
    await flyte.report.replace.aio(
        _wrap_report("<h2>Step 1/4: Loading molecules...</h2>"),
        do_flush=True,
    )
    mol_dir = await load_molecules(molecules_json=molecules_json)

    await flyte.report.replace.aio(
        _wrap_report("<h2>Step 2/4: Computing properties...</h2>"),
        do_flush=True,
    )
    props_json = await compute_properties(molecule_dir=mol_dir)

    await flyte.report.replace.aio(
        _wrap_report("<h2>Step 3/4: Screening candidates...</h2>"),
        do_flush=True,
    )
    screening_json = await screen_candidates(
        properties_json=props_json,
        target_profile=target_profile,
    )

    await flyte.report.replace.aio(
        _wrap_report("<h2>Step 4/4: Generating final report...</h2>"),
        do_flush=True,
    )
    summary = await generate_report(
        molecule_dir=mol_dir,
        properties_json=props_json,
        screening_json=screening_json,
    )

    return summary
