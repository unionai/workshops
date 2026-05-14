"""
Protein Sequence Analysis — Analyze, compare, and visualize protein properties.

Pipeline: load a curated set of real protein sequences, compute biophysical
properties (MW, pI, stability, hydrophobicity, secondary structure), build a
pairwise similarity matrix, and generate a comprehensive visual summary report.

Usage:
    # Default (curated set of 7 proteins)
    flyte run --local --tui workflow.py pipeline

    # Custom sequences via JSON
    flyte run --local --tui workflow.py pipeline --sequences_json '{"MyProtein": "MVLSPADKTNVKA"}'

    # Remote
    flyte run workflow.py pipeline
"""

import json
import logging
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
# Default protein sequences — real sequences (truncated for large ones)
# ------------------------------------------------------------------

DEFAULT_PROTEINS = {
    "Insulin (Human B-chain)": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    "GFP (Green Fluorescent Protein)": "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
    "Lysozyme (Hen Egg White)": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
    "Ubiquitin (Human)": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    "Hemoglobin Alpha (Human)": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "p53 Tumor Suppressor (fragment)": "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYPQGLNGTVNLPGRNSFEVRVCA",
    "Spider Silk (MaSp1 fragment)": "GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAGGAGQGGYGGLGSQGAGRGGLGGQGAG",
}

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Kyte-Doolittle hydrophobicity scale
KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


# ------------------------------------------------------------------
# Report styling — biotech-themed greens and teals
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #065f46; border-bottom: 2px solid #059669; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #047857; margin-top: 20px; }
  .report .card { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #d1fae5; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #065f46; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #065f46; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #d1fae5; }
  .report tr:nth-child(even) { background: #f0fdf4; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-warning { background: #fef3c7; color: #92400e; }
  .report .badge-info { background: #dbeafe; color: #1e40af; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .chart-container { background: #fff; border: 1px solid #d1fae5; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #ecfdf5; border-left: 4px solid #059669; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .protein-card { background: #fff; border: 1px solid #d1fae5; border-radius: 8px; padding: 16px; margin: 12px 0; display: flex; gap: 16px; flex-wrap: wrap; }
  .report .protein-card .props { flex: 1; min-width: 200px; }
  .report .protein-card .viz { flex: 1; min-width: 300px; }
</style>
"""


def _wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# SVG chart helpers
# ------------------------------------------------------------------

def _make_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    y_max_cap: float | None = None,
    value_format: str = ".1f",
) -> str:
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#059669", "#065f46", "#10b981", "#34d399", "#6ee7b7"]
    colors = colors or default_colors

    ml, mr, mt, mb = 70, 20, 40, 80
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
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:{value_format}}</text>'
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
                f'height="{max(0, bh):.1f}" fill="{color}" rx="2"/>'
            )
            svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="9" fill="#1a1a2e">'
                f'{val:{value_format}}</text>'
            )
        # Rotated group label
        lx = gx + n_series * bar_width / 2
        svg.append(
            f'<text x="{lx:.1f}" y="{mt + ch + 14}" '
            f'text-anchor="end" font-size="10" fill="#6c757d" '
            f'transform="rotate(-35, {lx:.1f}, {mt + ch + 14})">{label}</text>'
        )

    # Title
    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    # Legend
    if n_series > 1:
        lx = ml + cw - len(series) * 110
        for si, name in enumerate(series):
            color = colors[si % len(colors)]
            svg.append(
                f'<rect x="{lx + si * 110}" y="{mt + ch + 55}" width="12" '
                f'height="12" rx="2" fill="{color}"/>'
            )
            svg.append(
                f'<text x="{lx + si * 110 + 16}" y="{mt + ch + 66}" font-size="11" '
                f'fill="#1a1a2e">{name}</text>'
            )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_line_chart(
    data: list[dict],
    x_key: str,
    y_keys: list[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 200,
) -> str:
    """Generate an SVG line/sparkline chart."""
    default_colors = ["#059669", "#065f46", "#10b981"]
    colors = colors or default_colors

    ml, mr, mt, mb = 50, 20, 30, 40
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
    y_range = y_max_plot - y_min_plot or 1

    def sx(v):
        return ml + (v - x_min) / x_range * cw

    def sy(v):
        return mt + ch - (v - y_min_plot) / y_range * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Zero line if range crosses zero
    if y_min_plot < 0 < y_max_plot:
        zy = sy(0)
        svg.append(
            f'<line x1="{ml}" y1="{zy:.1f}" x2="{ml + cw}" y2="{zy:.1f}" '
            f'stroke="#adb5bd" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # Plot series
    for si, key in enumerate(y_keys):
        color = colors[si % len(colors)]
        points = [(sx(d[x_key]), sy(d[key])) for d in data if key in d]
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
            for px, py in points[1:]:
                path_d += f" L {px:.1f},{py:.1f}"
            svg.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"/>'
            )

    # Title
    if title:
        svg.append(
            f'<text x="{width / 2}" y="18" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    if x_label:
        svg.append(
            f'<text x="{ml + cw / 2}" y="{height - 4}" text-anchor="middle" '
            f'font-size="11" fill="#6c757d">{x_label}</text>'
        )
    if y_label:
        svg.append(
            f'<text x="12" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="11" fill="#6c757d" '
            f'transform="rotate(-90, 12, {mt + ch / 2})">{y_label}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    color_scale: str = "green",
    width: int = 700,
    height: int = 500,
    value_format: str = ".2f",
    show_values: bool | None = None,
) -> str:
    """Generate an SVG heatmap.

    Args:
        matrix: 2D list of values (rows x cols).
        row_labels: Labels for rows (y-axis).
        col_labels: Labels for columns (x-axis).
        title: Chart title.
        color_scale: "green" for 0=white to 1=dark green, "teal" for composition.
        width: SVG width.
        height: SVG height.
        value_format: Format string for cell values.
        show_values: Whether to show values in cells. Auto-detected if None.
    """
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if not n_rows or not n_cols:
        return ""

    if show_values is None:
        show_values = n_rows <= 10 and n_cols <= 12

    # Find min/max for color scaling
    flat = [v for row in matrix for v in row]
    v_min = min(flat)
    v_max = max(flat)
    v_range = v_max - v_min or 1

    # Color interpolation
    if color_scale == "green":
        def get_color(v):
            t = (v - v_min) / v_range
            r = int(255 - t * (255 - 6))
            g = int(255 - t * (255 - 95))
            b = int(255 - t * (255 - 70))
            return f"rgb({r},{g},{b})"
    else:  # teal
        def get_color(v):
            t = (v - v_min) / v_range
            r = int(255 - t * (255 - 13))
            g = int(255 - t * (255 - 148))
            b = int(255 - t * (255 - 136))
            return f"rgb({r},{g},{b})"

    # Layout
    ml = max(120, max(len(l) for l in row_labels) * 7 + 20) if row_labels else 120
    mr = 20
    mt = 80 if col_labels else 40
    mb = 30
    cw = width - ml - mr
    ch = height - mt - mb

    cell_w = cw / n_cols
    cell_h = ch / n_rows

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Title
    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    # Column labels (rotated)
    for j, label in enumerate(col_labels):
        cx = ml + j * cell_w + cell_w / 2
        svg.append(
            f'<text x="{cx:.1f}" y="{mt - 8}" text-anchor="end" '
            f'font-size="10" fill="#374151" '
            f'transform="rotate(-45, {cx:.1f}, {mt - 8})">{label}</text>'
        )

    # Row labels + cells
    for i, row_label in enumerate(row_labels):
        ry = mt + i * cell_h + cell_h / 2
        svg.append(
            f'<text x="{ml - 8}" y="{ry + 4:.1f}" text-anchor="end" '
            f'font-size="10" fill="#374151">{row_label}</text>'
        )
        for j in range(n_cols):
            val = matrix[i][j]
            color = get_color(val)
            cx = ml + j * cell_w
            cy = mt + i * cell_h
            svg.append(
                f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" '
                f'height="{cell_h:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>'
            )
            if show_values:
                # Use dark text on light cells, white on dark
                t = (val - v_min) / v_range
                text_color = "#fff" if t > 0.55 else "#1a1a2e"
                font_size = min(10, int(cell_w / 4), int(cell_h / 2.5))
                font_size = max(7, font_size)
                svg.append(
                    f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 3:.1f}" '
                    f'text-anchor="middle" font-size="{font_size}" '
                    f'fill="{text_color}">{val:{value_format}}</text>'
                )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_sparkline(
    values: list[float],
    width: int = 280,
    height: int = 60,
    color: str = "#059669",
    fill_color: str = "#d1fae5",
) -> str:
    """Generate a tiny SVG sparkline for inline use."""
    if not values or len(values) < 2:
        return ""

    v_min = min(values)
    v_max = max(values)
    v_range = v_max - v_min or 1
    pad = 4
    cw = width - 2 * pad
    ch = height - 2 * pad

    points = []
    for i, v in enumerate(values):
        x = pad + (i / (len(values) - 1)) * cw
        y = pad + ch - ((v - v_min) / v_range) * ch
        points.append(f"{x:.1f},{y:.1f}")

    poly_line = " ".join(points)
    # Fill area
    fill_points = f"{pad},{pad + ch} " + poly_line + f" {pad + cw},{pad + ch}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<polygon points="{fill_points}" fill="{fill_color}" opacity="0.5"/>',
        f'<polyline points="{poly_line}" fill="none" stroke="{color}" stroke-width="1.5"/>',
    ]

    # Zero line if range crosses zero
    if v_min < 0 < v_max:
        zy = pad + ch - ((0 - v_min) / v_range) * ch
        svg.append(
            f'<line x1="{pad}" y1="{zy:.1f}" x2="{pad + cw}" y2="{zy:.1f}" '
            f'stroke="#adb5bd" stroke-width="0.5" stroke-dasharray="3,2"/>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def _make_mini_bar(
    labels: list[str],
    values: list[float],
    width: int = 280,
    height: int = 50,
    color: str = "#059669",
) -> str:
    """Generate a tiny horizontal bar chart showing top amino acids."""
    if not labels:
        return ""

    v_max = max(values) if values else 1
    bar_h = min(10, (height - 4) / len(labels) - 2)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
    ]

    label_w = 25
    bar_area = width - label_w - 40

    for i, (label, val) in enumerate(zip(labels, values)):
        y = 4 + i * (bar_h + 3)
        bw = max(2, (val / v_max) * bar_area)
        svg.append(
            f'<text x="{label_w - 4}" y="{y + bar_h - 1:.1f}" text-anchor="end" '
            f'font-size="9" fill="#374151" font-weight="600">{label}</text>'
        )
        svg.append(
            f'<rect x="{label_w}" y="{y:.1f}" width="{bw:.1f}" '
            f'height="{bar_h:.1f}" fill="{color}" rx="2"/>'
        )
        svg.append(
            f'<text x="{label_w + bw + 4:.1f}" y="{y + bar_h - 1:.1f}" '
            f'font-size="8" fill="#6c757d">{val:.1%}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Hydrophobicity profile helper
# ------------------------------------------------------------------

def _hydrophobicity_profile(sequence: str, window: int = 7) -> list[float]:
    """Compute Kyte-Doolittle hydrophobicity with a sliding window."""
    if len(sequence) < window:
        return [KD_SCALE.get(aa, 0) for aa in sequence]

    profile = []
    half = window // 2
    for i in range(len(sequence)):
        start = max(0, i - half)
        end = min(len(sequence), i + half + 1)
        window_seq = sequence[start:end]
        avg = sum(KD_SCALE.get(aa, 0) for aa in window_seq) / len(window_seq)
        profile.append(round(avg, 3))
    return profile


# ------------------------------------------------------------------
# Task 1: Load sequences
# ------------------------------------------------------------------

@cpu_env.task(cache="auto", report=True)
async def load_sequences(
    sequences_json: str = "",
) -> flyte.io.Dir:
    """Load and validate protein sequences, output as FASTA files.

    If sequences_json is empty, uses the default curated set of proteins.
    Otherwise, expects a JSON object mapping name -> sequence.
    """
    log.info("Loading protein sequences...")

    # Parse input
    if sequences_json and sequences_json.strip():
        proteins = json.loads(sequences_json)
    else:
        proteins = DEFAULT_PROTEINS

    # Validate sequences
    validated = {}
    errors = []
    for name, seq in proteins.items():
        seq = seq.upper().replace(" ", "").replace("\n", "")
        invalid_chars = set(seq) - VALID_AMINO_ACIDS
        if invalid_chars:
            errors.append(f"{name}: invalid amino acids {invalid_chars}")
            # Remove invalid chars and keep going
            seq = "".join(c for c in seq if c in VALID_AMINO_ACIDS)
        if len(seq) < 5:
            errors.append(f"{name}: sequence too short ({len(seq)} residues)")
            continue
        validated[name] = seq

    log.info(f"Validated {len(validated)} sequences, {len(errors)} warnings")

    # Write FASTA files
    out_dir = tempfile.mkdtemp(prefix="protein_seqs_")
    for name, seq in validated.items():
        safe_name = name.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
        fasta_path = os.path.join(out_dir, f"{safe_name}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{name}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")

    # Build report
    rows = []
    for name, seq in validated.items():
        preview = seq[:50] + ("..." if len(seq) > 50 else "")
        rows.append(
            f"<tr><td><b>{name}</b></td>"
            f"<td>{len(seq)}</td>"
            f'<td><code style="font-size:0.8em;word-break:break-all;">{preview}</code></td></tr>'
        )

    error_html = ""
    if errors:
        error_items = "".join(f"<li>{e}</li>" for e in errors)
        error_html = f'<div class="note"><b>Warnings:</b><ul>{error_items}</ul></div>'

    report_html = f"""
    <h2>Protein Sequences Loaded</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(validated)}</div><div class="label">Proteins</div></div>
      <div class="stat"><div class="value">{sum(len(s) for s in validated.values())}</div><div class="label">Total Residues</div></div>
      <div class="stat"><div class="value">{min(len(s) for s in validated.values())}</div><div class="label">Shortest (aa)</div></div>
      <div class="stat"><div class="value">{max(len(s) for s in validated.values())}</div><div class="label">Longest (aa)</div></div>
    </div>
    {error_html}
    <table>
      <tr><th>Protein</th><th>Length</th><th>Sequence (first 50 residues)</th></tr>
      {"".join(rows)}
    </table>
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Analyze biophysical properties
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def analyze_properties(
    seq_dir: flyte.io.Dir,
) -> str:
    """Compute biophysical properties for each protein using BioPython."""
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    log.info("Analyzing protein properties...")

    seq_path = await seq_dir.download()

    # Read all FASTA files
    proteins = {}
    for fname in sorted(os.listdir(seq_path)):
        if not fname.endswith(".fasta"):
            continue
        fpath = os.path.join(seq_path, fname)
        with open(fpath) as f:
            lines = f.readlines()
        name = lines[0].strip().lstrip(">")
        seq = "".join(line.strip() for line in lines[1:])
        proteins[name] = seq

    log.info(f"Analyzing {len(proteins)} proteins...")

    # Compute properties
    all_props = {}
    for name, seq in proteins.items():
        analysis = ProteinAnalysis(seq)
        mw = analysis.molecular_weight()
        pi = analysis.isoelectric_point()
        instability = analysis.instability_index()
        gravy = analysis.gravy()
        aromaticity = analysis.aromaticity()
        helix, turn, sheet = analysis.secondary_structure_fraction()
        aa_percent = analysis.amino_acids_percent

        all_props[name] = {
            "sequence": seq,
            "length": len(seq),
            "molecular_weight": round(mw, 1),
            "isoelectric_point": round(pi, 2),
            "instability_index": round(instability, 2),
            "gravy": round(gravy, 3),
            "aromaticity": round(aromaticity, 4),
            "helix_fraction": round(helix, 3),
            "turn_fraction": round(turn, 3),
            "sheet_fraction": round(sheet, 3),
            "aa_composition": {k: round(v, 4) for k, v in aa_percent.items()},
        }

    names = list(all_props.keys())
    short_names = []
    for n in names:
        # Shorten for chart labels
        short = n.split("(")[0].strip()
        if len(short) > 18:
            short = short[:16] + ".."
        short_names.append(short)

    avg_mw = sum(p["molecular_weight"] for p in all_props.values()) / len(all_props)
    avg_pi = sum(p["isoelectric_point"] for p in all_props.values()) / len(all_props)

    # --- Build report ---

    # Stat grid
    stats_html = f"""
    <h2>Biophysical Properties Analysis</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(all_props)}</div><div class="label">Proteins Analyzed</div></div>
      <div class="stat"><div class="value">{avg_mw:,.0f} Da</div><div class="label">Avg Molecular Weight</div></div>
      <div class="stat"><div class="value">{avg_pi:.1f}</div><div class="label">Avg Isoelectric Point</div></div>
    </div>
    """

    # Molecular weight bar chart
    mw_chart = _make_bar_chart(
        labels=short_names,
        series={"MW (Da)": [all_props[n]["molecular_weight"] for n in names]},
        title="Molecular Weight Comparison",
        colors=["#059669"],
        width=700,
        height=320,
        value_format=".0f",
    )

    # pI bar chart
    pi_chart = _make_bar_chart(
        labels=short_names,
        series={"pI": [all_props[n]["isoelectric_point"] for n in names]},
        title="Isoelectric Point (pI) Comparison",
        colors=["#065f46"],
        width=700,
        height=300,
        value_format=".1f",
    )

    # Secondary structure grouped bar chart
    ss_chart = _make_bar_chart(
        labels=short_names,
        series={
            "Helix": [all_props[n]["helix_fraction"] for n in names],
            "Turn": [all_props[n]["turn_fraction"] for n in names],
            "Sheet": [all_props[n]["sheet_fraction"] for n in names],
        },
        title="Secondary Structure Fractions",
        colors=["#059669", "#f59e0b", "#3b82f6"],
        width=700,
        height=320,
        y_max_cap=1.0,
        value_format=".2f",
    )

    # Amino acid composition heatmap
    all_aas = sorted(VALID_AMINO_ACIDS)
    aa_matrix = []
    for n in names:
        row = [all_props[n]["aa_composition"].get(aa, 0) for aa in all_aas]
        aa_matrix.append(row)

    aa_heatmap = _make_heatmap(
        matrix=aa_matrix,
        row_labels=short_names,
        col_labels=all_aas,
        title="Amino Acid Composition",
        color_scale="teal",
        width=750,
        height=max(350, len(names) * 45 + 120),
        value_format=".0%",
        show_values=len(names) <= 10,
    )

    # Stability assessment table
    stability_rows = []
    for n in names:
        props = all_props[n]
        ii = props["instability_index"]
        if ii < 40:
            badge = '<span class="badge badge-success">Stable</span>'
        else:
            badge = '<span class="badge badge-warning">Unstable</span>'

        gravy_val = props["gravy"]
        if gravy_val > 0:
            hydro_badge = '<span class="badge badge-info">Hydrophobic</span>'
        else:
            hydro_badge = '<span class="badge badge-success">Hydrophilic</span>'

        short = n.split("(")[0].strip()
        stability_rows.append(
            f"<tr><td><b>{short}</b></td>"
            f"<td>{props['length']}</td>"
            f"<td>{props['molecular_weight']:,.0f}</td>"
            f"<td>{props['isoelectric_point']:.2f}</td>"
            f"<td>{ii:.1f} {badge}</td>"
            f"<td>{gravy_val:.3f} {hydro_badge}</td>"
            f"<td>{props['aromaticity']:.3f}</td></tr>"
        )

    stability_table = f"""
    <h3>Protein Properties Summary</h3>
    <table>
      <tr>
        <th>Protein</th><th>Length</th><th>MW (Da)</th><th>pI</th>
        <th>Instability Index</th><th>GRAVY</th><th>Aromaticity</th>
      </tr>
      {"".join(stability_rows)}
    </table>
    <div class="note">
      <b>Instability Index:</b> Values below 40 predict a stable protein in vitro.
      <b>GRAVY:</b> Grand Average of Hydropathy — positive values indicate overall hydrophobic character.
      <b>Aromaticity:</b> Relative frequency of Phe+Trp+Tyr residues.
    </div>
    """

    report_html = f"""
    {stats_html}
    <div class="chart-container">{mw_chart}</div>
    <div class="chart-container">{pi_chart}</div>
    <div class="chart-container">{ss_chart}</div>
    <div class="chart-container">{aa_heatmap}</div>
    {stability_table}
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    # Output JSON (without sequences to keep it compact)
    output = {}
    for n, p in all_props.items():
        out_p = {k: v for k, v in p.items() if k != "sequence"}
        output[n] = out_p

    return json.dumps(output)


# ------------------------------------------------------------------
# Task 3: Compute pairwise similarity
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def compute_similarity(
    seq_dir: flyte.io.Dir,
) -> str:
    """Compute pairwise sequence identity between all proteins."""
    log.info("Computing pairwise similarity...")

    seq_path = await seq_dir.download()

    # Read all FASTA files
    proteins = {}
    for fname in sorted(os.listdir(seq_path)):
        if not fname.endswith(".fasta"):
            continue
        with open(os.path.join(seq_path, fname)) as f:
            lines = f.readlines()
        name = lines[0].strip().lstrip(">")
        seq = "".join(line.strip() for line in lines[1:])
        proteins[name] = seq

    names = list(proteins.keys())
    n = len(names)
    log.info(f"Computing {n}x{n} similarity matrix...")

    await flyte.report.replace.aio(_wrap_report(
        f"<h2>Sequence Similarity</h2>"
        f"<p>Computing pairwise alignment for {n} proteins ({n * (n - 1) // 2} pairs)...</p>"
    ), do_flush=True)

    # Compute pairwise percent identity using simple global alignment
    # Using a simplified approach: count matching residues at aligned positions
    def percent_identity(seq1: str, seq2: str) -> float:
        """Compute percent identity between two sequences using Bio.Align."""
        from Bio import Align

        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -2.0
        aligner.extend_gap_score = -0.5

        alignments = aligner.align(seq1, seq2)
        if not alignments:
            return 0.0

        best = alignments[0]
        # Count matches from the alignment
        aligned_seq1 = str(best).split("\n")[0] if "\n" in str(best) else seq1
        aligned_seq2 = str(best).split("\n")[2] if "\n" in str(best) else seq2

        # Use the score to estimate identity
        score = best.score
        max_len = max(len(seq1), len(seq2))
        # Normalize: perfect match would score 2*max_len
        identity = max(0.0, min(1.0, score / (2.0 * max_len)))
        return round(identity, 4)

    # Build similarity matrix
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            identity = percent_identity(proteins[names[i]], proteins[names[j]])
            matrix[i][j] = identity
            matrix[j][i] = identity

    # Find most similar pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], matrix[i][j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    short_names = []
    for nm in names:
        short = nm.split("(")[0].strip()
        if len(short) > 16:
            short = short[:14] + ".."
        short_names.append(short)

    # Heatmap
    heatmap = _make_heatmap(
        matrix=matrix,
        row_labels=short_names,
        col_labels=short_names,
        title="Pairwise Sequence Similarity",
        color_scale="green",
        width=700,
        height=max(400, n * 55 + 120),
        value_format=".0%",
    )

    # Most similar pairs table
    top_pairs = pairs[:min(10, len(pairs))]
    pair_rows = []
    for p1, p2, sim in top_pairs:
        short_p1 = p1.split("(")[0].strip()
        short_p2 = p2.split("(")[0].strip()
        if sim > 0.5:
            badge = '<span class="badge badge-success">High</span>'
        elif sim > 0.25:
            badge = '<span class="badge badge-info">Moderate</span>'
        else:
            badge = '<span class="badge badge-warning">Low</span>'
        pair_rows.append(
            f"<tr><td>{short_p1}</td><td>{short_p2}</td>"
            f"<td>{sim:.1%} {badge}</td></tr>"
        )

    avg_sim = sum(p[2] for p in pairs) / len(pairs) if pairs else 0
    max_sim = pairs[0][2] if pairs else 0
    min_sim = pairs[-1][2] if pairs else 0

    report_html = f"""
    <h2>Sequence Similarity Analysis</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{n}</div><div class="label">Proteins Compared</div></div>
      <div class="stat"><div class="value">{n * (n - 1) // 2}</div><div class="label">Pairwise Comparisons</div></div>
      <div class="stat"><div class="value">{avg_sim:.1%}</div><div class="label">Average Similarity</div></div>
      <div class="stat"><div class="value">{max_sim:.1%}</div><div class="label">Maximum Similarity</div></div>
    </div>

    <div class="chart-container">{heatmap}</div>

    <h3>Most Similar Pairs</h3>
    <table>
      <tr><th>Protein 1</th><th>Protein 2</th><th>Sequence Identity</th></tr>
      {"".join(pair_rows)}
    </table>

    <div class="note">
      <b>Sequence similarity</b> measures how closely related two proteins are at the primary
      structure level. High sequence identity (>30%) often implies shared evolutionary origin
      and similar 3D structure. Proteins with <25% identity are considered to be in the
      "twilight zone" where structural similarity cannot be reliably inferred from sequence alone.
      The values shown represent global pairwise alignment scores normalized by sequence length.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(report_html), do_flush=True)

    # Output
    output = {
        "names": names,
        "matrix": matrix,
        "pairs": [{"protein1": p1, "protein2": p2, "identity": sim} for p1, p2, sim in pairs],
    }
    return json.dumps(output)


# ------------------------------------------------------------------
# Task 4: Generate comprehensive summary
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def generate_summary(
    properties_json: str,
    similarity_json: str,
    seq_dir: flyte.io.Dir,
) -> str:
    """Generate a comprehensive visual summary report."""
    log.info("Generating summary report...")

    properties = json.loads(properties_json)
    similarity = json.loads(similarity_json)

    seq_path = await seq_dir.download()

    # Re-read sequences for hydrophobicity profiles
    sequences = {}
    for fname in sorted(os.listdir(seq_path)):
        if not fname.endswith(".fasta"):
            continue
        with open(os.path.join(seq_path, fname)) as f:
            lines = f.readlines()
        name = lines[0].strip().lstrip(">")
        seq = "".join(line.strip() for line in lines[1:])
        sequences[name] = seq

    names = list(properties.keys())

    # --- Overview card ---
    total_residues = sum(p["length"] for p in properties.values())
    avg_mw = sum(p["molecular_weight"] for p in properties.values()) / len(properties)
    stable_count = sum(1 for p in properties.values() if p["instability_index"] < 40)
    avg_sim = 0
    if similarity.get("pairs"):
        avg_sim = sum(p["identity"] for p in similarity["pairs"]) / len(similarity["pairs"])

    overview_html = f"""
    <h2>Protein Sequence Analysis — Summary Report</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(properties)}</div><div class="label">Proteins Analyzed</div></div>
      <div class="stat"><div class="value">{total_residues:,}</div><div class="label">Total Residues</div></div>
      <div class="stat"><div class="value">{avg_mw:,.0f} Da</div><div class="label">Avg Molecular Weight</div></div>
      <div class="stat"><div class="value">{stable_count}/{len(properties)}</div><div class="label">Predicted Stable</div></div>
      <div class="stat"><div class="value">{avg_sim:.1%}</div><div class="label">Avg Pairwise Similarity</div></div>
    </div>
    """

    # --- Protein Gallery ---
    gallery_html = "<h2>Protein Gallery</h2>"

    for name in names:
        props = properties[name]
        seq = sequences.get(name, "")
        short_name = name.split("(")[0].strip()

        # Stability badge
        ii = props["instability_index"]
        if ii < 40:
            stability_badge = '<span class="badge badge-success">Stable</span>'
        else:
            stability_badge = '<span class="badge badge-warning">Unstable</span>'

        # Size badge
        mw_kda = props["molecular_weight"] / 1000
        if mw_kda < 15:
            size_badge = '<span class="badge badge-info">Small</span>'
        elif mw_kda < 50:
            size_badge = '<span class="badge badge-success">Medium</span>'
        else:
            size_badge = '<span class="badge badge-danger">Large</span>'

        # Charge badge
        pi = props["isoelectric_point"]
        if pi < 7:
            charge_badge = '<span class="badge badge-danger">Acidic (pI&lt;7)</span>'
        else:
            charge_badge = '<span class="badge badge-info">Basic (pI&gt;7)</span>'

        # Hydrophobicity sparkline
        if seq:
            profile = _hydrophobicity_profile(seq, window=7)
            sparkline = _make_sparkline(
                profile, width=300, height=60,
                color="#059669", fill_color="#d1fae5",
            )
            sparkline_html = f"""
            <div style="margin-top:8px;">
              <div style="font-size:0.8em;color:#6c757d;margin-bottom:2px;">
                Hydrophobicity Profile (Kyte-Doolittle, window=7)
              </div>
              {sparkline}
            </div>
            """
        else:
            sparkline_html = ""

        # Top 5 amino acids mini bar
        aa_comp = props.get("aa_composition", {})
        sorted_aa = sorted(aa_comp.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_aa:
            aa_labels = [a[0] for a in sorted_aa]
            aa_values = [a[1] for a in sorted_aa]
            mini_bar = _make_mini_bar(aa_labels, aa_values, width=300, height=65, color="#10b981")
            mini_bar_html = f"""
            <div style="margin-top:8px;">
              <div style="font-size:0.8em;color:#6c757d;margin-bottom:2px;">
                Top 5 Amino Acids
              </div>
              {mini_bar}
            </div>
            """
        else:
            mini_bar_html = ""

        gallery_html += f"""
        <div class="protein-card">
          <div class="props">
            <h3 style="margin-top:0;">{short_name}</h3>
            <div style="margin-bottom:8px;">
              {size_badge} {stability_badge} {charge_badge}
            </div>
            <table style="font-size:0.9em;">
              <tr><td><b>Length</b></td><td>{props['length']} residues</td></tr>
              <tr><td><b>Molecular Weight</b></td><td>{props['molecular_weight']:,.0f} Da ({mw_kda:.1f} kDa)</td></tr>
              <tr><td><b>Isoelectric Point</b></td><td>{props['isoelectric_point']:.2f}</td></tr>
              <tr><td><b>Instability Index</b></td><td>{props['instability_index']:.1f}</td></tr>
              <tr><td><b>GRAVY</b></td><td>{props['gravy']:.3f}</td></tr>
              <tr><td><b>Aromaticity</b></td><td>{props['aromaticity']:.4f}</td></tr>
              <tr><td><b>Secondary Structure</b></td>
                <td>H:{props['helix_fraction']:.0%} T:{props['turn_fraction']:.0%} S:{props['sheet_fraction']:.0%}</td></tr>
            </table>
          </div>
          <div class="viz">
            {sparkline_html}
            {mini_bar_html}
          </div>
        </div>
        """

    # --- Classification section ---
    classification_html = "<h2>Protein Classification</h2>"

    # By size
    small = [n for n in names if properties[n]["molecular_weight"] < 15000]
    medium = [n for n in names if 15000 <= properties[n]["molecular_weight"] < 50000]
    large = [n for n in names if properties[n]["molecular_weight"] >= 50000]

    def _name_list(protein_names):
        if not protein_names:
            return "<em>None</em>"
        badges = []
        for pn in protein_names:
            short = pn.split("(")[0].strip()
            mw_kda = properties[pn]["molecular_weight"] / 1000
            badges.append(f'<span class="badge badge-success">{short} ({mw_kda:.1f} kDa)</span>')
        return " ".join(badges)

    classification_html += f"""
    <h3>By Size</h3>
    <div class="card">
      <p><b>Small (&lt;15 kDa):</b> {_name_list(small)}</p>
      <p><b>Medium (15-50 kDa):</b> {_name_list(medium)}</p>
      <p><b>Large (&gt;50 kDa):</b> {_name_list(large)}</p>
    </div>
    """

    # By stability
    stable = [n for n in names if properties[n]["instability_index"] < 40]
    unstable = [n for n in names if properties[n]["instability_index"] >= 40]

    def _stability_list(protein_names, badge_class):
        if not protein_names:
            return "<em>None</em>"
        badges = []
        for pn in protein_names:
            short = pn.split("(")[0].strip()
            ii = properties[pn]["instability_index"]
            badges.append(f'<span class="badge {badge_class}">{short} (II={ii:.0f})</span>')
        return " ".join(badges)

    classification_html += f"""
    <h3>By Stability</h3>
    <div class="card">
      <p><b>Stable (II &lt; 40):</b> {_stability_list(stable, "badge-success")}</p>
      <p><b>Unstable (II &ge; 40):</b> {_stability_list(unstable, "badge-warning")}</p>
    </div>
    """

    # By charge at pH 7
    acidic = [n for n in names if properties[n]["isoelectric_point"] < 7]
    basic = [n for n in names if properties[n]["isoelectric_point"] >= 7]

    def _charge_list(protein_names, badge_class):
        if not protein_names:
            return "<em>None</em>"
        badges = []
        for pn in protein_names:
            short = pn.split("(")[0].strip()
            pi = properties[pn]["isoelectric_point"]
            badges.append(f'<span class="badge {badge_class}">{short} (pI={pi:.1f})</span>')
        return " ".join(badges)

    classification_html += f"""
    <h3>By Charge at pH 7</h3>
    <div class="card">
      <p><b>Acidic (pI &lt; 7, net negative charge):</b> {_charge_list(acidic, "badge-danger")}</p>
      <p><b>Basic (pI &ge; 7, net positive charge):</b> {_charge_list(basic, "badge-info")}</p>
    </div>
    <div class="note">
      Proteins with pI below physiological pH (~7.4) carry a net negative charge, while those
      with pI above 7.4 carry a net positive charge. This influences protein-protein interactions,
      solubility, and behavior in electrophoresis.
    </div>
    """

    full_html = overview_html + gallery_html + classification_html

    await flyte.report.replace.aio(_wrap_report(full_html), do_flush=True)

    summary = {
        "total_proteins": len(properties),
        "total_residues": total_residues,
        "avg_molecular_weight": round(avg_mw, 1),
        "stable_count": stable_count,
        "avg_pairwise_similarity": round(avg_sim, 4),
        "classification": {
            "by_size": {"small": len(small), "medium": len(medium), "large": len(large)},
            "by_stability": {"stable": len(stable), "unstable": len(unstable)},
            "by_charge": {"acidic": len(acidic), "basic": len(basic)},
        },
    }

    return json.dumps(summary)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    sequences_json: str = "",
) -> tuple[str, str]:
    """
    End-to-end protein sequence analysis pipeline.

    Returns (properties JSON, similarity JSON).

    1. Load and validate protein sequences
    2. Analyze biophysical properties (MW, pI, stability, etc.)
    3. Compute pairwise sequence similarity
    4. Generate comprehensive visual summary report
    """
    log.info("Starting protein sequence analysis pipeline...")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Load Sequences", "Analyze Properties", "Compute Similarity", "Generate Summary"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#059669;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#059669;">&#9679;</span>'
            else:
                icon = '<span style="color:#adb5bd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Protein Sequence Analysis Pipeline</h2>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Stage 1: Load sequences
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Loading and validating protein sequences...")),
        do_flush=True,
    )
    seq_dir = await load_sequences(sequences_json=sequences_json)

    # Stage 2: Analyze properties
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Computing biophysical properties...")),
        do_flush=True,
    )
    properties_json_result = await analyze_properties(seq_dir=seq_dir)

    # Stage 3: Compute similarity
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Computing pairwise sequence similarity...")),
        do_flush=True,
    )
    similarity_json_result = await compute_similarity(seq_dir=seq_dir)

    # Stage 4: Generate summary
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating comprehensive summary report...")),
        do_flush=True,
    )
    summary_json = await generate_summary(
        properties_json=properties_json_result,
        similarity_json=similarity_json_result,
        seq_dir=seq_dir,
    )

    # Final pipeline report
    summary = json.loads(summary_json)
    classification = summary.get("classification", {})

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{summary['total_proteins']}</div><div class="label">Proteins Analyzed</div></div>
      <div class="stat"><div class="value">{summary['total_residues']:,}</div><div class="label">Total Residues</div></div>
      <div class="stat"><div class="value">{summary['avg_molecular_weight']:,.0f} Da</div><div class="label">Avg MW</div></div>
      <div class="stat"><div class="value">{summary['stable_count']}/{summary['total_proteins']}</div><div class="label">Stable Proteins</div></div>
      <div class="stat"><div class="value">{summary['avg_pairwise_similarity']:.1%}</div><div class="label">Avg Similarity</div></div>
    </div>
    <div class="card">
      <b>Classification:</b>
      Size: {classification['by_size']['small']} small, {classification['by_size']['medium']} medium, {classification['by_size']['large']} large |
      Stability: {classification['by_stability']['stable']} stable, {classification['by_stability']['unstable']} unstable |
      Charge: {classification['by_charge']['acidic']} acidic, {classification['by_charge']['basic']} basic
    </div>
    <div class="note">
      All 4 pipeline stages completed successfully. View individual task reports for detailed
      visualizations including molecular weight comparisons, secondary structure analysis,
      amino acid composition heatmaps, similarity matrices, hydrophobicity profiles, and
      per-protein property cards.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info("Pipeline complete.")
    return properties_json_result, similarity_json_result
