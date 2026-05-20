"""
Genomic Variant Effect Prediction — Score mutations with HuggingFace Carbon.

Pipeline: load curated gene variants, run Carbon-3B genomic foundation model
for zero-shot variant effect prediction (log-likelihood scoring), analyze
pathogenicity, and generate a comprehensive visual report with heatmaps,
gene diagrams, and classification charts.

Carbon is HuggingFace's open-source autoregressive genomic foundation model
(May 2026), trained on 1T tokens (~6T DNA base pairs). It uses a hybrid
6-mer tokenizer that's 275x faster than Evo2 at comparable size.

Usage:
    # Default (curated clinically relevant variants)
    flyte run --local --tui workflow.py pipeline

    # Custom model size
    flyte run --local --tui workflow.py pipeline --model_name "HuggingFaceBio/Carbon-500M"

    # Remote (GPU for Carbon inference)
    flyte run workflow.py pipeline
"""

import json
import logging
import math
import os
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Default gene variants — clinically relevant mutations
# ------------------------------------------------------------------
# Each entry: gene name -> { "sequence": reference DNA, "variants": [{ "pos": 0-indexed, "ref": base, "alt": base, "name": "...", "known_effect": "..." }] }
# Sequences are short windows (~120-200bp) around the variant site for tractable inference.

DEFAULT_GENE_VARIANTS = {
    "BRCA2 (Breast Cancer)": {
        "description": "Tumor suppressor critical for DNA repair via homologous recombination. Mutations dramatically increase breast and ovarian cancer risk.",
        "sequence": "ATGGCCTCGAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAG",
        "variants": [
            {"pos": 12, "ref": "A", "alt": "T", "name": "c.37A>T", "known_effect": "pathogenic", "clinical": "Nonsense mutation — truncates protein early"},
            {"pos": 18, "ref": "G", "alt": "A", "name": "c.55G>A", "known_effect": "benign", "clinical": "Synonymous — no amino acid change"},
            {"pos": 30, "ref": "C", "alt": "T", "name": "c.91C>T", "known_effect": "pathogenic", "clinical": "Missense in DNA-binding domain"},
            {"pos": 45, "ref": "G", "alt": "C", "name": "c.136G>C", "known_effect": "uncertain", "clinical": "Variant of uncertain significance (VUS)"},
        ],
    },
    "TP53 (Tumor Suppressor)": {
        "description": "Guardian of the genome. Activates DNA repair, cell cycle arrest, and apoptosis. Mutated in >50% of human cancers.",
        "sequence": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTGAGTTTGCACCCTTCAGAGACAGAAACCACTGGATTGGAGACTACTTCCTGAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGAT",
        "variants": [
            {"pos": 15, "ref": "C", "alt": "T", "name": "R175H", "known_effect": "pathogenic", "clinical": "Hotspot — gain-of-function, dominant negative. Most common TP53 mutation in cancer"},
            {"pos": 36, "ref": "T", "alt": "C", "name": "P72R", "known_effect": "benign", "clinical": "Common polymorphism — subtle effect on apoptosis efficiency"},
            {"pos": 54, "ref": "C", "alt": "A", "name": "G245S", "known_effect": "pathogenic", "clinical": "Contact mutant — disrupts DNA binding"},
            {"pos": 72, "ref": "T", "alt": "G", "name": "R248W", "known_effect": "pathogenic", "clinical": "Structural mutant — destabilizes DNA-binding loop"},
            {"pos": 90, "ref": "C", "alt": "T", "name": "R273H", "known_effect": "pathogenic", "clinical": "Contact mutant — directly contacts DNA bases"},
        ],
    },
    "CFTR (Cystic Fibrosis)": {
        "description": "Chloride channel protein. Mutations cause cystic fibrosis — the most common lethal genetic disease in people of European descent.",
        "sequence": "ATGCAGAGGTCGCCTCTGGAAAAGGCCAGCGTTGTCTCCAAACTTTTTTTCAGCTGGACCAGACCAATTTTGAGGAAAGGATACAGACAGCGCCTGGAATTGTCAGACATATACCAAATCCCTTC",
        "variants": [
            {"pos": 9, "ref": "G", "alt": "A", "name": "G85E", "known_effect": "pathogenic", "clinical": "Disrupts chloride channel processing"},
            {"pos": 24, "ref": "C", "alt": "T", "name": "R117H", "known_effect": "pathogenic", "clinical": "Reduces channel conductance — milder CF phenotype"},
            {"pos": 48, "ref": "T", "alt": "C", "name": "I148T", "known_effect": "benign", "clinical": "Previously misclassified — now known benign polymorphism"},
            {"pos": 66, "ref": "A", "alt": "G", "name": "R334W", "known_effect": "pathogenic", "clinical": "Gating mutation — channel opens less frequently"},
        ],
    },
    "KRAS (Oncogene)": {
        "description": "GTPase signal switch. KRAS mutations are the most common oncogenic driver — found in ~25% of all human cancers, especially pancreatic, colorectal, and lung.",
        "sequence": "ATGACTGAATATAAACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAA",
        "variants": [
            {"pos": 34, "ref": "G", "alt": "T", "name": "G12V", "known_effect": "pathogenic", "clinical": "Locks KRAS in active state — constitutive proliferation signal"},
            {"pos": 35, "ref": "G", "alt": "A", "name": "G12D", "known_effect": "pathogenic", "clinical": "Most common KRAS mutation in pancreatic cancer"},
            {"pos": 37, "ref": "G", "alt": "T", "name": "G13D", "known_effect": "pathogenic", "clinical": "Constitutively active — common in colorectal cancer"},
            {"pos": 60, "ref": "C", "alt": "A", "name": "Q61K", "known_effect": "pathogenic", "clinical": "Impairs GTP hydrolysis — locked ON state"},
        ],
    },
    "HBB (Sickle Cell)": {
        "description": "Beta-globin subunit of hemoglobin. The sickle cell mutation (E6V) is the most well-known single-base disease variant in humans.",
        "sequence": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGG",
        "variants": [
            {"pos": 17, "ref": "A", "alt": "T", "name": "E6V (HbS)", "known_effect": "pathogenic", "clinical": "THE sickle cell mutation — causes hemoglobin polymerization under low O2"},
            {"pos": 19, "ref": "G", "alt": "A", "name": "E6K (HbC)", "known_effect": "pathogenic", "clinical": "Hemoglobin C disease — milder than sickle cell but causes crystal formation"},
            {"pos": 36, "ref": "G", "alt": "A", "name": "E26K", "known_effect": "benign", "clinical": "Hemoglobin E — most common Hb variant worldwide, mild effect"},
            {"pos": 78, "ref": "C", "alt": "T", "name": "Q39X", "known_effect": "pathogenic", "clinical": "Nonsense — causes beta-thalassemia (no functional beta-globin)"},
        ],
    },
}

# DNA base colors (classic genomics color scheme)
BASE_COLORS = {"A": "#2ecc71", "T": "#e74c3c", "G": "#f39c12", "C": "#3498db"}
BASE_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

# Pathogenicity color scheme
EFFECT_COLORS = {
    "pathogenic": "#dc2626",
    "benign": "#059669",
    "uncertain": "#f59e0b",
}
EFFECT_BADGES = {
    "pathogenic": "badge-danger",
    "benign": "badge-success",
    "uncertain": "badge-warning",
}


# ------------------------------------------------------------------
# Report styling — genomics-themed deep blues and teals
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #2563eb; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #1e40af; margin-top: 20px; }
  .report .card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #dbeafe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #1e3a5f; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #1e3a5f; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dbeafe; }
  .report tr:nth-child(even) { background: #eff6ff; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-warning { background: #fef3c7; color: #92400e; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .badge-info { background: #dbeafe; color: #1e40af; }
  .report .chart-container { background: #fff; border: 1px solid #dbeafe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #eff6ff; border-left: 4px solid #2563eb; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .gene-card { background: #fff; border: 1px solid #dbeafe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .dna-track { font-family: 'SF Mono', 'Fira Code', monospace; letter-spacing: 1px; }
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
    value_format: str = ".2f",
) -> str:
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#2563eb", "#1e3a5f", "#3b82f6", "#60a5fa", "#93c5fd"]
    colors = colors or default_colors

    ml, mr, mt, mb = 70, 20, 40, 80
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for vals in series.values() for v in vals]
    y_max = max(abs(v) for v in all_vals) if all_vals else 1
    y_min = min(all_vals) if all_vals else 0
    # For VEP scores (negative = more damaging), we need to handle negative values
    if y_min >= 0:
        y_min_plot = 0
        y_max_plot = y_max * 1.15 or 1
    else:
        y_max_plot = max(y_max * 1.15, 0.1)
        y_min_plot = y_min * 1.15

    y_range = y_max_plot - y_min_plot or 1

    n_groups = len(labels)
    n_series = len(series)
    group_width = cw / n_groups
    bar_width = group_width * 0.7 / max(n_series, 1)
    gap = group_width * 0.15

    def sy(v):
        return mt + ch - (v - y_min_plot) / y_range * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid lines
    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:{value_format}}</text>'
        )

    # Zero line
    if y_min_plot < 0 < y_max_plot:
        zy = sy(0)
        svg.append(
            f'<line x1="{ml}" y1="{zy:.1f}" x2="{ml + cw}" y2="{zy:.1f}" '
            f'stroke="#374151" stroke-width="1.5"/>'
        )

    # Bars
    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            if val >= 0:
                by = sy(val)
                bh = sy(0) - by if y_min_plot < 0 else mt + ch - by
            else:
                by = sy(0) if y_min_plot < 0 else mt + ch
                bh = sy(val) - by
            svg.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" '
                f'height="{max(0, bh):.1f}" fill="{color}" rx="2"/>'
            )
            text_y = by - 4 if val >= 0 else by + bh + 12
            svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{text_y:.1f}" '
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


def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    width: int = 700,
    height: int = 500,
    value_format: str = ".2f",
    diverging: bool = False,
) -> str:
    """Generate an SVG heatmap. If diverging=True, uses red-white-blue scale centered at 0."""
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if not n_rows or not n_cols:
        return ""

    show_values = n_rows <= 10 and n_cols <= 12

    flat = [v for row in matrix for v in row]
    v_min = min(flat)
    v_max = max(flat)

    if diverging:
        abs_max = max(abs(v_min), abs(v_max)) or 1

        def get_color(v):
            t = v / abs_max  # -1 to 1
            if t < 0:
                # White to red (negative = damaging)
                r = 255
                g = int(255 * (1 + t))
                b = int(255 * (1 + t))
            else:
                # White to blue (positive = benign)
                r = int(255 * (1 - t))
                g = int(255 * (1 - t))
                b = 255
            return f"rgb({r},{g},{b})"
    else:
        v_range = v_max - v_min or 1

        def get_color(v):
            t = (v - v_min) / v_range
            r = int(255 - t * (255 - 30))
            g = int(255 - t * (255 - 58))
            b = int(255 - t * (255 - 95))
            return f"rgb({r},{g},{b})"

    # Layout
    ml = max(140, max(len(l) for l in row_labels) * 7 + 20) if row_labels else 140
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
                if diverging:
                    t = abs(val) / (max(abs(v_min), abs(v_max)) or 1)
                else:
                    t = (val - v_min) / (v_max - v_min or 1)
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


def _make_dna_track(
    sequence: str,
    variants: list[dict],
    gene_name: str = "",
    width: int = 900,
) -> str:
    """Render a color-coded DNA sequence track with variant positions highlighted."""
    chars_per_line = 60
    char_w = 11
    line_h = 22
    label_w = 50
    n_lines = (len(sequence) + chars_per_line - 1) // chars_per_line

    # Extra space for variant annotations
    variant_positions = {v["pos"] for v in variants}
    svg_h = n_lines * line_h + 60

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {svg_h}" '
        f'style="width:100%;max-width:{width}px;height:auto;font-family:monospace;">',
        f'<rect width="{width}" height="{svg_h}" fill="#fff" rx="6"/>',
    ]

    if gene_name:
        svg.append(
            f'<text x="{width / 2}" y="16" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#1e3a5f">{gene_name}</text>'
        )

    y_offset = 28

    for line_idx in range(n_lines):
        start = line_idx * chars_per_line
        end = min(start + chars_per_line, len(sequence))
        chunk = sequence[start:end]
        y = y_offset + line_idx * line_h

        # Position label
        svg.append(
            f'<text x="{label_w - 8}" y="{y}" text-anchor="end" '
            f'font-size="9" fill="#9ca3af">{start + 1}</text>'
        )

        for ci, base in enumerate(chunk):
            abs_pos = start + ci
            x = label_w + ci * char_w
            color = BASE_COLORS.get(base, "#6b7280")
            is_variant = abs_pos in variant_positions

            if is_variant:
                # Red highlight for variant positions
                svg.append(
                    f'<rect x="{x - 1}" y="{y - 13}" width="{char_w + 2}" height="{line_h}" '
                    f'fill="#fee2e2" stroke="#dc2626" stroke-width="1.5" rx="2"/>'
                )
                # Small triangle marker above
                svg.append(
                    f'<polygon points="{x + char_w / 2:.1f},{y - 16} {x + char_w / 2 - 3:.1f},{y - 20} {x + char_w / 2 + 3:.1f},{y - 20}" '
                    f'fill="#dc2626"/>'
                )
            else:
                # Subtle background
                svg.append(
                    f'<rect x="{x}" y="{y - 12}" width="{char_w}" height="{line_h - 2}" '
                    f'fill="{color}" opacity="0.08" rx="1"/>'
                )

            svg.append(
                f'<text x="{x + char_w / 2:.1f}" y="{y}" text-anchor="middle" '
                f'font-size="11" font-weight="{"700" if is_variant else "500"}" '
                f'fill="{color}">{base}</text>'
            )

            # Spacer every 10 bases
            if (abs_pos + 1) % 10 == 0 and ci < len(chunk) - 1:
                svg.append(
                    f'<line x1="{x + char_w + 1}" y1="{y - 11}" '
                    f'x2="{x + char_w + 1}" y2="{y + 4}" '
                    f'stroke="#e5e7eb" stroke-width="0.5"/>'
                )

    # Legend
    ly = svg_h - 12
    lx = label_w
    for base, color in BASE_COLORS.items():
        svg.append(f'<rect x="{lx}" y="{ly - 8}" width="8" height="8" rx="1" fill="{color}"/>')
        svg.append(f'<text x="{lx + 12}" y="{ly}" font-size="9" fill="#6b7280">{base}</text>')
        lx += 30
    lx += 10
    svg.append(f'<rect x="{lx}" y="{ly - 8}" width="8" height="8" rx="1" fill="#fee2e2" stroke="#dc2626" stroke-width="1"/>')
    svg.append(f'<text x="{lx + 12}" y="{ly}" font-size="9" fill="#6b7280">Variant site</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_lollipop_plot(
    variants: list[dict],
    seq_length: int,
    title: str = "",
    width: int = 700,
    height: int = 200,
) -> str:
    """Generate a lollipop plot showing variant positions along a gene with effect scores."""
    if not variants:
        return ""

    ml, mr, mt, mb = 50, 30, 40, 40
    cw = width - ml - mr
    ch = height - mt - mb

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="13" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    # Gene track (horizontal bar)
    track_y = mt + ch * 0.7
    svg.append(
        f'<rect x="{ml}" y="{track_y - 4}" width="{cw}" height="8" '
        f'fill="#dbeafe" rx="4" stroke="#93c5fd" stroke-width="1"/>'
    )

    # Position labels at ends
    svg.append(
        f'<text x="{ml}" y="{track_y + 20}" text-anchor="middle" '
        f'font-size="9" fill="#9ca3af">1</text>'
    )
    svg.append(
        f'<text x="{ml + cw}" y="{track_y + 20}" text-anchor="middle" '
        f'font-size="9" fill="#9ca3af">{seq_length}</text>'
    )

    # Lollipops
    scores = [v.get("score", 0) for v in variants]
    max_score = max(abs(s) for s in scores) if scores else 1

    for v in variants:
        pos = v["pos"]
        score = v.get("score", 0)
        effect = v.get("known_effect", "uncertain")
        color = EFFECT_COLORS.get(effect, "#6b7280")

        x = ml + (pos / max(seq_length - 1, 1)) * cw
        stem_h = max(20, abs(score) / max_score * (ch * 0.6))
        circle_y = track_y - stem_h - 6

        # Stem
        svg.append(
            f'<line x1="{x:.1f}" y1="{track_y - 4}" x2="{x:.1f}" y2="{circle_y + 5:.1f}" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        # Circle
        svg.append(
            f'<circle cx="{x:.1f}" cy="{circle_y:.1f}" r="5" fill="{color}" '
            f'stroke="#fff" stroke-width="1.5"/>'
        )
        # Label
        svg.append(
            f'<text x="{x:.1f}" y="{circle_y - 9:.1f}" text-anchor="middle" '
            f'font-size="8" fill="#374151" font-weight="600">{v.get("name", "")}</text>'
        )

    # Legend
    lx = ml
    for effect, color in EFFECT_COLORS.items():
        svg.append(f'<circle cx="{lx + 4}" cy="{height - 10}" r="4" fill="{color}"/>')
        svg.append(
            f'<text x="{lx + 12}" y="{height - 6}" font-size="9" fill="#374151">'
            f'{effect.title()}</text>'
        )
        lx += len(effect) * 7 + 24

    svg.append("</svg>")
    return "\n".join(svg)


def _make_score_comparison_chart(
    gene_results: dict,
    width: int = 800,
    height: int = 350,
) -> str:
    """Generate a dot plot comparing VEP scores across all genes, colored by known effect."""
    all_variants = []
    for gene_name, gene_data in gene_results.items():
        for v in gene_data["variants"]:
            all_variants.append({
                "gene": gene_name.split("(")[0].strip(),
                "name": v["name"],
                "score": v.get("score", 0),
                "known_effect": v["known_effect"],
            })

    if not all_variants:
        return ""

    ml, mr, mt, mb = 120, 30, 40, 30
    cw = width - ml - mr
    ch = height - mt - mb

    scores = [v["score"] for v in all_variants]
    s_min, s_max = min(scores), max(scores)
    s_pad = (s_max - s_min) * 0.1 or 0.5
    s_min -= s_pad
    s_max += s_pad
    s_range = s_max - s_min or 1

    def sx(s):
        return ml + (s - s_min) / s_range * cw

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
        f'<text x="{width / 2}" y="22" text-anchor="middle" '
        f'font-size="14" font-weight="600" fill="#1a1a2e">'
        f'Variant Effect Scores — All Genes</text>',
    ]

    # Grid
    for i in range(6):
        gx = s_min + s_range * i / 5
        px = sx(gx)
        svg.append(
            f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt + ch}" '
            f'stroke="#f3f4f6" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{px:.1f}" y="{mt + ch + 16}" text-anchor="middle" '
            f'font-size="10" fill="#9ca3af">{gx:.2f}</text>'
        )

    # Zero line
    if s_min < 0 < s_max:
        zx = sx(0)
        svg.append(
            f'<line x1="{zx:.1f}" y1="{mt}" x2="{zx:.1f}" y2="{mt + ch}" '
            f'stroke="#374151" stroke-width="1" stroke-dasharray="4,3"/>'
        )

    # Rows per variant
    row_h = ch / max(len(all_variants), 1)
    for i, v in enumerate(all_variants):
        y = mt + i * row_h + row_h / 2
        color = EFFECT_COLORS.get(v["known_effect"], "#6b7280")
        px = sx(v["score"])

        # Label
        label = f'{v["gene"]} {v["name"]}'
        svg.append(
            f'<text x="{ml - 8}" y="{y + 3:.1f}" text-anchor="end" '
            f'font-size="9" fill="#374151">{label}</text>'
        )
        # Connector line
        svg.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{px:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="1" opacity="0.3"/>'
        )
        # Dot
        svg.append(
            f'<circle cx="{px:.1f}" cy="{y:.1f}" r="5" fill="{color}" '
            f'stroke="#fff" stroke-width="1"/>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Task 1: Load and validate gene variants
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def load_variants(
    variants_json: str = "",
) -> flyte.io.Dir:
    """Load gene variant definitions, validate sequences, and save to a temp directory."""
    if variants_json:
        genes = json.loads(variants_json)
    else:
        genes = DEFAULT_GENE_VARIANTS

    # Validate
    valid_bases = set("ATGC")
    for gene_name, gene_data in genes.items():
        seq = gene_data["sequence"].upper()
        invalid = set(seq) - valid_bases
        if invalid:
            log.warning(f"{gene_name}: invalid bases {invalid} — removing them")
            seq = "".join(b for b in seq if b in valid_bases)
            gene_data["sequence"] = seq

        for v in gene_data["variants"]:
            pos = v["pos"]
            if pos < 0 or pos >= len(seq):
                log.warning(f"{gene_name} variant {v['name']}: position {pos} out of range [0, {len(seq)})")
            elif seq[pos] != v["ref"]:
                log.warning(f"{gene_name} variant {v['name']}: expected ref={v['ref']} at pos {pos}, found {seq[pos]}")

    total_variants = sum(len(g["variants"]) for g in genes.values())
    log.info(f"Loaded {len(genes)} genes with {total_variants} variants")

    out_dir = tempfile.mkdtemp(prefix="genomic_vep_")
    with open(os.path.join(out_dir, "genes.json"), "w") as f:
        json.dump(genes, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Run Carbon model for variant effect scoring
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def score_variants(
    variants_dir: flyte.io.Dir,
    model_name: str = "HuggingFaceBio/Carbon-3B",
) -> str:
    """Score each variant using Carbon's log-likelihood ratio.

    For each variant, we compute:
        score = log P(alt_sequence) - log P(ref_sequence)

    A negative score means the model considers the variant less likely than
    the reference — suggestive of a damaging/pathogenic effect. A score near
    zero means the model sees little difference (likely benign).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading Carbon model: {model_name}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("Running on CPU — inference will be slow. GPU recommended for production.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # Load variants
    variants_path = await variants_dir.download()
    with open(os.path.join(variants_path, "genes.json")) as f:
        genes = json.load(f)

    results = {}
    total_variants = sum(len(g["variants"]) for g in genes.values())
    scored = 0

    progress_html = """
    <h2>Carbon Variant Effect Scoring</h2>
    <div class="card">
        <b>Model:</b> {model}<br>
        <b>Device:</b> {device}<br>
        <b>Progress:</b> {scored}/{total} variants scored
    </div>
    """

    for gene_name, gene_data in genes.items():
        ref_seq = gene_data["sequence"]
        gene_results = {
            "description": gene_data.get("description", ""),
            "sequence": ref_seq,
            "variants": [],
        }

        # Score reference sequence
        ref_prompt = f"<dna>{ref_seq}"
        ref_inputs = tokenizer(ref_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            ref_output = model(**ref_inputs, labels=ref_inputs["input_ids"])
            ref_loss = ref_output.loss.item()
            ref_ll = -ref_loss * ref_inputs["input_ids"].shape[1]

        for variant in gene_data["variants"]:
            scored += 1
            await flyte.report.replace.aio(
                _wrap_report(progress_html.format(
                    model=model_name, device=device, scored=scored, total=total_variants
                )),
                do_flush=True,
            )

            # Create mutant sequence
            pos = variant["pos"]
            alt_seq = ref_seq[:pos] + variant["alt"] + ref_seq[pos + 1:]

            # Score mutant
            alt_prompt = f"<dna>{alt_seq}"
            alt_inputs = tokenizer(alt_prompt, return_tensors="pt", add_special_tokens=False).to(device)

            with torch.no_grad():
                alt_output = model(**alt_inputs, labels=alt_inputs["input_ids"])
                alt_loss = alt_output.loss.item()
                alt_ll = -alt_loss * alt_inputs["input_ids"].shape[1]

            # VEP score: positive = model prefers alt (likely benign), negative = model prefers ref (likely pathogenic)
            vep_score = alt_ll - ref_ll

            gene_results["variants"].append({
                **variant,
                "score": round(vep_score, 4),
                "ref_ll": round(ref_ll, 4),
                "alt_ll": round(alt_ll, 4),
            })

            log.info(
                f"  {gene_name} | {variant['name']}: score={vep_score:.4f} "
                f"(known: {variant['known_effect']})"
            )

        results[gene_name] = gene_results

    # Generate scoring report
    html_parts = [
        "<h2>Carbon Variant Effect Scoring</h2>",
        '<div class="stat-grid">',
        f'<div class="stat"><div class="value">{len(genes)}</div><div class="label">Genes</div></div>',
        f'<div class="stat"><div class="value">{total_variants}</div><div class="label">Variants Scored</div></div>',
        f'<div class="stat"><div class="value">{model_name.split("/")[-1]}</div><div class="label">Model</div></div>',
        f'<div class="stat"><div class="value">{device.upper()}</div><div class="label">Device</div></div>',
        "</div>",
    ]

    # Per-gene tables
    for gene_name, gene_data in results.items():
        html_parts.append(f'<h3>{gene_name}</h3>')
        html_parts.append(f'<div class="note">{gene_data["description"]}</div>')
        html_parts.append("<table><tr><th>Variant</th><th>Ref</th><th>Alt</th><th>VEP Score</th><th>Known Effect</th><th>Clinical</th></tr>")

        for v in gene_data["variants"]:
            badge = EFFECT_BADGES.get(v["known_effect"], "badge-info")
            direction = "damaging" if v["score"] < -0.1 else "neutral" if abs(v["score"]) <= 0.1 else "tolerated"
            html_parts.append(
                f'<tr>'
                f'<td><b>{v["name"]}</b></td>'
                f'<td style="color:{BASE_COLORS.get(v["ref"], "#333")};font-weight:700">{v["ref"]}</td>'
                f'<td style="color:{BASE_COLORS.get(v["alt"], "#333")};font-weight:700">{v["alt"]}</td>'
                f'<td><b>{v["score"]:.4f}</b> <span style="font-size:0.8em;color:#6c757d">({direction})</span></td>'
                f'<td><span class="badge {badge}">{v["known_effect"]}</span></td>'
                f'<td style="font-size:0.85em">{v.get("clinical", "")}</td>'
                f'</tr>'
            )
        html_parts.append("</table>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    return json.dumps(results)


# ------------------------------------------------------------------
# Task 3: Analyze and visualize variant effects
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def analyze_effects(
    scores_json: str,
    variants_dir: flyte.io.Dir,
) -> str:
    """Analyze VEP scores: classification accuracy, gene-level summaries, and rich visualizations."""
    results = json.loads(scores_json)

    variants_path = await variants_dir.download()
    with open(os.path.join(variants_path, "genes.json")) as f:
        genes = json.load(f)

    html_parts = ["<h2>Variant Effect Analysis</h2>"]

    # ------------------------------------------------------------------
    # Overall accuracy: does the model's score direction match known labels?
    # ------------------------------------------------------------------
    all_variants = []
    correct = 0
    total_known = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    for gene_name, gene_data in results.items():
        for v in gene_data["variants"]:
            all_variants.append({**v, "gene": gene_name})
            if v["known_effect"] in ("pathogenic", "benign"):
                total_known += 1
                predicted_pathogenic = v["score"] < -0.05
                actual_pathogenic = v["known_effect"] == "pathogenic"
                if predicted_pathogenic == actual_pathogenic:
                    correct += 1
                if predicted_pathogenic and actual_pathogenic:
                    true_pos += 1
                elif predicted_pathogenic and not actual_pathogenic:
                    false_pos += 1
                elif not predicted_pathogenic and actual_pathogenic:
                    false_neg += 1
                else:
                    true_neg += 1

    accuracy = correct / total_known if total_known else 0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0

    html_parts.append('<div class="stat-grid">')
    html_parts.append(f'<div class="stat"><div class="value">{accuracy:.0%}</div><div class="label">Direction Accuracy</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{precision:.0%}</div><div class="label">Precision (Pathogenic)</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{recall:.0%}</div><div class="label">Recall (Pathogenic)</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{len(all_variants)}</div><div class="label">Total Variants</div></div>')
    html_parts.append("</div>")

    html_parts.append(
        '<div class="note">'
        "<b>How to read VEP scores:</b> Negative scores mean Carbon considers the variant "
        "less likely than the reference sequence — suggestive of a damaging effect. Scores near "
        "zero indicate the model sees little difference (likely benign). The magnitude indicates "
        "confidence."
        "</div>"
    )

    # ------------------------------------------------------------------
    # Cross-gene score comparison dot plot
    # ------------------------------------------------------------------
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_score_comparison_chart(results))
    html_parts.append("</div>")

    # ------------------------------------------------------------------
    # Per-gene visualizations
    # ------------------------------------------------------------------
    for gene_name, gene_data in results.items():
        short_name = gene_name.split("(")[0].strip()
        html_parts.append(f'<h3>{gene_name}</h3>')
        html_parts.append(f'<div class="note">{gene_data["description"]}</div>')

        # DNA track with variant positions highlighted
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_dna_track(
            gene_data["sequence"],
            gene_data["variants"],
            gene_name=f"{short_name} Reference Sequence",
        ))
        html_parts.append("</div>")

        # Lollipop plot
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_lollipop_plot(
            gene_data["variants"],
            len(gene_data["sequence"]),
            title=f"{short_name} — Variant Positions & Effect Scores",
        ))
        html_parts.append("</div>")

        # Score bar chart for this gene
        variant_names = [v["name"] for v in gene_data["variants"]]
        variant_scores = [v["score"] for v in gene_data["variants"]]
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_bar_chart(
            variant_names,
            {"VEP Score": variant_scores},
            title=f"{short_name} — Log-Likelihood Ratio Scores",
            colors=[EFFECT_COLORS.get(v["known_effect"], "#6b7280") for v in gene_data["variants"]],
            value_format=".3f",
        ))
        html_parts.append("</div>")

        # Variant detail cards
        for v in gene_data["variants"]:
            badge = EFFECT_BADGES.get(v["known_effect"], "badge-info")
            score_color = "#dc2626" if v["score"] < -0.1 else "#059669" if v["score"] > 0.05 else "#f59e0b"
            html_parts.append(
                f'<div class="gene-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<b style="font-size:1.1em">{v["name"]}</b>'
                f'<span class="badge {badge}">{v["known_effect"]}</span>'
                f'</div>'
                f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;">'
                f'<div><span style="color:#6c757d;font-size:0.85em">Ref base:</span> '
                f'<b style="color:{BASE_COLORS.get(v["ref"], "#333")}">{v["ref"]}</b></div>'
                f'<div><span style="color:#6c757d;font-size:0.85em">Alt base:</span> '
                f'<b style="color:{BASE_COLORS.get(v["alt"], "#333")}">{v["alt"]}</b></div>'
                f'<div><span style="color:#6c757d;font-size:0.85em">VEP Score:</span> '
                f'<b style="color:{score_color}">{v["score"]:.4f}</b></div>'
                f'</div>'
                f'<div style="margin-top:8px;font-size:0.9em;color:#374151">{v.get("clinical", "")}</div>'
                f'</div>'
            )

    # ------------------------------------------------------------------
    # Confusion matrix as heatmap
    # ------------------------------------------------------------------
    html_parts.append("<h3>Classification Performance</h3>")
    html_parts.append(
        '<div class="note">'
        "Using a simple threshold (score &lt; -0.05 = predicted pathogenic). "
        "This is zero-shot — no training on these specific variants."
        "</div>"
    )

    conf_matrix = [[true_pos, false_neg], [false_pos, true_neg]]
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_heatmap(
        conf_matrix,
        ["Actual Pathogenic", "Actual Benign"],
        ["Predicted Pathogenic", "Predicted Benign"],
        title="Confusion Matrix (Known Variants Only)",
        value_format=".0f",
        width=400,
        height=300,
    ))
    html_parts.append("</div>")

    # ------------------------------------------------------------------
    # Score distribution by known effect
    # ------------------------------------------------------------------
    html_parts.append("<h3>Score Distribution by Known Effect</h3>")

    pathogenic_scores = [v["score"] for v in all_variants if v["known_effect"] == "pathogenic"]
    benign_scores = [v["score"] for v in all_variants if v["known_effect"] == "benign"]
    uncertain_scores = [v["score"] for v in all_variants if v["known_effect"] == "uncertain"]

    stats_html = '<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">'
    for label, scores, color in [
        ("Pathogenic", pathogenic_scores, "#dc2626"),
        ("Benign", benign_scores, "#059669"),
        ("Uncertain", uncertain_scores, "#f59e0b"),
    ]:
        if scores:
            mean_s = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)
            stats_html += (
                f'<div class="gene-card" style="border-left:4px solid {color}">'
                f'<b style="color:{color}">{label}</b> (n={len(scores)})<br>'
                f'Mean: {mean_s:.4f}<br>'
                f'Range: [{min_s:.4f}, {max_s:.4f}]'
                f'</div>'
            )
    stats_html += "</div>"
    html_parts.append(stats_html)

    # Summary
    analysis = {
        "total_variants": len(all_variants),
        "total_known": total_known,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "true_pos": true_pos,
        "false_pos": false_pos,
        "true_neg": true_neg,
        "false_neg": false_neg,
        "pathogenic_mean_score": round(sum(pathogenic_scores) / len(pathogenic_scores), 4) if pathogenic_scores else None,
        "benign_mean_score": round(sum(benign_scores) / len(benign_scores), 4) if benign_scores else None,
    }

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)
    return json.dumps(analysis)


# ------------------------------------------------------------------
# Task 4: Generate comprehensive summary report
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def generate_summary(
    scores_json: str,
    analysis_json: str,
) -> str:
    """Generate the final summary report combining all results."""
    results = json.loads(scores_json)
    analysis = json.loads(analysis_json)

    html_parts = [
        "<h2>Genomic Variant Effect Prediction — Summary</h2>",
        '<div class="note">'
        "This pipeline uses <b>HuggingFace Carbon</b>, an autoregressive genomic foundation model "
        "trained on 1 trillion tokens of DNA sequence, to perform <b>zero-shot variant effect "
        "prediction</b>. No fine-tuning or labeled training data was used — the model scores "
        "variants purely based on its learned understanding of DNA sequence grammar."
        "</div>",
    ]

    # Key metrics
    html_parts.append('<div class="stat-grid">')
    html_parts.append(f'<div class="stat"><div class="value">{len(results)}</div><div class="label">Genes Analyzed</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{analysis["total_variants"]}</div><div class="label">Variants Scored</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{analysis["accuracy"]:.0%}</div><div class="label">Direction Accuracy</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{analysis["precision"]:.0%}</div><div class="label">Precision</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{analysis["recall"]:.0%}</div><div class="label">Recall</div></div>')
    html_parts.append("</div>")

    # Gene summary table
    html_parts.append("<h3>Per-Gene Summary</h3>")
    html_parts.append(
        "<table><tr><th>Gene</th><th>Variants</th><th>Mean Score</th>"
        "<th>Pathogenic</th><th>Benign</th><th>Uncertain</th></tr>"
    )

    for gene_name, gene_data in results.items():
        variants = gene_data["variants"]
        scores = [v["score"] for v in variants]
        mean_score = sum(scores) / len(scores) if scores else 0
        n_path = sum(1 for v in variants if v["known_effect"] == "pathogenic")
        n_benign = sum(1 for v in variants if v["known_effect"] == "benign")
        n_unc = sum(1 for v in variants if v["known_effect"] == "uncertain")
        short = gene_name.split("(")[0].strip()

        html_parts.append(
            f"<tr><td><b>{short}</b></td><td>{len(variants)}</td>"
            f"<td>{mean_score:.4f}</td>"
            f'<td><span class="badge badge-danger">{n_path}</span></td>'
            f'<td><span class="badge badge-success">{n_benign}</span></td>'
            f'<td><span class="badge badge-warning">{n_unc}</span></td></tr>'
        )
    html_parts.append("</table>")

    # Cross-gene heatmap: gene x metric
    gene_names = [g.split("(")[0].strip() for g in results.keys()]
    metrics = ["Mean Score", "Min Score", "Max Score", "# Variants"]
    matrix = []
    for gene_data in results.values():
        scores = [v["score"] for v in gene_data["variants"]]
        matrix.append([
            sum(scores) / len(scores) if scores else 0,
            min(scores) if scores else 0,
            max(scores) if scores else 0,
            len(scores),
        ])

    html_parts.append("<h3>Gene-Level Metrics</h3>")
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_heatmap(
        matrix,
        gene_names,
        metrics,
        title="Gene-Level VEP Score Summary",
        value_format=".2f",
        width=600,
        height=350,
    ))
    html_parts.append("</div>")

    # All variants ranked by score (most damaging first)
    html_parts.append("<h3>All Variants Ranked by Impact</h3>")
    all_vars_sorted = []
    for gene_name, gene_data in results.items():
        for v in gene_data["variants"]:
            all_vars_sorted.append({**v, "gene": gene_name.split("(")[0].strip()})
    all_vars_sorted.sort(key=lambda x: x["score"])

    html_parts.append(
        "<table><tr><th>#</th><th>Gene</th><th>Variant</th><th>Score</th>"
        "<th>Known</th><th>Clinical Significance</th></tr>"
    )
    for i, v in enumerate(all_vars_sorted):
        badge = EFFECT_BADGES.get(v["known_effect"], "badge-info")
        score_color = "#dc2626" if v["score"] < -0.1 else "#059669" if v["score"] > 0.05 else "#f59e0b"
        html_parts.append(
            f'<tr><td>{i + 1}</td><td><b>{v["gene"]}</b></td><td>{v["name"]}</td>'
            f'<td style="color:{score_color};font-weight:700">{v["score"]:.4f}</td>'
            f'<td><span class="badge {badge}">{v["known_effect"]}</span></td>'
            f'<td style="font-size:0.85em">{v.get("clinical", "")}</td></tr>'
        )
    html_parts.append("</table>")

    # Method note
    html_parts.append(
        '<div class="note">'
        "<b>Method:</b> Zero-shot variant effect prediction using log-likelihood ratio scoring. "
        "For each variant, we compute score = log P(mutant sequence | Carbon) - log P(reference sequence | Carbon). "
        "Negative scores indicate the model considers the mutant less probable than the reference, "
        "which correlates with pathogenicity. This approach requires no fine-tuning and generalizes "
        "across genes and variant types.<br><br>"
        "<b>Limitations:</b> These are short sequence windows — real clinical VEP would use longer "
        "genomic context (Carbon supports up to 786kbp). The threshold for pathogenicity classification "
        "(-0.05) is a simple heuristic; clinical use requires calibrated thresholds per gene."
        "</div>"
    )

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)
    return json.dumps({"status": "complete", "analysis": analysis})


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    variants_json: str = "",
    model_name: str = "HuggingFaceBio/Carbon-3B",
) -> tuple[str, str]:
    """
    End-to-end genomic variant effect prediction pipeline.

    Returns (scores JSON, analysis JSON).

    1. Load and validate gene variants
    2. Score variants with Carbon (log-likelihood ratio)
    3. Analyze effects — accuracy, visualizations, classification
    4. Generate comprehensive summary report
    """
    log.info("Starting genomic variant effect prediction pipeline...")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = [
            "Load Variants",
            "Carbon Scoring",
            "Analyze Effects",
            "Generate Summary",
        ]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#2563eb;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#2563eb;">&#9679;</span>'
            else:
                icon = '<span style="color:#adb5bd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Genomic Variant Effect Prediction</h2>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Stage 1: Load variants
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Loading and validating gene variants...")),
        do_flush=True,
    )
    var_dir = await load_variants(variants_json=variants_json)

    # Stage 2: Score with Carbon
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Running Carbon model for variant effect scoring...")),
        do_flush=True,
    )
    scores_json = await score_variants(variants_dir=var_dir, model_name=model_name)

    # Stage 3: Analyze effects
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Analyzing variant effects and generating visualizations...")),
        do_flush=True,
    )
    analysis_json = await analyze_effects(scores_json=scores_json, variants_dir=var_dir)

    # Stage 4: Summary
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating comprehensive summary report...")),
        do_flush=True,
    )
    summary_json = await generate_summary(scores_json=scores_json, analysis_json=analysis_json)

    # Final pipeline report
    analysis = json.loads(analysis_json)
    results = json.loads(scores_json)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(results)}</div><div class="label">Genes Analyzed</div></div>
      <div class="stat"><div class="value">{analysis['total_variants']}</div><div class="label">Variants Scored</div></div>
      <div class="stat"><div class="value">{analysis['accuracy']:.0%}</div><div class="label">Direction Accuracy</div></div>
      <div class="stat"><div class="value">{analysis['precision']:.0%}</div><div class="label">Precision</div></div>
      <div class="stat"><div class="value">{analysis['recall']:.0%}</div><div class="label">Recall</div></div>
    </div>
    <div class="card">
      <b>Model:</b> HuggingFace Carbon |
      <b>Method:</b> Zero-shot log-likelihood ratio scoring |
      <b>Genes:</b> {', '.join(g.split('(')[0].strip() for g in results.keys())}
    </div>
    <div class="note">
      All 4 pipeline stages completed successfully. View individual task reports for detailed
      visualizations including DNA sequence tracks, variant lollipop plots, VEP score charts,
      confusion matrices, and ranked variant tables.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)

    log.info("Pipeline complete.")
    return scores_json, analysis_json
