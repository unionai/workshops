"""
DNA Sequence Generation & Analysis — Generate and analyze DNA with HuggingFace Carbon.

Pipeline: provide real gene sequence prompts, generate DNA continuations with
Carbon-3B, then analyze and compare generated vs real sequences — GC content,
codon usage, open reading frames (ORFs), and sequence composition. Rich SVG
visualizations throughout.

Carbon is HuggingFace's open-source autoregressive genomic foundation model
(May 2026), trained on 1T tokens (~6T DNA base pairs). It uses a hybrid
6-mer tokenizer that's 275x faster than Evo2 at comparable size.

Usage:
    # Default (curated gene prompts)
    flyte run --local --tui workflow.py pipeline

    # Custom model / generation length
    flyte run --local --tui workflow.py pipeline --model_name "HuggingFaceBio/Carbon-500M" --gen_length 300

    # Remote (GPU)
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
# Default gene prompts — real gene starts used as seeds for generation
# ------------------------------------------------------------------
# Each entry: name -> { "prompt": first N bases of a real gene, "full_sequence": the real continuation (for comparison), "description": ... }
# Prompts are ~60bp from the start of real coding sequences.

DEFAULT_GENE_PROMPTS = {
    "Human Insulin (INS)": {
        "description": "Encodes preproinsulin — the precursor to insulin, the master regulator of blood sugar. One of the first genes ever cloned (1978).",
        "prompt": "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGAC",
        "full_sequence": "ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCTGACCCAGCCGCAGCCTTTGTGAACCAACACCTGTGCGGCTCACACCTGGTGGAAGCTCTCTACCTAGTGTGCGGGGAACGAGGCTTCTTCTACACACCCAAGACCCGCCGGGAGGCAGAGGACCTGCAGGTGGGGCAGGTGGAGCTGGGCGGGGGCCCTGGTGCAGGCAGCCTGCAGCCCTTGGCCCTGGAGGGGTCCCTGCAGAAGCGTGGCATTGTGGAACAATGCTGTACCAGCATCTGCTCCCTCTACCAGCTGGAGAACTACTGCAAC",
    },
    "Human Hemoglobin Beta (HBB)": {
        "description": "Beta-globin subunit of hemoglobin — carries oxygen from lungs to tissues. Mutations cause sickle cell disease and beta-thalassemia.",
        "prompt": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAAC",
        "full_sequence": "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCTCACTGCAGTGCCAGCCTGGGAGCTTGTGGCACC",
    },
    "GFP (Green Fluorescent Protein)": {
        "description": "From jellyfish Aequorea victoria — revolutionized cell biology. Tag any protein with GFP to make it glow green under UV. Nobel Prize 2008.",
        "prompt": "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTG",
        "full_sequence": "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCTCTTATGGTGTTCAATGCTTTTCAAGATACCCAGATCATATGAAACGGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACT",
    },
    "E. coli LacZ (Beta-galactosidase)": {
        "description": "The classic reporter gene — cleaves lactose and turns X-gal blue. Foundation of blue-white colony screening in molecular cloning.",
        "prompt": "ATGACCATGATTACGCCAAGCTATTTAGGTGACACTATAGAATACTCAAGCTATGCATCAA",
        "full_sequence": "ATGACCATGATTACGCCAAGCTATTTAGGTGACACTATAGAATACTCAAGCTATGCATCAAGCTTGGTACCGAGCTCGGATCCACTAGTAACGGCCGCCAGTGTGCTGGAATTCGGCTTCCCGGTTTCGGCCCACGGCACGATCTCGTCGATCAGGACCTGGCCAGTTTCGACTCGGCCTTCGACGTCGCCGGAATCCCGTTTGACCCGGCCTTTGTCGGCTTTGACGATCTTGGCAGCAACCAATGGCCGTTTCTCAATGCGCACTGGCTCGCGTTTCTCGGCGCTCGG",
    },
    "SARS-CoV-2 Spike (fragment)": {
        "description": "The spike protein that lets COVID-19 enter human cells via ACE2 receptor. Target of all mRNA vaccines. Most studied viral protein in history.",
        "prompt": "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACC",
        "full_sequence": "ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTCTCTAGTCAGTGTGTTAATCTTACAACCAGAACTCAATTACCCCCTGCATACACTAATTCTTTCACACGTGGTGTTTATTACCCTGACAAAGTTTTCAGATCCTCAGTTTTACATTCAACTCAGGACTTGTTCTTACCTTTCTTTTCCAATGTTACTTGGTTCCATGCTATACATGTCTCTGGGACCAATGGTACTAAGAGGTTTGATAACCCTGTCCTACCATTTAATGATGGTGTTTATTTTGCTTCCACTGAG",
    },
    "Human TP53 (Tumor Suppressor)": {
        "description": "Guardian of the genome — activates DNA repair, cell cycle arrest, and apoptosis. Mutated in >50% of human cancers.",
        "prompt": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTGAGTTTGCACCCTTCAGAGACAGAAACCACT",
        "full_sequence": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTGAGTTTGCACCCTTCAGAGACAGAAACCACTGGATTGGAGACTACTTCCTGAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCACTGAAGACCCAGGTCCAGATGAAGCTCCCAGAATGCCAGAGGCTGCTCCCCCCGTGGCCCCTGCACCAGCAGCTCCTACACCGGCGGCCCCTGCACCAGCCCTGTCCTGGAGAGACCGTCGCAAAG",
    },
}

# Standard genetic code
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# Amino acid abbreviations for display
AA_NAMES = {
    "F": "Phe", "L": "Leu", "I": "Ile", "M": "Met", "V": "Val",
    "S": "Ser", "P": "Pro", "T": "Thr", "A": "Ala", "Y": "Tyr",
    "*": "Stop", "H": "His", "Q": "Gln", "N": "Asn", "K": "Lys",
    "D": "Asp", "E": "Glu", "C": "Cys", "W": "Trp", "R": "Arg",
    "G": "Gly",
}

# DNA base colors
BASE_COLORS = {"A": "#2ecc71", "T": "#e74c3c", "G": "#f39c12", "C": "#3498db"}


# ------------------------------------------------------------------
# Report styling — genomics-themed with DNA helix accents
# ------------------------------------------------------------------

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #2563eb; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #1e40af; margin-top: 20px; }
  .report .card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
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
  .report .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 12px 0; }
  @media (max-width: 700px) { .report .compare-grid { grid-template-columns: 1fr; } }
</style>
"""


def _wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# Sequence analysis helpers
# ------------------------------------------------------------------

def _gc_content(seq: str) -> float:
    """Compute GC content as a fraction."""
    if not seq:
        return 0.0
    gc = sum(1 for b in seq.upper() if b in "GC")
    return gc / len(seq)


def _gc_content_sliding(seq: str, window: int = 30) -> list[float]:
    """Compute GC content with a sliding window."""
    if len(seq) < window:
        return [_gc_content(seq)]
    return [_gc_content(seq[i:i + window]) for i in range(len(seq) - window + 1)]


def _base_composition(seq: str) -> dict[str, float]:
    """Compute base frequencies."""
    seq = seq.upper()
    n = len(seq) or 1
    return {base: seq.count(base) / n for base in "ATGC"}


def _codon_usage(seq: str) -> dict[str, int]:
    """Count codons in reading frame 0."""
    seq = seq.upper()
    counts = {}
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        if len(codon) == 3 and all(b in "ATGC" for b in codon):
            counts[codon] = counts.get(codon, 0) + 1
    return counts


def _translate(seq: str) -> str:
    """Translate DNA to protein in reading frame 0."""
    seq = seq.upper()
    protein = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        if len(codon) == 3:
            aa = CODON_TABLE.get(codon, "?")
            if aa == "*":
                break
            protein.append(aa)
    return "".join(protein)


def _find_orfs(seq: str, min_length: int = 30) -> list[dict]:
    """Find open reading frames (ATG to stop) in all 3 forward frames."""
    seq = seq.upper()
    orfs = []
    stop_codons = {"TAA", "TAG", "TGA"}

    for frame in range(3):
        i = frame
        while i < len(seq) - 2:
            codon = seq[i:i + 3]
            if codon == "ATG":
                # Found start — scan for stop
                start = i
                j = i + 3
                while j < len(seq) - 2:
                    c = seq[j:j + 3]
                    if c in stop_codons:
                        orf_len = j + 3 - start
                        if orf_len >= min_length:
                            protein = _translate(seq[start:j])
                            orfs.append({
                                "frame": frame,
                                "start": start,
                                "end": j + 3,
                                "length_nt": orf_len,
                                "length_aa": len(protein),
                                "protein": protein[:50] + ("..." if len(protein) > 50 else ""),
                            })
                        i = j + 3
                        break
                    j += 3
                else:
                    # No stop found — ORF runs to end
                    orf_len = len(seq) - start
                    if orf_len >= min_length:
                        protein = _translate(seq[start:])
                        orfs.append({
                            "frame": frame,
                            "start": start,
                            "end": len(seq),
                            "length_nt": orf_len,
                            "length_aa": len(protein),
                            "protein": protein[:50] + ("..." if len(protein) > 50 else ""),
                        })
                    break
            i += 3 if seq[i:i + 3] != "ATG" else 1
            # Advance by 1 to catch overlapping ORFs
            if codon != "ATG":
                i = i  # already advanced by 3

    orfs.sort(key=lambda x: x["length_nt"], reverse=True)
    return orfs


def _dinucleotide_freq(seq: str) -> dict[str, float]:
    """Compute dinucleotide frequencies."""
    seq = seq.upper()
    counts = {}
    total = 0
    for i in range(len(seq) - 1):
        di = seq[i:i + 2]
        if all(b in "ATGC" for b in di):
            counts[di] = counts.get(di, 0) + 1
            total += 1
    return {di: c / total for di, c in counts.items()} if total else {}


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
    value_format: str = ".1f",
) -> str:
    """Generate an SVG grouped bar chart."""
    if not labels:
        return ""

    default_colors = ["#2563eb", "#dc2626", "#059669", "#f59e0b", "#7c3aed"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 80
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for vals in series.values() for v in vals]
    y_max = max(all_vals) if all_vals else 1
    y_max_plot = y_max * 1.15 or 1

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

    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        svg.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" stroke="#e9ecef" stroke-width="1"/>')
        svg.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{y_tick:{value_format}}</text>')

    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" height="{max(0, bh):.1f}" fill="{color}" rx="2"/>')
            svg.append(f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-size="9" fill="#1a1a2e">{val:{value_format}}</text>')
        lx = gx + n_series * bar_width / 2
        svg.append(f'<text x="{lx:.1f}" y="{mt + ch + 14}" text-anchor="end" font-size="10" fill="#6c757d" transform="rotate(-35, {lx:.1f}, {mt + ch + 14})">{label}</text>')

    if title:
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    if n_series > 1:
        lx = ml + cw - len(series) * 120
        for si, name in enumerate(series):
            color = colors[si % len(colors)]
            svg.append(f'<rect x="{lx + si * 120}" y="{mt + ch + 55}" width="12" height="12" rx="2" fill="{color}"/>')
            svg.append(f'<text x="{lx + si * 120 + 16}" y="{mt + ch + 66}" font-size="11" fill="#1a1a2e">{name}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_sparkline(
    values: list[float],
    title: str = "",
    width: int = 600,
    height: int = 100,
    color: str = "#2563eb",
    fill_color: str = "#dbeafe",
    ref_line: float | None = None,
    ref_label: str = "",
) -> str:
    """Generate an SVG sparkline/area chart."""
    if not values or len(values) < 2:
        return ""

    ml, mr, mt, mb = 40, 20, 25 if title else 8, 20
    cw = width - ml - mr
    ch = height - mt - mb

    v_min = min(values)
    v_max = max(values)
    v_pad = (v_max - v_min) * 0.1 or 0.05
    v_min -= v_pad
    v_max += v_pad
    v_range = v_max - v_min or 1

    def sx(i):
        return ml + (i / (len(values) - 1)) * cw

    def sy(v):
        return mt + ch - (v - v_min) / v_range * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{width / 2}" y="16" text-anchor="middle" font-size="12" font-weight="600" fill="#1a1a2e">{title}</text>')

    # Fill area
    points = [f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values)]
    fill_points = f"{sx(0):.1f},{mt + ch} " + " ".join(points) + f" {sx(len(values) - 1):.1f},{mt + ch}"
    svg.append(f'<polygon points="{fill_points}" fill="{fill_color}" opacity="0.5"/>')
    svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.5"/>')

    # Reference line
    if ref_line is not None and v_min <= ref_line <= v_max:
        ry = sy(ref_line)
        svg.append(f'<line x1="{ml}" y1="{ry:.1f}" x2="{ml + cw}" y2="{ry:.1f}" stroke="#dc2626" stroke-width="1" stroke-dasharray="4,3" opacity="0.6"/>')
        if ref_label:
            svg.append(f'<text x="{ml + cw + 4}" y="{ry + 3:.1f}" font-size="9" fill="#dc2626">{ref_label}</text>')

    # Y-axis labels
    for i in range(3):
        yv = v_min + v_range * i / 2
        py = sy(yv)
        svg.append(f'<text x="{ml - 6}" y="{py + 3:.1f}" text-anchor="end" font-size="9" fill="#9ca3af">{yv:.2f}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_heatmap(
    matrix: list[list[float]],
    row_labels: list[str],
    col_labels: list[str],
    title: str = "",
    width: int = 700,
    height: int = 500,
    value_format: str = ".3f",
) -> str:
    """Generate an SVG heatmap."""
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if matrix else 0
    if not n_rows or not n_cols:
        return ""

    show_values = n_rows <= 16 and n_cols <= 16

    flat = [v for row in matrix for v in row]
    v_min = min(flat)
    v_max = max(flat)
    v_range = v_max - v_min or 1

    def get_color(v):
        t = (v - v_min) / v_range
        r = int(255 - t * (255 - 30))
        g = int(255 - t * (255 - 58))
        b = int(255 - t * (255 - 95))
        return f"rgb({r},{g},{b})"

    ml = max(50, max(len(l) for l in row_labels) * 7 + 10) if row_labels else 50
    mr = 20
    mt = 70 if col_labels else 40
    mb = 20
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
        svg.append(f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    for j, label in enumerate(col_labels):
        cx = ml + j * cell_w + cell_w / 2
        svg.append(f'<text x="{cx:.1f}" y="{mt - 8}" text-anchor="end" font-size="9" fill="#374151" transform="rotate(-45, {cx:.1f}, {mt - 8})">{label}</text>')

    for i, row_label in enumerate(row_labels):
        ry = mt + i * cell_h + cell_h / 2
        svg.append(f'<text x="{ml - 6}" y="{ry + 3:.1f}" text-anchor="end" font-size="9" fill="#374151">{row_label}</text>')
        for j in range(n_cols):
            val = matrix[i][j]
            color = get_color(val)
            cx = ml + j * cell_w
            cy = mt + i * cell_h
            svg.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{color}" stroke="#fff" stroke-width="1"/>')
            if show_values:
                t = (val - v_min) / v_range
                text_color = "#fff" if t > 0.55 else "#1a1a2e"
                fs = min(9, int(cell_w / 4), int(cell_h / 2.5))
                fs = max(6, fs)
                svg.append(f'<text x="{cx + cell_w / 2:.1f}" y="{cy + cell_h / 2 + 3:.1f}" text-anchor="middle" font-size="{fs}" fill="{text_color}">{val:{value_format}}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_dna_track(
    sequence: str,
    label: str = "",
    highlight_start: int | None = None,
    width: int = 900,
) -> str:
    """Render a color-coded DNA sequence with optional prompt/generated boundary."""
    chars_per_line = 60
    char_w = 11
    line_h = 20
    label_w = 50
    n_lines = (len(sequence) + chars_per_line - 1) // chars_per_line
    svg_h = n_lines * line_h + 40

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {svg_h}" '
        f'style="width:100%;max-width:{width}px;height:auto;font-family:monospace;">',
        f'<rect width="{width}" height="{svg_h}" fill="#fff" rx="6"/>',
    ]

    if label:
        svg.append(f'<text x="{width / 2}" y="14" text-anchor="middle" font-size="11" font-weight="600" fill="#1e3a5f">{label}</text>')

    y_offset = 26

    for line_idx in range(n_lines):
        start = line_idx * chars_per_line
        end = min(start + chars_per_line, len(sequence))
        y = y_offset + line_idx * line_h

        svg.append(f'<text x="{label_w - 8}" y="{y}" text-anchor="end" font-size="9" fill="#9ca3af">{start + 1}</text>')

        for ci, base in enumerate(sequence[start:end]):
            abs_pos = start + ci
            x = label_w + ci * char_w
            color = BASE_COLORS.get(base.upper(), "#6b7280")
            is_generated = highlight_start is not None and abs_pos >= highlight_start

            # Background: light for prompt, slightly tinted for generated
            bg_opacity = "0.15" if is_generated else "0.08"
            svg.append(f'<rect x="{x}" y="{y - 12}" width="{char_w}" height="{line_h - 2}" fill="{color}" opacity="{bg_opacity}" rx="1"/>')
            svg.append(f'<text x="{x + char_w / 2:.1f}" y="{y}" text-anchor="middle" font-size="10" font-weight="500" fill="{color}">{base.upper()}</text>')

            if (abs_pos + 1) % 10 == 0 and ci < end - start - 1:
                svg.append(f'<line x1="{x + char_w + 1}" y1="{y - 11}" x2="{x + char_w + 1}" y2="{y + 3}" stroke="#e5e7eb" stroke-width="0.5"/>')

        # Boundary marker
        if highlight_start is not None and start <= highlight_start < end:
            bx = label_w + (highlight_start - start) * char_w
            svg.append(f'<line x1="{bx}" y1="{y - 13}" x2="{bx}" y2="{y + 5}" stroke="#dc2626" stroke-width="2" stroke-dasharray="3,2"/>')

    # Legend
    ly = svg_h - 10
    lx = label_w
    for base, color in BASE_COLORS.items():
        svg.append(f'<rect x="{lx}" y="{ly - 8}" width="8" height="8" rx="1" fill="{color}"/>')
        svg.append(f'<text x="{lx + 12}" y="{ly}" font-size="9" fill="#6b7280">{base}</text>')
        lx += 30
    if highlight_start is not None:
        lx += 10
        svg.append(f'<line x1="{lx}" y1="{ly - 6}" x2="{lx + 12}" y2="{ly - 6}" stroke="#dc2626" stroke-width="2" stroke-dasharray="3,2"/>')
        svg.append(f'<text x="{lx + 16}" y="{ly}" font-size="9" fill="#6b7280">Prompt | Generated</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_orf_track(
    orfs: list[dict],
    seq_length: int,
    title: str = "",
    width: int = 700,
    height: int = 140,
) -> str:
    """Generate an SVG showing ORFs as arrows on a sequence track."""
    if not orfs:
        return f'<div style="color:#6c757d;font-size:0.9em;text-align:center;padding:8px;">No ORFs found (&ge;30 nt)</div>'

    ml, mr, mt, mb = 50, 20, 30, 20
    cw = width - ml - mr
    ch = height - mt - mb
    frame_colors = ["#2563eb", "#059669", "#f59e0b"]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{width / 2}" y="18" text-anchor="middle" font-size="12" font-weight="600" fill="#1a1a2e">{title}</text>')

    # Frame tracks
    frame_h = ch / 3 - 4
    for frame in range(3):
        fy = mt + frame * (frame_h + 4)
        color = frame_colors[frame]

        # Track background
        svg.append(f'<rect x="{ml}" y="{fy}" width="{cw}" height="{frame_h}" fill="{color}" opacity="0.05" rx="3"/>')
        svg.append(f'<text x="{ml - 6}" y="{fy + frame_h / 2 + 3:.1f}" text-anchor="end" font-size="9" fill="#6c757d">+{frame + 1}</text>')

        # ORFs in this frame
        frame_orfs = [o for o in orfs if o["frame"] == frame]
        for orf in frame_orfs:
            ox = ml + (orf["start"] / seq_length) * cw
            ow = max(4, (orf["length_nt"] / seq_length) * cw)

            # Arrow body
            svg.append(f'<rect x="{ox:.1f}" y="{fy + 2}" width="{ow:.1f}" height="{frame_h - 4}" fill="{color}" opacity="0.7" rx="2"/>')

            # Label if wide enough
            if ow > 40:
                svg.append(f'<text x="{ox + ow / 2:.1f}" y="{fy + frame_h / 2 + 3:.1f}" text-anchor="middle" font-size="8" fill="#fff" font-weight="600">{orf["length_aa"]}aa</text>')

    # Scale
    svg.append(f'<text x="{ml}" y="{height - 4}" font-size="9" fill="#9ca3af">1</text>')
    svg.append(f'<text x="{ml + cw}" y="{height - 4}" text-anchor="end" font-size="9" fill="#9ca3af">{seq_length}bp</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def _make_codon_wheel(
    usage: dict[str, int],
    title: str = "",
    width: int = 300,
    height: int = 300,
) -> str:
    """Generate an SVG donut chart of amino acid usage from codon counts."""
    # Aggregate by amino acid
    aa_counts = {}
    for codon, count in usage.items():
        aa = CODON_TABLE.get(codon, "?")
        if aa != "?" and aa != "*":
            aa_counts[aa] = aa_counts.get(aa, 0) + count

    if not aa_counts:
        return ""

    total = sum(aa_counts.values())
    # Sort by count descending, take top 10
    sorted_aa = sorted(aa_counts.items(), key=lambda x: x[1], reverse=True)[:12]
    other = total - sum(c for _, c in sorted_aa)

    cx, cy = width / 2, height / 2 + 10
    r_outer = min(cx, cy) - 40
    r_inner = r_outer * 0.55

    aa_colors = [
        "#2563eb", "#059669", "#dc2626", "#f59e0b", "#7c3aed",
        "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
        "#14b8a6", "#e11d48",
    ]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    if title:
        svg.append(f'<text x="{width / 2}" y="18" text-anchor="middle" font-size="12" font-weight="600" fill="#1a1a2e">{title}</text>')

    angle = -math.pi / 2
    items = sorted_aa + ([("Other", other)] if other > 0 else [])

    for i, (aa, count) in enumerate(items):
        frac = count / total
        sweep = frac * 2 * math.pi
        if sweep < 0.01:
            angle += sweep
            continue

        color = aa_colors[i % len(aa_colors)] if aa != "Other" else "#d1d5db"

        x1_o = cx + r_outer * math.cos(angle)
        y1_o = cy + r_outer * math.sin(angle)
        x2_o = cx + r_outer * math.cos(angle + sweep)
        y2_o = cy + r_outer * math.sin(angle + sweep)
        x1_i = cx + r_inner * math.cos(angle + sweep)
        y1_i = cy + r_inner * math.sin(angle + sweep)
        x2_i = cx + r_inner * math.cos(angle)
        y2_i = cy + r_inner * math.sin(angle)

        large = 1 if sweep > math.pi else 0

        path = (
            f"M {x1_o:.1f},{y1_o:.1f} "
            f"A {r_outer},{r_outer} 0 {large},1 {x2_o:.1f},{y2_o:.1f} "
            f"L {x1_i:.1f},{y1_i:.1f} "
            f"A {r_inner},{r_inner} 0 {large},0 {x2_i:.1f},{y2_i:.1f} Z"
        )
        svg.append(f'<path d="{path}" fill="{color}" stroke="#fff" stroke-width="1.5"/>')

        # Label
        if frac > 0.04:
            mid_angle = angle + sweep / 2
            lr = (r_outer + r_inner) / 2
            lx = cx + lr * math.cos(mid_angle)
            ly = cy + lr * math.sin(mid_angle)
            label = f"{AA_NAMES.get(aa, aa)}" if aa != "Other" else "Other"
            svg.append(f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle" font-size="8" fill="#fff" font-weight="600">{label}</text>')

        angle += sweep

    # Center text
    svg.append(f'<text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="14" font-weight="700" fill="#1e3a5f">{total}</text>')
    svg.append(f'<text x="{cx}" y="{cy + 12}" text-anchor="middle" font-size="9" fill="#6c757d">codons</text>')

    svg.append("</svg>")
    return "\n".join(svg)


# ------------------------------------------------------------------
# Task 1: Load gene prompts
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def load_prompts(
    prompts_json: str = "",
) -> flyte.io.Dir:
    """Load gene prompt sequences and their real continuations for comparison."""
    if prompts_json:
        genes = json.loads(prompts_json)
    else:
        genes = DEFAULT_GENE_PROMPTS

    valid_bases = set("ATGC")
    for name, data in genes.items():
        for key in ("prompt", "full_sequence"):
            seq = data.get(key, "").upper()
            invalid = set(seq) - valid_bases
            if invalid:
                log.warning(f"{name}: invalid bases {invalid} in {key}")
                data[key] = "".join(b for b in seq if b in valid_bases)

    log.info(f"Loaded {len(genes)} gene prompts")

    out_dir = tempfile.mkdtemp(prefix="dna_gen_")
    with open(os.path.join(out_dir, "prompts.json"), "w") as f:
        json.dump(genes, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Generate DNA with Carbon
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def generate_dna(
    prompts_dir: flyte.io.Dir,
    model_name: str = "HuggingFaceBio/Carbon-3B",
    gen_length: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate DNA continuations from prompts using Carbon.

    For each gene prompt, generates gen_length new bases and returns
    both the generated and real continuations for comparison.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info(f"Loading Carbon model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("Running on CPU — generation will be slow. GPU recommended.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    prompts_path = await prompts_dir.download()
    with open(os.path.join(prompts_path, "prompts.json")) as f:
        genes = json.load(f)

    results = {}
    total = len(genes)

    progress_html = """
    <h2>Carbon DNA Generation</h2>
    <div class="card">
        <b>Model:</b> {model}<br>
        <b>Device:</b> {device}<br>
        <b>Temperature:</b> {temp} | <b>Top-p:</b> {top_p}<br>
        <b>Progress:</b> {done}/{total} genes generated
    </div>
    """

    for i, (gene_name, gene_data) in enumerate(genes.items()):
        await flyte.report.replace.aio(
            _wrap_report(progress_html.format(
                model=model_name, device=device, temp=temperature,
                top_p=top_p, done=i, total=total
            )),
            do_flush=True,
        )

        prompt_seq = gene_data["prompt"]
        full_seq = gene_data["full_sequence"]
        real_continuation = full_seq[len(prompt_seq):]

        # Generate with Carbon
        dna_prompt = f"<dna>{prompt_seq}"
        inputs = tokenizer(dna_prompt, return_tensors="pt", add_special_tokens=False).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=gen_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
            )

        generated_full = tokenizer.decode(output[0], skip_special_tokens=False)
        # Extract just the DNA portion after <dna> tag
        if "<dna>" in generated_full:
            generated_dna = generated_full.split("<dna>")[-1].strip()
        else:
            generated_dna = generated_full.strip()

        # Clean: keep only valid DNA bases
        generated_dna = "".join(b for b in generated_dna.upper() if b in "ATGC")
        generated_continuation = generated_dna[len(prompt_seq):]

        log.info(f"  {gene_name}: generated {len(generated_continuation)}bp continuation")

        results[gene_name] = {
            "description": gene_data["description"],
            "prompt": prompt_seq,
            "real_continuation": real_continuation[:gen_length],
            "generated_continuation": generated_continuation[:gen_length],
            "full_generated": prompt_seq + generated_continuation[:gen_length],
            "full_real": full_seq[:len(prompt_seq) + gen_length],
        }

    # Generation report
    html_parts = [
        "<h2>Carbon DNA Generation</h2>",
        '<div class="stat-grid">',
        f'<div class="stat"><div class="value">{total}</div><div class="label">Genes</div></div>',
        f'<div class="stat"><div class="value">{gen_length}bp</div><div class="label">Generated per Gene</div></div>',
        f'<div class="stat"><div class="value">{model_name.split("/")[-1]}</div><div class="label">Model</div></div>',
        f'<div class="stat"><div class="value">{temperature}</div><div class="label">Temperature</div></div>',
        "</div>",
    ]

    for gene_name, data in results.items():
        short = gene_name.split("(")[0].strip()
        html_parts.append(f'<h3>{gene_name}</h3>')
        html_parts.append(f'<div class="note">{data["description"]}</div>')

        # Show generated sequence with prompt boundary
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_dna_track(
            data["full_generated"],
            label=f"{short} — Prompt + Generated",
            highlight_start=len(data["prompt"]),
        ))
        html_parts.append("</div>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    return json.dumps(results)


# ------------------------------------------------------------------
# Task 3: Analyze and compare sequences
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def analyze_sequences(
    generation_json: str,
) -> str:
    """Analyze generated vs real sequences: GC content, codon usage, ORFs, composition."""
    results = json.loads(generation_json)

    html_parts = ["<h2>Sequence Analysis — Generated vs Real</h2>"]

    all_analyses = {}

    for gene_name, data in results.items():
        short = gene_name.split("(")[0].strip()
        real = data["real_continuation"]
        generated = data["generated_continuation"]

        # Analyze both
        analysis = {
            "real": {
                "length": len(real),
                "gc_content": round(_gc_content(real), 4),
                "base_comp": _base_composition(real),
                "codon_usage": _codon_usage(real),
                "orfs": _find_orfs(real),
                "gc_profile": _gc_content_sliding(real),
                "dinucleotide": _dinucleotide_freq(real),
            },
            "generated": {
                "length": len(generated),
                "gc_content": round(_gc_content(generated), 4),
                "base_comp": _base_composition(generated),
                "codon_usage": _codon_usage(generated),
                "orfs": _find_orfs(generated),
                "gc_profile": _gc_content_sliding(generated),
                "dinucleotide": _dinucleotide_freq(generated),
            },
        }
        all_analyses[gene_name] = analysis

        # --- Per-gene visualizations ---
        html_parts.append(f'<h3>{gene_name}</h3>')
        html_parts.append(f'<div class="note">{data["description"]}</div>')

        # Quick stats comparison
        gc_diff = abs(analysis["generated"]["gc_content"] - analysis["real"]["gc_content"])
        gc_match = "badge-success" if gc_diff < 0.05 else "badge-warning" if gc_diff < 0.1 else "badge-danger"

        html_parts.append('<div class="stat-grid">')
        html_parts.append(f'<div class="stat"><div class="value">{analysis["real"]["gc_content"]:.1%}</div><div class="label">Real GC%</div></div>')
        html_parts.append(f'<div class="stat"><div class="value">{analysis["generated"]["gc_content"]:.1%}</div><div class="label">Generated GC%</div></div>')
        html_parts.append(f'<div class="stat"><div class="value"><span class="badge {gc_match}">{gc_diff:.1%}</span></div><div class="label">GC Difference</div></div>')
        html_parts.append(f'<div class="stat"><div class="value">{len(analysis["real"]["orfs"])}</div><div class="label">Real ORFs</div></div>')
        html_parts.append(f'<div class="stat"><div class="value">{len(analysis["generated"]["orfs"])}</div><div class="label">Generated ORFs</div></div>')
        html_parts.append("</div>")

        # Base composition comparison
        bases = ["A", "T", "G", "C"]
        real_comp = [analysis["real"]["base_comp"].get(b, 0) for b in bases]
        gen_comp = [analysis["generated"]["base_comp"].get(b, 0) for b in bases]
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_bar_chart(
            bases,
            {"Real": real_comp, "Generated": gen_comp},
            title=f"{short} — Base Composition",
            colors=["#2563eb", "#f59e0b"],
            value_format=".2f",
            height=250,
        ))
        html_parts.append("</div>")

        # GC content sliding window
        html_parts.append('<div class="compare-grid">')
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_sparkline(
            analysis["real"]["gc_profile"],
            title=f"{short} Real — GC Content (30bp window)",
            color="#2563eb",
            fill_color="#dbeafe",
            ref_line=0.5,
            ref_label="50%",
        ))
        html_parts.append("</div>")
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_sparkline(
            analysis["generated"]["gc_profile"],
            title=f"{short} Generated — GC Content (30bp window)",
            color="#f59e0b",
            fill_color="#fef3c7",
            ref_line=0.5,
            ref_label="50%",
        ))
        html_parts.append("</div>")
        html_parts.append("</div>")

        # Codon usage wheels
        html_parts.append('<div class="compare-grid">')
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_codon_wheel(
            analysis["real"]["codon_usage"],
            title=f"Real — Amino Acid Usage",
        ))
        html_parts.append("</div>")
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_codon_wheel(
            analysis["generated"]["codon_usage"],
            title=f"Generated — Amino Acid Usage",
        ))
        html_parts.append("</div>")
        html_parts.append("</div>")

        # ORF tracks
        html_parts.append('<div class="compare-grid">')
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_orf_track(
            analysis["real"]["orfs"],
            len(real),
            title=f"Real — Open Reading Frames",
        ))
        html_parts.append("</div>")
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_orf_track(
            analysis["generated"]["orfs"],
            len(generated),
            title=f"Generated — Open Reading Frames",
        ))
        html_parts.append("</div>")
        html_parts.append("</div>")

        # ORF table if any found
        all_orfs = [(s, o) for s, orfs in [("Real", analysis["real"]["orfs"]), ("Generated", analysis["generated"]["orfs"])] for o in orfs]
        if all_orfs:
            html_parts.append("<table><tr><th>Source</th><th>Frame</th><th>Start</th><th>Length (nt)</th><th>Length (aa)</th><th>Protein (first 50aa)</th></tr>")
            for source, orf in all_orfs[:10]:
                badge = "badge-info" if source == "Real" else "badge-warning"
                html_parts.append(
                    f'<tr><td><span class="badge {badge}">{source}</span></td>'
                    f'<td>+{orf["frame"] + 1}</td><td>{orf["start"]}</td>'
                    f'<td>{orf["length_nt"]}</td><td>{orf["length_aa"]}</td>'
                    f'<td style="font-family:monospace;font-size:0.8em">{orf["protein"]}</td></tr>'
                )
            html_parts.append("</table>")

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)
    return json.dumps(all_analyses)


# ------------------------------------------------------------------
# Task 4: Generate summary report
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def generate_summary(
    generation_json: str,
    analysis_json: str,
) -> str:
    """Generate cross-gene summary comparing real vs generated DNA."""
    results = json.loads(generation_json)
    analyses = json.loads(analysis_json)

    html_parts = [
        "<h2>DNA Generation Summary — Carbon vs Reality</h2>",
        '<div class="note">'
        "This pipeline uses <b>HuggingFace Carbon</b> to generate DNA sequence continuations "
        "from real gene prompts, then compares the generated sequences against the actual genes. "
        "A good genomic language model should produce DNA that looks statistically similar to "
        "real DNA — matching GC content, codon usage patterns, and producing valid open reading frames."
        "</div>",
    ]

    # Cross-gene comparison table
    gene_names = []
    real_gcs = []
    gen_gcs = []
    real_orf_counts = []
    gen_orf_counts = []

    for gene_name, analysis in analyses.items():
        short = gene_name.split("(")[0].strip()
        gene_names.append(short)
        real_gcs.append(analysis["real"]["gc_content"])
        gen_gcs.append(analysis["generated"]["gc_content"])
        real_orf_counts.append(len(analysis["real"]["orfs"]))
        gen_orf_counts.append(len(analysis["generated"]["orfs"]))

    # GC content comparison chart
    html_parts.append('<div class="chart-container">')
    html_parts.append(_make_bar_chart(
        gene_names,
        {"Real GC%": real_gcs, "Generated GC%": gen_gcs},
        title="GC Content — Real vs Generated",
        colors=["#2563eb", "#f59e0b"],
        value_format=".2f",
    ))
    html_parts.append("</div>")

    # Summary stats
    gc_diffs = [abs(r - g) for r, g in zip(real_gcs, gen_gcs)]
    avg_gc_diff = sum(gc_diffs) / len(gc_diffs) if gc_diffs else 0
    close_gc = sum(1 for d in gc_diffs if d < 0.05)

    html_parts.append('<div class="stat-grid">')
    html_parts.append(f'<div class="stat"><div class="value">{len(gene_names)}</div><div class="label">Genes Analyzed</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{avg_gc_diff:.1%}</div><div class="label">Avg GC Difference</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{close_gc}/{len(gene_names)}</div><div class="label">GC Within 5%</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{sum(gen_orf_counts)}</div><div class="label">Generated ORFs</div></div>')
    html_parts.append(f'<div class="stat"><div class="value">{sum(real_orf_counts)}</div><div class="label">Real ORFs</div></div>')
    html_parts.append("</div>")

    # Detailed comparison table
    html_parts.append("<h3>Per-Gene Comparison</h3>")
    html_parts.append(
        "<table><tr><th>Gene</th><th>Real GC%</th><th>Gen GC%</th>"
        "<th>Diff</th><th>Real ORFs</th><th>Gen ORFs</th><th>Match Quality</th></tr>"
    )

    for i, gene_name in enumerate(gene_names):
        diff = gc_diffs[i]
        badge = "badge-success" if diff < 0.05 else "badge-warning" if diff < 0.1 else "badge-danger"
        quality = "Excellent" if diff < 0.03 else "Good" if diff < 0.05 else "Fair" if diff < 0.1 else "Poor"
        html_parts.append(
            f"<tr><td><b>{gene_name}</b></td>"
            f"<td>{real_gcs[i]:.1%}</td><td>{gen_gcs[i]:.1%}</td>"
            f'<td><span class="badge {badge}">{diff:.1%}</span></td>'
            f"<td>{real_orf_counts[i]}</td><td>{gen_orf_counts[i]}</td>"
            f'<td><span class="badge {badge}">{quality}</span></td></tr>'
        )
    html_parts.append("</table>")

    # Dinucleotide frequency heatmap (cross-gene average)
    dinucs = sorted(set().union(*(
        set(a["real"]["dinucleotide"].keys()) | set(a["generated"]["dinucleotide"].keys())
        for a in analyses.values()
    )))
    if dinucs:
        matrix = []
        row_labels = []
        for gene_name, analysis in analyses.items():
            short = gene_name.split("(")[0].strip()
            real_freqs = [analysis["real"]["dinucleotide"].get(d, 0) for d in dinucs]
            gen_freqs = [analysis["generated"]["dinucleotide"].get(d, 0) for d in dinucs]
            # Show the difference (generated - real)
            diff_row = [g - r for g, r in zip(gen_freqs, real_freqs)]
            matrix.append(diff_row)
            row_labels.append(short)

        html_parts.append("<h3>Dinucleotide Frequency Deviation (Generated - Real)</h3>")
        html_parts.append(
            '<div class="note">'
            "Values near zero (lighter) mean Carbon matches the real dinucleotide frequencies. "
            "Consistent patterns across genes would indicate systematic biases in the model."
            "</div>"
        )
        html_parts.append('<div class="chart-container">')
        html_parts.append(_make_heatmap(
            matrix,
            row_labels,
            dinucs,
            title="Dinucleotide Frequency Deviation",
            value_format=".3f",
            height=max(250, len(row_labels) * 50 + 100),
        ))
        html_parts.append("</div>")

    # Method note
    html_parts.append(
        '<div class="note">'
        "<b>Method:</b> Each gene's first ~60bp is used as a prompt for Carbon's autoregressive "
        "generation (temperature sampling). The generated continuation is then compared against "
        "the actual gene sequence across multiple metrics: base composition, GC content profiles, "
        "codon usage, open reading frame structure, and dinucleotide frequencies.<br><br>"
        "<b>What good generation looks like:</b> GC content within 5% of real, similar codon usage "
        "distribution, comparable number and length of ORFs, and no systematic dinucleotide biases. "
        "Note that exact sequence matching is not expected — generation is stochastic and the model "
        "is producing plausible DNA, not memorizing specific genes."
        "</div>"
    )

    await flyte.report.replace.aio(_wrap_report("\n".join(html_parts)), do_flush=True)

    summary = {
        "total_genes": len(gene_names),
        "avg_gc_diff": round(avg_gc_diff, 4),
        "gc_within_5pct": close_gc,
        "total_real_orfs": sum(real_orf_counts),
        "total_generated_orfs": sum(gen_orf_counts),
    }
    return json.dumps(summary)


# ------------------------------------------------------------------
# Pipeline orchestrator
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    prompts_json: str = "",
    model_name: str = "HuggingFaceBio/Carbon-3B",
    gen_length: int = 200,
    temperature: float = 0.8,
) -> tuple[str, str]:
    """
    End-to-end DNA generation and analysis pipeline.

    Returns (generation JSON, analysis JSON).

    1. Load gene sequence prompts
    2. Generate DNA continuations with Carbon
    3. Analyze and compare generated vs real sequences
    4. Generate comprehensive summary report
    """
    log.info("Starting DNA generation & analysis pipeline...")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = [
            "Load Prompts",
            "Carbon Generation",
            "Analyze Sequences",
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
        <h2>DNA Generation &amp; Analysis</h2>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Stage 1
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(1, "Loading gene prompts...")),
        do_flush=True,
    )
    prompts_dir = await load_prompts(prompts_json=prompts_json)

    # Stage 2
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(2, "Generating DNA with Carbon...")),
        do_flush=True,
    )
    gen_json = await generate_dna(
        prompts_dir=prompts_dir,
        model_name=model_name,
        gen_length=gen_length,
        temperature=temperature,
    )

    # Stage 3
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(3, "Analyzing generated vs real sequences...")),
        do_flush=True,
    )
    analysis_json = await analyze_sequences(generation_json=gen_json)

    # Stage 4
    await flyte.report.replace.aio(
        _wrap_report(_pipeline_progress(4, "Generating summary report...")),
        do_flush=True,
    )
    summary_json = await generate_summary(generation_json=gen_json, analysis_json=analysis_json)

    # Final report
    summary = json.loads(summary_json)
    final_html = f"""
    <h2>Pipeline Complete</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{summary['total_genes']}</div><div class="label">Genes</div></div>
      <div class="stat"><div class="value">{gen_length}bp</div><div class="label">Generated / Gene</div></div>
      <div class="stat"><div class="value">{summary['avg_gc_diff']:.1%}</div><div class="label">Avg GC Diff</div></div>
      <div class="stat"><div class="value">{summary['gc_within_5pct']}/{summary['total_genes']}</div><div class="label">GC Within 5%</div></div>
      <div class="stat"><div class="value">{summary['total_generated_orfs']}</div><div class="label">Generated ORFs</div></div>
    </div>
    <div class="card">
      <b>Model:</b> {model_name} |
      <b>Temperature:</b> {temperature} |
      <b>Genes:</b> {summary['total_genes']}
    </div>
    <div class="note">
      All 4 pipeline stages completed. View individual task reports for DNA sequence tracks,
      base composition charts, GC content profiles, codon usage wheels, ORF track diagrams,
      and dinucleotide frequency heatmaps.
    </div>
    """

    await flyte.report.replace.aio(_wrap_report(final_html), do_flush=True)
    log.info("Pipeline complete.")
    return gen_json, analysis_json
