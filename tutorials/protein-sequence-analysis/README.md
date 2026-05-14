# Protein Sequence Analysis

Analyze, compare, and visualize biophysical properties of real protein sequences using a Flyte pipeline with rich visual reports.

## What this tutorial does

A 4-stage pipeline that takes a set of protein sequences and produces comprehensive visual reports:

1. **Load Sequences** — Parses and validates protein sequences, outputs FASTA files. Default set includes Insulin, GFP, Lysozyme, Ubiquitin, Hemoglobin, p53, and Spider Silk.
2. **Analyze Properties** — Computes molecular weight, isoelectric point, instability index, GRAVY hydrophobicity, aromaticity, secondary structure fractions, and full amino acid composition using BioPython.
3. **Compute Similarity** — Builds a pairwise sequence identity matrix via global alignment, identifies most similar protein pairs.
4. **Generate Summary** — Creates a gallery of protein cards with hydrophobicity sparklines, amino acid frequency charts, and classification by size/stability/charge.

## Reports

Each task produces a styled HTML report with SVG visualizations:

- Bar charts comparing molecular weights and isoelectric points
- Grouped bar charts for secondary structure fractions (helix/turn/sheet)
- Amino acid composition heatmap (proteins x amino acids)
- Pairwise similarity heatmap
- Per-protein hydrophobicity profiles (Kyte-Doolittle sparklines)
- Classification tables with stability/size/charge badges

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Default — analyze 7 curated proteins
flyte run --local --tui workflow.py pipeline

# Custom sequences via JSON
flyte run --local --tui workflow.py pipeline \
  --sequences_json '{"MyProtein": "MVLSPADKTNVKAAWGKVGAH", "Another": "MQIFVKTLTGKTITLEVEPS"}'

# Remote execution
flyte run workflow.py pipeline

# Run individual tasks
flyte run --local --tui workflow.py load_sequences
flyte run --local --tui workflow.py analyze_properties --seq_dir /path/to/sequences
```

## Requirements

- Python 3.10+
- BioPython (sequence analysis)
- NumPy
- Matplotlib (available for local use)
