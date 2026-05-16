# Protein Sequence Analysis

Analyze, compare, and visualize biophysical properties of real protein sequences, run protein language models, and predict 3D structures — all orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why Protein Analysis Matters

Proteins are the molecular machines of life. Every enzyme that digests your food, every antibody that fights infection, every receptor that lets a cell respond to its environment — they're all proteins. A protein is just a string of amino acids (20 possible "letters"), but that sequence determines everything: how the protein folds, what it binds, whether it's stable, and what it does.

Being able to quickly characterize a set of proteins — their size, charge, stability, hydrophobicity, structural tendencies, and evolutionary relationships — is foundational to drug discovery, protein engineering, and understanding disease.

## What This Pipeline Does

This tutorial takes a set of protein sequences and runs a computational analysis pipeline that a bioinformatician might use as a first pass when studying a new set of proteins:

| Stage | What it answers | Biology context |
|-------|----------------|-----------------|
| **Load Sequences** | Are these valid proteins? How long are they? | Sequences are written in the one-letter amino acid code (e.g., `M` = methionine, `K` = lysine). Invalid characters get flagged. |
| **Analyze Properties** | How big? What charge? Stable or not? | Molecular weight tells you if it's a small peptide or a large enzyme. Isoelectric point (pI) tells you its charge behavior — critical for purification. Instability index predicts whether it'll survive in a test tube. |
| **Compute Similarity** | Are any of these proteins related? | Sequence alignment finds shared ancestry. Proteins with >30% identity usually share 3D structure and function. Below 25% is the "twilight zone" — can't tell from sequence alone. |
| **ESM-2 Embeddings** | What does a protein language model see? | ESM-2 (Meta) is a transformer trained on 250M protein sequences. It captures evolutionary and structural patterns that sequence alignment misses — two proteins can look very different in sequence but cluster together in embedding space because they share a fold. |
| **ESMFold 3D Structures** | What does this protein look like in 3D? | ESMFold predicts a protein's 3D structure directly from its sequence — no multiple sequence alignment needed. The report shows interactive rotating structures you can explore. |
| **Generate Summary** | What's the big picture? | Per-protein cards with hydrophobicity profiles (which regions are water-loving vs water-hating) and structural predictions. Classification by size, stability, and charge. |

## The Default Proteins

The pipeline ships with 7 real proteins chosen to show biological diversity:

| Protein | What it is | Why it's interesting |
|---------|-----------|---------------------|
| **Insulin (B-chain)** | Blood sugar regulator | Tiny (30 aa). First protein ever sequenced (1951). |
| **GFP** | Jellyfish fluorescent protein | Revolutionized cell biology — you can tag any protein with it to make it glow green under UV. Nobel Prize 2008. |
| **Lysozyme** | Antimicrobial enzyme | Found in tears and saliva. Destroys bacterial cell walls. One of the best-studied enzymes in history. |
| **Ubiquitin** | Protein recycling tag | Cells attach it to damaged proteins to mark them for destruction. Extremely conserved across all life. |
| **Hemoglobin Alpha** | Oxygen transport | Carries O2 from lungs to tissues. Mutations cause sickle cell disease. |
| **p53** | Tumor suppressor | "Guardian of the genome." Mutated in >50% of human cancers. |
| **Spider Silk (MaSp1)** | Structural fiber | Stronger than steel by weight. Highly repetitive sequence — mostly glycine and alanine. |

## Key Concepts Explained

**Molecular Weight (MW)** — The mass of the protein in Daltons (Da). Small peptides are ~3-5 kDa, typical enzymes 20-80 kDa, large complexes 100+ kDa.

**Isoelectric Point (pI)** — The pH where the protein has zero net charge. Below its pI, a protein is positively charged; above, negatively charged. This matters for lab techniques like gel electrophoresis and column purification.

**Instability Index** — A prediction of whether a protein will be stable in a test tube, based on dipeptide frequencies. Below 40 = predicted stable; above 40 = predicted unstable. Not perfect, but a useful first filter.

**GRAVY (Grand Average of Hydropathy)** — Measures overall hydrophobicity. Positive values mean the protein prefers to be in membranes or oil-like environments. Negative values mean it's happy in water (most soluble proteins).

**Secondary Structure** — Proteins fold into local patterns: alpha helices (springy coils), beta sheets (flat zig-zag ribbons), and turns/loops. The fraction of each gives a rough structural fingerprint.

**Sequence Similarity** — Two proteins with similar sequences likely evolved from a common ancestor and probably fold into similar shapes. Measured by aligning sequences and counting matching positions.

**Protein Language Models (ESM-2)** — Just as GPT learns language by reading billions of sentences, ESM-2 learns protein "grammar" from 250 million sequences. It picks up patterns that evolution has conserved: which residues tend to co-occur, which positions are flexible vs constrained, what a "typical" protein looks like. The result is an embedding — a vector representation — for each protein that encodes structural and functional information far beyond what raw sequence comparison can capture. This is the same technology that powered breakthroughs like ESMFold and is now standard in protein engineering labs.

**Contact Maps** — A contact map predicts which residue pairs are physically close in 3D space (typically <8 Angstroms). You can extract approximate contact maps from ESM-2's attention weights — the model has learned which residues "pay attention" to each other, and this correlates with spatial proximity. Contact maps look like symmetric heatmaps with patterns along the diagonal (local structure) and off-diagonal spots (long-range contacts that define the fold).

**ESMFold and pLDDT** — ESMFold is Meta's single-sequence structure prediction model. Unlike AlphaFold (which needs a multiple sequence alignment from evolutionary relatives), ESMFold predicts 3D coordinates directly from one sequence in seconds. **pLDDT** (predicted Local Distance Difference Test) is the per-residue confidence score: >90 means very high confidence (well-structured core), 70-90 is confident (most of the protein), 50-70 is low (flexible loops), and <50 suggests the region is disordered (no stable structure). In the report, pLDDT is shown as a colored sparkline using the same AlphaFold color scheme (dark blue → light blue → yellow → orange).

## Reports

Each task generates a Flyte report with rich visualizations:

- Color-coded amino acid sequences (residues colored by physicochemical class)
- MW vs pI scatter plot with stability coloring and pH 7 reference line
- Radar/spider charts showing each protein's normalized property profile
- Secondary structure fraction bar charts (helix/turn/sheet)
- Amino acid composition heatmap (proteins x 20 amino acids)
- Hierarchical clustering dendrogram (UPGMA) from sequence similarity
- Pairwise similarity heatmap
- ESM-2 embedding space scatter plot (t-SNE projection)
- ESM-2 cosine similarity heatmap (compare with sequence similarity)
- Per-protein contact maps from ESM-2 attention weights
- Interactive 3D protein structures (3Dmol.js viewers with spinning cartoon rendering)
- Per-residue pLDDT confidence sparklines (AlphaFold color scheme)
- Per-protein cards with hydrophobicity sparklines and structure tracks

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU for bioinformatics, GPU for ESM-2 and ESMFold |
| `workflow.py` | 6-stage pipeline: load → properties → similarity → ESM-2 → ESMFold 3D → summary |

## Setup

```bash
cd tutorials/protein-sequence-analysis

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Default — analyze 7 curated proteins
flyte run --local --tui workflow.py pipeline

# Remote execution (GPU for ESM-2 + ESMFold)
flyte run workflow.py pipeline

# Swap ESM-2 model (larger = better contact maps)
flyte run workflow.py pipeline --esm_model "facebook/esm2_t30_150M_UR50D"

# Custom sequences via JSON
flyte run --local --tui workflow.py pipeline \
  --sequences_json '{"MyProtein": "MVLSPADKTNVKAAWGKVGAH", "Another": "MQIFVKTLTGKTITLEVEPS"}'

# Run individual tasks
flyte run workflow.py load_sequences
flyte run workflow.py analyze_properties --seq_dir <path>
flyte run workflow.py esm_analysis --seq_dir <path>
flyte run workflow.py predict_structures --seq_dir <path>
```

## GPU Requirements

The ESM-2 and ESMFold tasks run on GPU. Both fit comfortably on a T4 (16GB) for the default protein set (sequences up to 238 residues). For longer proteins:

| GPU | VRAM | Max sequence length (ESMFold) |
|-----|------|-------------------------------|
| T4 | 16 GB | ~400 residues |
| L4 / A10 | 24 GB | ~600 residues |
| A100 | 40-80 GB | ~1000+ residues |

ESMFold memory usage scales roughly quadratically with sequence length due to the attention mechanism. The `max_length` parameter (default 400) skips proteins longer than the threshold.

## Why Flyte / Union?

Even a simple bioinformatics pipeline like this benefits from orchestration:

- **Caching** — sequence loading is cached. Change an analysis parameter and only downstream tasks re-run. In real pipelines with expensive structure predictions, this saves hours of GPU time.
- **Reproducibility** — every run is versioned. When a collaborator asks "what parameters did you use for that analysis?", you can point them to the exact execution.
- **Reports** — results render directly in the Flyte UI. No Jupyter notebooks to share, no screenshots to paste into Slack. The 3D protein structures are interactive right in the browser.
- **Resource isolation** — biophysical analysis runs on CPU, ESM-2 and ESMFold run on GPU. Each task gets exactly the resources it needs — you don't pay for a GPU while computing sequence alignments.
- **Scale** — this tutorial runs 7 proteins. The same pipeline handles hundreds — ESMFold predictions parallelize across a GPU cluster, and caching means you never re-fold a sequence you've already seen.
