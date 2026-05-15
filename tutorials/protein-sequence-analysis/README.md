# Protein Sequence Analysis

Analyze, compare, and visualize biophysical properties of real protein sequences — all orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

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

**Protein Language Models (ESM-2)** — Just as GPT learns language by reading billions of sentences, ESM-2 learns protein "grammar" from 250 million sequences. It picks up patterns that evolution has conserved: which residues tend to co-occur, which positions are flexible vs constrained, what a "typical" protein looks like. The result is an embedding — a vector representation — for each protein that encodes structural and functional information far beyond what raw sequence comparison can capture. This is the same technology that powered breakthroughs like ESMFold (Meta's fast structure predictor) and is now standard in protein engineering labs.

**Contact Maps** — A contact map predicts which residue pairs are physically close in 3D space (typically <8 Angstroms). You can extract approximate contact maps from ESM-2's attention weights — the model has learned which residues "pay attention" to each other, and this correlates with spatial proximity. Contact maps look like symmetric heatmaps with patterns along the diagonal (local structure) and off-diagonal spots (long-range contacts that define the fold).

## Reports

Each task generates a Flyte report with SVG visualizations:

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
- Per-protein cards with hydrophobicity sparklines and structure tracks

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environments — CPU tasks with image and resource config |
| `workflow.py` | Pipeline: load sequences, analyze properties, compute similarity, generate summary |

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

# Remote execution (GPU for ESM-2)
flyte run workflow.py pipeline

# Swap ESM-2 model (larger = more accurate but slower)
flyte run workflow.py pipeline --esm_model "facebook/esm2_t12_35M_UR50D"

# Custom sequences via JSON
flyte run --local --tui workflow.py pipeline \
  --sequences_json '{"MyProtein": "MVLSPADKTNVKAAWGKVGAH", "Another": "MQIFVKTLTGKTITLEVEPS"}'

# Run individual tasks
flyte run --local --tui workflow.py load_sequences
flyte run --local --tui workflow.py analyze_properties --seq_dir /path/to/sequences
```

## Why Flyte / Union?

Even a simple bioinformatics pipeline like this benefits from orchestration:

- **Caching** — sequence loading is cached. Change an analysis parameter and only downstream tasks re-run. In real pipelines with expensive structure predictions, this saves hours of GPU time.
- **Reproducibility** — every run is versioned. When a collaborator asks "what parameters did you use for that analysis?", you can point them to the exact execution.
- **Reports** — results render directly in the Flyte UI. No Jupyter notebooks to share, no screenshots to paste into Slack.
- **Resource isolation** — biophysical analysis runs on CPU, ESM-2 inference runs on GPU. Each task gets exactly the resources it needs.
- **Scale** — ESM-2 runs on a single GPU here. Swap in a larger model (ESM-2 650M) or add structure prediction (ESMFold, AlphaFold) and the same pipeline scales to a cluster.
