# Gene Comparison Across Species

Compare how the same gene evolves across species — from DNA sequence to 3D protein structure — using HuggingFace [Carbon](https://github.com/huggingface/carbon) and [ESMFold](https://github.com/facebookresearch/esm), orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

Every species on Earth shares a common ancestor, and we carry the proof in our DNA. The insulin gene in a human and a zebrafish diverged ~450 million years ago, but they're still recognizably the same gene — because insulin is so important that evolution can't change it much without killing the organism.

This tutorial explores that principle: take a single gene, compare it across species, and see what stays the same vs what changes. The punchline is that **protein structure is more conserved than sequence** — even when DNA diverges significantly, the 3D fold often stays remarkably similar because it's the fold, not the exact sequence, that matters for function.

## What This Pipeline Does

| Stage | What it answers | Biology context |
|-------|----------------|-----------------|
| **Load Genes** | What species are we comparing? | Curated homologous coding sequences from 6 species per gene set. |
| **Carbon Scoring** | How does each species' gene look to a DNA language model? | Carbon log-likelihood scores reveal whether some species' sequences look more "natural" to the model than others. Also computes pairwise DNA and protein identity matrices. |
| **ESMFold Structures** | Do the proteins fold the same way? | Translates each gene to protein and folds with ESMFold. Interactive 3D viewers let you visually compare structures side-by-side. |
| **Generate Summary** | What's the evolutionary story? | Phylogenetic trees from both DNA and protein identity, conservation analysis, evolutionary insights. |

## Gene Sets

Three curated gene sets span different evolutionary pressures:

### Insulin (default)
| Species | Common Name | Why it's interesting |
|---------|------------|---------------------|
| **Human** | Homo sapiens | Reference — the insulin we all know |
| **Mouse** | Mus musculus | Diverged ~90M years ago. Mouse insulin works in humans (and vice versa) |
| **Chicken** | Gallus gallus | Diverged ~310M years ago. Bird insulin is slightly different but still functional |
| **Zebrafish** | Danio rerio | Diverged ~450M years ago. Fish insulin can still lower blood sugar in mammals |
| **Frog** | Xenopus laevis | Classic model organism for developmental biology |
| **Cow** | Bos taurus | Bovine insulin was used to treat diabetics before recombinant human insulin |

### Hemoglobin Beta
The most studied gene in molecular evolution. Carries oxygen. Mutations cause sickle cell disease (humans), and sequence differences power the "molecular clock" hypothesis.

### p53 (Tumor Suppressor)
Guardian of the genome. Includes **elephant** — which has 20 copies of p53 (humans have 1), potentially explaining their extremely low cancer rates despite massive body size (Peto's paradox).

## Key Concepts

**Homologous Genes** — Genes in different species that evolved from a shared ancestor. Orthologs (separated by speciation) vs paralogs (separated by gene duplication). This tutorial focuses on orthologs.

**Sequence Identity** — Percentage of positions that match when two sequences are aligned. DNA identity is always lower than protein identity for coding genes because of synonymous (silent) mutations — changes in the DNA that don't change the amino acid.

**Synonymous vs Non-synonymous Mutations** — The genetic code is redundant: multiple codons encode the same amino acid. A mutation that changes the DNA but not the protein (synonymous) accumulates freely. A mutation that changes the protein (non-synonymous) faces natural selection. This is why protein is more conserved than DNA.

**Phylogenetic Trees** — Diagrams showing evolutionary relationships. Species that diverged recently (human/mouse) cluster together; those that diverged long ago (human/zebrafish) are further apart. We build trees from both DNA and protein identity using UPGMA clustering.

**pLDDT (predicted Local Distance Difference Test)** — ESMFold's per-residue confidence score. >90 = very high confidence (structured core), 70-90 = confident, 50-70 = low (flexible loops), <50 = likely disordered.

**Structure Conservation** — Even when sequence identity drops to 30-40%, proteins often maintain the same 3D fold. This is because a protein's function depends on its shape, and evolution selects for shape. Comparing structures across species reveals the "essential scaffold" of a protein.

## Reports

| Report | What you'll see |
|--------|----------------|
| **Score & Compare** | Per-species table (length, GC%, Carbon LL), DNA identity heatmap, protein identity heatmap, phylogenetic dendrogram, Carbon score bar chart |
| **Fold Proteins** | Side-by-side interactive 3D protein structures (spinning, color-coded), pLDDT sparklines per species, confidence comparison chart |
| **Generate Summary** | Cross-species stats, evolutionary insight, GC content comparison, DNA + protein phylogenetic trees, method notes |
| **Pipeline** | Overview: avg DNA/protein identity, avg pLDDT, species list |

## Setup

```bash
cd tutorials/genomic-gene-comparison

# Create environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Default (Insulin across 6 species)
flyte run --local --tui workflow.py pipeline

# Hemoglobin comparison
flyte run workflow.py pipeline --gene_set "hemoglobin"

# p53 — includes elephant!
flyte run --local --tui workflow.py pipeline --gene_set "p53"

# Remote (GPU — recommended for ESMFold)
flyte run workflow.py pipeline

# Remote with different gene set
flyte run workflow.py pipeline --gene_set "hemoglobin"
```

## How It Works

```
Homologous Genes ──> Carbon-3B ──> Pairwise Identity ──> ESMFold ──> Summary
  (6 species)           │            Matrices              │           │
                        │  Log-likelihood    DNA ──> Tree   │  3D structures
                        │  per species    Protein ──> Tree  │  per species
                        ▼                                   ▼           ▼
                  "Which species'          "How do these   Side-by-side
                   DNA looks most           genes relate   3D viewers +
                   natural?"                 to each       evolutionary
                                            other?"        analysis
```

## Architecture

Same pattern as [genomic-variant-effect](../genomic-variant-effect/) and [genomic-dna-generation](../genomic-dna-generation/):

- **config.py** — Flyte image + environments (GPU for Carbon + ESMFold, CPU for analysis)
- **workflow.py** — Tasks, SVG visualizations, 3D viewers, and pipeline orchestrator
- 32Gi GPU memory to fit both Carbon-3B and ESMFold
- Results passed as JSON strings between tasks
