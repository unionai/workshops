# Genomic Variant Effect Prediction

Score clinically relevant DNA mutations using HuggingFace's [Carbon](https://github.com/huggingface/carbon) genomic foundation model — all orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why Variant Effect Prediction Matters

Every human carries ~4-5 million genetic variants compared to the reference genome. Most are harmless, but some cause disease — sickle cell anemia from a single base change in hemoglobin, cancer from mutations in TP53 or BRCA2, cystic fibrosis from defects in the CFTR chloride channel. The challenge is figuring out which variants matter.

Traditional approaches (SIFT, PolyPhen, CADD) use conservation, protein structure, and hand-crafted features. Genomic language models like Carbon take a different approach: they learn the "grammar" of DNA from billions of base pairs, then score variants by how surprising they look to the model. A mutation that dramatically changes the model's likelihood score is probably disrupting something important.

## What Is Carbon?

[Carbon](https://github.com/huggingface/carbon) is HuggingFace's open-source family of autoregressive genomic foundation models, released May 2026:

| Model | Parameters | Context | Key Feature |
|-------|-----------|---------|-------------|
| Carbon-500M | 500M | 131kbp | Fast draft model, speculative decoding |
| Carbon-3B | 3B | 131kbp | **Flagship** — matches/beats Evo2 7B |
| Carbon-8B | 8B | 786kbp | Largest, best for long-range effects |

Trained on **1 trillion tokens (~6 trillion DNA base pairs)** from eukaryotic genes, mRNA transcripts, and prokaryotic genomes. Uses a hybrid tokenizer that splits DNA into 6-mer chunks while maintaining single-base resolution — making it **275x faster** than Evo2 at comparable size.

## What This Pipeline Does

| Stage | What it answers | Biology context |
|-------|----------------|-----------------|
| **Load Variants** | Which genes and mutations are we analyzing? | We curate clinically relevant variants with known pathogenicity labels from ClinVar-style annotations. |
| **Carbon Scoring** | How does the DNA language model rate each mutation? | For each variant, compute log P(mutant) - log P(reference). Negative = model thinks the mutation is unlikely = potentially damaging. |
| **Analyze Effects** | Does the model's intuition match clinical reality? | Compare Carbon's zero-shot scores against known pathogenic/benign labels. Compute accuracy, precision, recall. |
| **Generate Summary** | What's the big picture? | Ranked variant tables, per-gene summaries, cross-gene comparisons, confusion matrices. |

## The Default Genes

The pipeline ships with 5 clinically important genes chosen to span different disease mechanisms:

| Gene | Disease | Why it's interesting |
|------|---------|---------------------|
| **BRCA2** | Breast/ovarian cancer | Tumor suppressor for DNA repair. BRCA2 mutations increase breast cancer risk to 45-85%. |
| **TP53** | >50% of all cancers | "Guardian of the genome." Most commonly mutated gene in human cancer. Hotspot mutations like R175H are gain-of-function. |
| **CFTR** | Cystic fibrosis | Chloride channel. Most common lethal genetic disease in Europeans. deltaF508 deletion causes ~70% of cases. |
| **KRAS** | Pancreatic, colorectal, lung cancer | Oncogenic GTPase switch. KRAS G12D is the single most common oncogenic mutation across all cancer types. |
| **HBB** | Sickle cell disease | Beta-globin. The E6V (HbS) mutation is the most famous single-nucleotide disease variant in history. |

## Key Concepts Explained

**Log-Likelihood Ratio (LLR)** — The core of zero-shot VEP. We compute how likely the model thinks a DNA sequence is with and without the mutation. `score = log P(mutant) - log P(reference)`. Negative means the mutant is less likely — the model learned that this position "expects" the reference base, suggesting the mutation disrupts something.

**Zero-Shot Prediction** — No fine-tuning on labeled variant data. Carbon's understanding comes purely from learning DNA sequence patterns during pretraining. This is powerful because it generalizes to any gene without needing gene-specific training data.

**Pathogenic vs Benign** — In clinical genetics, "pathogenic" means a variant causes disease. "Benign" means it doesn't. "VUS" (variant of uncertain significance) is the frustrating middle ground — clinically detected but not enough evidence to classify. Better computational tools help resolve VUS cases.

**Variant Nomenclature** — `c.37A>T` means "at coding position 37, the reference A is changed to T." Protein-level names like `R175H` mean "arginine at position 175 is changed to histidine." `G12V` means "glycine 12 to valine."

**Hotspot Mutations** — Some positions mutate far more often than chance would predict. TP53 R175H, R248W, and R273H together account for ~15% of all TP53 mutations in cancer. These "hotspots" are where mutations give cancer cells the biggest growth advantage.

## Reports

Each task generates a Flyte report with rich visualizations:

| Report | What you'll see |
|--------|----------------|
| **Score Variants** | Per-gene tables with VEP scores, ref/alt bases, clinical annotations |
| **Analyze Effects** | DNA sequence tracks with highlighted variants, lollipop plots, score bar charts, variant detail cards, confusion matrix, score distributions by effect class |
| **Generate Summary** | Cross-gene metrics heatmap, all variants ranked by impact, method description |
| **Pipeline** | Overview stats with accuracy, precision, recall |

## Setup

```bash
cd tutorials/genomic-variant-effect

# Create environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Local (CPU — slower but works)
flyte run --local --tui workflow.py pipeline

# Local with smaller model (faster)
flyte run --local --tui workflow.py pipeline --model_name "HuggingFaceBio/Carbon-500M"

# Remote (GPU — recommended)
flyte run workflow.py pipeline

# Remote with largest model
flyte run workflow.py pipeline --model_name "HuggingFaceBio/Carbon-8B"

# Custom variants via JSON
flyte run --local --tui workflow.py pipeline --variants_json '{"MyGene": {"description": "...", "sequence": "ATGCCC...", "variants": [{"pos": 5, "ref": "C", "alt": "T", "name": "C5T", "known_effect": "pathogenic", "clinical": "..."}]}}'
```

## How It Works

```
Gene Variants ──> Carbon-3B ──> Log-Likelihood Scores ──> Analysis & Visualization
                    │                                           │
                    │  For each variant:                        │  Compare predictions
                    │  score = logP(mut) - logP(ref)           │  vs known labels
                    │  negative = likely damaging               │
                    ▼                                           ▼
              Per-variant scores                    Accuracy, confusion matrix,
              with confidence                       ranked variant tables
```

## Architecture

The pipeline follows the same pattern as the [protein-sequence-analysis](../protein-sequence-analysis/) tutorial:

- **config.py** — Flyte image + environment definitions (GPU for Carbon inference, CPU for analysis)
- **workflow.py** — All tasks and pipeline orchestrator with embedded SVG visualizations
- Tasks use `flyte.report` for live progress updates and rich HTML reports
- Results are passed as JSON strings between tasks
