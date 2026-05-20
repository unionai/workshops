# DNA Sequence Generation & Analysis

Generate DNA with HuggingFace's [Carbon](https://github.com/huggingface/carbon) genomic foundation model, then analyze how well the generated sequences match real biology — all orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

If a language model truly understands DNA, it should be able to generate sequences that look biologically real — matching the statistical signatures of actual genes. This tutorial puts Carbon to the test: given the first ~60 bases of a real gene, can it produce a plausible continuation?

We compare generated DNA against reality across multiple dimensions: base composition, GC content profiles, codon usage patterns, open reading frame structure, and dinucleotide frequencies. These are the same metrics a bioinformatician would use to evaluate synthetic gene design.

## What Is Carbon?

[Carbon](https://github.com/huggingface/carbon) is HuggingFace's open-source family of autoregressive genomic foundation models, released May 2026:

| Model | Parameters | Speed | Best for |
|-------|-----------|-------|----------|
| Carbon-500M | 500M | Fastest | Quick experiments, speculative decoding |
| Carbon-3B | 3B | **Default** | Best quality/speed tradeoff |
| Carbon-8B | 8B | Slowest | Maximum quality, long-range patterns |

Trained on **1 trillion tokens (~6 trillion DNA base pairs)**. The hybrid 6-mer tokenizer makes it **275x faster** than Evo2 at comparable size.

## What This Pipeline Does

| Stage | What it answers | Biology context |
|-------|----------------|-----------------|
| **Load Prompts** | What genes are we using as seeds? | Real gene starts from human, jellyfish, E. coli, and SARS-CoV-2 — diverse organisms to test generalization. |
| **Carbon Generation** | Can Carbon continue a real gene plausibly? | Autoregressive generation: given 60bp of prompt, generate 200bp of new DNA. Like GPT completing a sentence, but for DNA. |
| **Analyze Sequences** | How does generated DNA compare to real? | Compare GC content, base composition, codon usage, ORFs, and dinucleotide frequencies side-by-side. |
| **Generate Summary** | Is Carbon good at this? | Cross-gene metrics, quality ratings, systematic bias detection. |

## The Default Gene Prompts

| Gene | Organism | Why it's interesting |
|------|----------|---------------------|
| **Insulin (INS)** | Human | One of the first genes cloned (1978). Encodes the blood sugar regulator. |
| **Hemoglobin Beta (HBB)** | Human | Carries oxygen. Mutations cause sickle cell disease. |
| **GFP** | Jellyfish | Glows green under UV. Revolutionized cell biology. Nobel Prize 2008. |
| **LacZ** | E. coli | The classic reporter gene. Foundation of blue-white screening in cloning. |
| **Spike Protein** | SARS-CoV-2 | Target of mRNA vaccines. Most studied viral protein in history. |
| **TP53** | Human | Guardian of the genome. Mutated in >50% of cancers. |

These span different organisms (human, jellyfish, bacteria, virus), GC content ranges, and gene functions — a tough test for any genomic model.

## Key Concepts Explained

**GC Content** — The fraction of bases that are G or C (vs A or T). Different organisms have characteristic GC levels: E. coli ~51%, human genes ~40-60%, AT-rich organisms as low as 20%. A good model should match the GC content of whatever organism's DNA it's continuing.

**Codon Usage** — DNA is read in triplets (codons), each encoding one amino acid. But there's redundancy: 6 different codons all encode Leucine. Different organisms prefer different codons for the same amino acid ("codon bias"). E. coli loves CTG for Leucine; humans prefer CTG too but use TTG more. A model that learns organism-specific codon preferences has learned something real about biology.

**Open Reading Frames (ORFs)** — A stretch of DNA from a start codon (ATG) to a stop codon (TAA/TAG/TGA) that could encode a protein. Real genes contain ORFs; random DNA has short ones by chance. If Carbon generates DNA with ORF patterns similar to real genes, it's learned protein-coding structure.

**Dinucleotide Frequencies** — How often each pair of adjacent bases appears (AA, AT, AG, AC, TA, ...). This is a deeper statistical fingerprint than base composition alone. The CpG dinucleotide is famously rare in vertebrate genomes (methylation depletes it) but common in bacteria. A model that captures these patterns has learned organism-level signatures.

**Temperature Sampling** — Controls randomness in generation. Temperature=0 always picks the most likely next base (deterministic). Temperature=1.0 samples proportionally from the probability distribution. We use 0.8 — slightly creative but still grounded.

## Reports

Each task generates a Flyte report with rich visualizations:

| Report | What you'll see |
|--------|----------------|
| **Generate DNA** | Color-coded DNA tracks showing prompt (dim) vs generated (bright) with boundary markers |
| **Analyze Sequences** | Side-by-side comparisons: base composition bar charts, GC content sparklines, codon usage donut wheels, ORF track diagrams, ORF tables |
| **Generate Summary** | Cross-gene GC comparison chart, per-gene quality ratings, dinucleotide frequency deviation heatmap |
| **Pipeline** | Overview stats: avg GC difference, how many genes matched within 5%, total ORFs found |

## Setup

```bash
cd tutorials/genomic-dna-generation

# Create environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Local (CPU — slow but works)
flyte run --local --tui workflow.py pipeline

# Local with smaller model
flyte run --local --tui workflow.py pipeline --model_name "HuggingFaceBio/Carbon-500M"

# Longer generation
flyte run --local --tui workflow.py pipeline --gen_length 500

# More creative generation
flyte run --local --tui workflow.py pipeline --temperature 1.0

# Remote (GPU — recommended)
flyte run workflow.py pipeline

# Remote with largest model
flyte run workflow.py pipeline --model_name "HuggingFaceBio/Carbon-8B"
```

## How It Works

```
Real Gene Prompts ──> Carbon-3B ──> Generated DNA ──> Analysis & Comparison
     (~60bp)            │             (~200bp)              │
                        │  Autoregressive                   │  GC content
                        │  sampling                         │  Codon usage
                        │  (temperature=0.8)                │  ORF structure
                        ▼                                   │  Dinucleotides
                  DNA continuations                         ▼
                  for each gene               Side-by-side reports
                                              Generated vs Real
```

## Architecture

Same pattern as [genomic-variant-effect](../genomic-variant-effect/) and [protein-sequence-analysis](../protein-sequence-analysis/):

- **config.py** — Flyte image + environments (GPU for Carbon, CPU for analysis)
- **workflow.py** — Tasks, SVG visualizations, and pipeline orchestrator
- Tasks use `flyte.report` for live progress and rich HTML reports
- Results passed as JSON strings between tasks
