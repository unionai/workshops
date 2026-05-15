# Drug Molecule Screening

Virtual drug screening pipeline built with [Flyte](https://flyte.org/) and [RDKit](https://www.rdkit.org/). Screens a library of drug molecules for drug-likeness using physicochemical property analysis, Lipinski's Rule of Five, and target profile matching.

## Pipeline Overview

| Task | Description |
|------|-------------|
| `load_molecules` | Parse SMILES strings, validate with RDKit, generate 2D molecular depictions |
| `compute_properties` | Compute MW, LogP, HBD, HBA, TPSA, QED, Lipinski compliance |
| `screen_candidates` | Score molecules against a target drug profile, compute Tanimoto similarity |
| `generate_report` | Produce a comprehensive visual report with ranked candidates |

## Setup

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Run locally with default molecule library (~15 well-known drugs)
flyte run --local --tui workflow.py pipeline

# Run remotely
flyte run workflow.py pipeline

# Custom target profile (stricter filters)
flyte run --local --tui workflow.py pipeline \
    --target_profile '{"mw": [100, 400], "logp": [-0.5, 4.0]}'
```

## Reports

Each task generates a rich visual report with:
- Molecule structure galleries (2D depictions)
- Molecular weight and LogP distribution charts
- LogP vs MW scatter plot with Lipinski boundaries
- Property heatmaps
- Screening funnel visualization
- Tanimoto similarity matrix
- Top candidate spotlight cards with full property breakdowns
- Box-plot property distributions
- Chemical diversity analysis
