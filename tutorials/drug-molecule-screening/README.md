# Drug Molecule Screening

Virtual drug screening pipeline built with [Flyte](https://flyte.org/) and [RDKit](https://www.rdkit.org/). Screens a library of drug molecules for drug-likeness using physicochemical property analysis, Lipinski's Rule of Five, and target profile matching.

## Why Virtual Screening Matters

Bringing a new drug to market takes 10-15 years and costs over $1 billion. Most candidates fail — roughly 90% of drugs that enter clinical trials never get approved. The earlier you can filter out bad candidates, the more time and money you save.

Virtual screening is the computational first pass. Before you ever synthesize a molecule or run a lab assay, you can ask: does this molecule have the right physical and chemical properties to be a viable drug? Can it dissolve in blood? Can it cross a cell membrane? Is it small enough to reach its target? These aren't guarantees — but they're strong filters that eliminate obviously poor candidates before expensive experiments begin.

This pipeline automates that first pass: take a library of molecules, compute their physicochemical properties, check them against established drug-likeness rules, and rank the survivors.

## What This Pipeline Does

| Stage | What it answers | Chemistry context |
|-------|----------------|-------------------|
| **Load Molecules** | Are these valid chemical structures? | Molecules are represented as SMILES strings — a line notation that encodes atoms, bonds, and connectivity (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O` is aspirin). RDKit parses these into molecular graphs and generates 2D structural depictions. |
| **Compute Properties** | What are the key physicochemical descriptors? | Computes molecular weight, lipophilicity (LogP), hydrogen bond donors/acceptors, topological polar surface area (TPSA), and QED drug-likeness score. These determine whether a molecule can be absorbed, distributed, and eventually cleared by the body. |
| **Screen Candidates** | Which molecules fit the target drug profile? | Scores each molecule against configurable property ranges, checks Lipinski's Rule of Five, and computes pairwise Tanimoto similarity using Morgan fingerprints to assess chemical diversity. |
| **Generate Report** | What's the verdict? | Executive summary with top candidates, property distributions, chemical diversity analysis, and spotlight cards for the best-scoring molecules. |

## The Default Molecules

The pipeline ships with 15 real, well-known drugs chosen to show chemical diversity:

| Molecule | What it is | Why it's interesting |
|----------|-----------|---------------------|
| **Aspirin** | Pain reliever / anti-inflammatory | One of the oldest drugs still in use (1899). Small, simple structure — a classic "drug-like" molecule. |
| **Ibuprofen** | NSAID pain reliever | Over-the-counter staple. Good example of a molecule that easily passes Lipinski's rules. |
| **Caffeine** | Stimulant | Not technically a drug in the therapeutic sense, but has clear pharmacological activity. Very small MW. |
| **Penicillin G** | Antibiotic | The molecule that launched the antibiotic era. Contains a beta-lactam ring — the key to its mechanism. |
| **Metformin** | Diabetes medication | First-line treatment for type 2 diabetes. Unusually small and hydrophilic for an oral drug. |
| **Paracetamol** | Pain reliever (acetaminophen) | One of the most widely used drugs worldwide. Very simple structure. |
| **Diazepam** | Anxiolytic (Valium) | Benzodiazepine — acts on GABA receptors in the brain. Contains a 7-membered ring fused to a benzene ring. |
| **Omeprazole** | Proton pump inhibitor | Treats acid reflux. Contains a sulfoxide group critical to its mechanism. |
| **Atorvastatin** | Cholesterol lowering (Lipitor) | Best-selling drug in history. Larger molecule — pushes the Lipinski boundaries. |
| **Methotrexate** | Cancer / autoimmune therapy | Folic acid antagonist. Fails some Lipinski criteria — a reminder that the rules aren't absolute. |
| **Doxorubicin** | Cancer chemotherapy | Anthracycline antibiotic. Large, complex, multiple ring systems — clearly outside "drug-like" space by simple rules, but clinically essential. |
| **Tamoxifen** | Breast cancer treatment | Selective estrogen receptor modulator. Triphenylethylene scaffold. |
| **Lopinavir** | HIV protease inhibitor | Large peptidomimetic — designed to fit the active site of HIV protease. Breaks Lipinski's MW rule. |
| **Remdesivir** | Antiviral (COVID-19) | Nucleotide analog prodrug. Complex phosphoramidate structure. |
| **Erlotinib** | Cancer therapy (EGFR inhibitor) | Targeted kinase inhibitor — designed using structural biology to fit the ATP binding pocket. |

## Key Concepts Explained

**SMILES (Simplified Molecular-Input Line-Entry System)** — A compact text notation for chemical structures. Atoms are letters (C, N, O, S), bonds are implicit or explicit (`=` for double, `#` for triple), and rings are denoted by matching digits. Parentheses create branches. For example, `CC(=O)O` is acetic acid (a carbon, connected to a methyl group, a double-bonded oxygen, and a hydroxyl). SMILES lets you represent any organic molecule as a string — which is why it's the standard input format for cheminformatics tools.

**Molecular Weight (MW)** — The mass of a molecule in Daltons (Da). Most oral drugs fall between 150-500 Da. Too heavy and a molecule struggles to cross cell membranes by passive diffusion; too light and it may not have enough molecular surface to bind its target specifically.

**LogP (Partition Coefficient)** — Measures lipophilicity: how much a molecule prefers oil over water. Calculated as the log of the ratio of concentrations in octanol vs water. A LogP of 0 means equal preference; positive values mean lipophilic (fat-loving), negative means hydrophilic (water-loving). Oral drugs typically need LogP between -0.5 and 5 — enough lipophilicity to cross membranes, but not so much that they get stuck in fat tissue and never reach the bloodstream.

**Hydrogen Bond Donors (HBD) and Acceptors (HBA)** — HBDs are NH and OH groups that can donate a hydrogen to a hydrogen bond. HBAs are electronegative atoms (N, O) that can accept one. These interactions are critical for binding drug targets (proteins), but too many make a molecule too polar to cross the lipid bilayer of cell membranes.

**TPSA (Topological Polar Surface Area)** — The surface area of a molecule occupied by nitrogen, oxygen, and their attached hydrogens. TPSA correlates with membrane permeability: molecules with TPSA < 140 square Angstroms tend to have good oral absorption; those above often can't cross the gut lining.

**Lipinski's Rule of Five** — The most famous drug-likeness filter, published by Christopher Lipinski at Pfizer in 1997. A molecule is likely to be orally bioavailable if it satisfies: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10. The "five" comes from the multiples of 5 in the cutoffs. About 90% of approved oral drugs pass these rules — but plenty of important drugs (antibiotics, antivirals, cancer drugs) break them, which is why the rules are a guide, not a gate.

**QED (Quantitative Estimate of Drug-likeness)** — A score from 0 to 1 that combines eight molecular properties (MW, LogP, HBD, HBA, TPSA, rotatable bonds, aromatic rings, and structural alerts) into a single drug-likeness measure. Unlike Lipinski's binary pass/fail, QED captures how "drug-like" a molecule is on a continuous scale. A QED above 0.5 is generally considered favorable. Developed by Bickerton et al. (2012) by analyzing what property distributions actual approved drugs have.

**Morgan Fingerprints** — A molecular fingerprint is a fixed-length bit vector that encodes the substructural features present in a molecule. Morgan fingerprints (also called circular or ECFP fingerprints) work by examining each atom's neighborhood out to a given radius and hashing those environments into bit positions. Two molecules with similar fingerprints have similar local chemical environments — similar functional groups, ring systems, and connectivity patterns.

**Tanimoto Similarity** — The standard way to compare molecular fingerprints. For two bit vectors A and B, Tanimoto = (bits in common) / (bits in A + bits in B - bits in common). Ranges from 0 (completely different) to 1 (identical). In drug discovery, molecules with Tanimoto > 0.85 are generally considered highly similar; < 0.3 means they're chemically diverse. The similarity matrix in this pipeline tells you whether your candidate library covers different regions of chemical space or is clustered around similar scaffolds.

## Reports

Each task generates a Flyte report with rich visualizations:

- Molecule structure galleries (2D depictions from RDKit)
- Molecular weight distribution (horizontal bar chart, sorted)
- LogP vs MW scatter plot with Lipinski boundaries and drug-like zone shading
- Normalized property heatmap (molecules x physicochemical descriptors)
- Lipinski Rule of Five compliance table (per-rule pass/fail badges)
- QED drug-likeness scores (horizontal bar chart, sorted)
- Screening funnel (total -> MW filter -> LogP filter -> Lipinski -> all criteria)
- Ranked candidate table with composite scores
- Top 5 candidate cards with structure depictions and full property breakdowns
- Pairwise Tanimoto similarity heatmap (Morgan fingerprints)
- Box-plot property distributions (MW, LogP, TPSA, QED)
- Chemical diversity summary statistics
- Executive summary with top candidate recommendation

## What's Here

| File | What it does |
|------|-------------|
| `config.py` | Flyte environment — CPU image with RDKit and X11 rendering libraries |
| `workflow.py` | 4-stage pipeline: load molecules -> compute properties -> screen candidates -> generate report |

## Setup

```bash
cd tutorials/drug-molecule-screening

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Default — screen 15 well-known drugs
flyte run --local --tui workflow.py pipeline

# Remote execution
flyte run workflow.py pipeline

# Custom target profile (stricter MW and LogP filters)
flyte run --local --tui workflow.py pipeline \
    --target_profile '{"mw": [100, 400], "logp": [-0.5, 4.0]}'

# Custom molecules via JSON
flyte run --local --tui workflow.py pipeline \
    --molecules_json '{"Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O", "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}'
```

## A Note on Scope

This tutorial is designed for educational purposes — it teaches real cheminformatics concepts (SMILES, Lipinski's rules, molecular fingerprints, drug-likeness scoring) and real Flyte patterns (task decomposition, caching, reports) using a curated set of 15 well-known drugs. In real-world drug discovery, virtual screening campaigns operate on libraries of hundreds of thousands to millions of compounds sourced from databases like ZINC or ChEMBL, incorporate molecular docking against 3D protein targets, ADMET (absorption, distribution, metabolism, excretion, toxicity) prediction, and ML-based scoring models. The pipeline architecture here — load, compute, screen, report — is the same structure those production pipelines follow, just at a smaller scale.

## Why Flyte / Union?

Drug screening pipelines benefit from orchestration even at tutorial scale:

- **Caching** — molecule loading is cached. Change your target profile and only the screening and report tasks re-run. In real pipelines with thousands of molecules and expensive docking simulations, this saves hours.
- **Reproducibility** — every run is versioned with its exact inputs (molecule library, target profile). When a collaborator asks "which filters did you use?", you can point them to the execution.
- **Reports** — results render directly in the Flyte UI. No Jupyter notebooks to share, no screenshots. Molecule structures, charts, and tables are all interactive in the browser.
- **Resource isolation** — this tutorial uses CPU, but a real screening pipeline might run molecular docking on GPU. Flyte lets each task request exactly the resources it needs.
- **Scale** — this tutorial screens 15 molecules. The same pipeline handles thousands — property computation and fingerprint generation parallelize, and caching means you never recompute a molecule you've already seen.
