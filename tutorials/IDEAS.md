# Future Fine-Tuning Tutorials

Ideas for follow-up tutorials using this pipeline as a template.

## Next Models to Try

| Model | Params | Why | LoRA targets |
|-------|--------|-----|-------------|
| Qwen/Qwen2.5-0.5B | 500M | Drop-in swap, big quality jump over SmolLM2 | Same as default |
| Qwen/Qwen3-0.6B | 600M | Built-in thinking/reasoning | Same as default |
| google/gemma-4-1b | 1B | Latest from Google, multimodal-ready | Check modules |
| google/gemma-3-1b-pt | 1B | Strong small model from Google | Check modules |
| meta-llama/Llama-3.2-1B | 1B | Most popular family, gated (shows HF_TOKEN) | Same as default |
| microsoft/Phi-4-mini | 3.8B | Punches above its weight, QLoRA actually needed on T4 | `q_proj`, `k_proj`, `v_proj`, `dense` |
| Qwen/Qwen2.5-1.5B | 1.5B | Where QLoRA starts to matter | Same as default |

## Datasets to Pair

| Dataset | Task | Good pairing |
|---------|------|-------------|
| `tatsu-lab/alpaca` | General instruction following | Any model, classic benchmark |
| `sahil2801/CodeAlpaca-20k` | Code generation | Qwen2.5-0.5B — "SQL but for Python" |
| `FinGPT/fingpt-sentiment-train` | Financial sentiment | Phi-4-mini — real business domain, QLoRA needed |
| `medalpaca/medical_meadow_medical_flashcards` | Medical Q&A | Gemma-4 — domain adaptation, biotech angle |
| `OpenAssistant/oasst2` | Chat/dialogue | Multi-turn chatbot fine-tuning |
| `iamtarun/python_code_instructions_18k` | Python from docstrings | Clean format, easy to evaluate |
| `gsm8k` | Math reasoning | Shows reasoning improvements |
| `argilla/distilabel-capybara-dpo-7k-binarized` | DPO/preference | Different training paradigm |

## Suggested Workshop Series

1. **This tutorial** — SmolLM2 + SQL (LoRA basics, all three methods compared)
2. **Qwen2.5-0.5B + CodeAlpaca** — code generation, same pipeline template
3. **Phi-4-mini + FinGPT sentiment** — QLoRA actually needed, real business domain
4. **Gemma-4 + medical flashcards** — domain adaptation, biotech angle

Each reuses the same pipeline — just swap `--model_name` and `--dataset_name` (with a tweaked `format_example` in workflow.py).

## Carbon Genomic Foundation Model Tutorials

HuggingFace released [Carbon](https://github.com/huggingface/carbon) (May 2026) — an open-source family of autoregressive genomic foundation models (500M, 3B, 8B) trained on 1T tokens of DNA. 275x faster than Evo2. Hybrid tokenizer (6-mer + single-base resolution).

| Tutorial Idea | What it does | Visuals |
|--------------|-------------|---------|
| **Variant Effect Prediction** (done) | Score clinically relevant mutations (BRCA2, TP53, CFTR) as pathogenic vs benign using Carbon's zero-shot log-likelihood scoring | Mutation impact heatmaps, gene diagrams, pathogenicity classification charts |
| **DNA Sequence Generation & Analysis** (done) | Generate DNA with Carbon, then analyze GC content, codon usage, open reading frames, compare to real genes | Codon frequency wheels, ORF track diagrams, GC content sparklines, generated vs real sequence comparisons |
| **Gene Comparison Across Species + DNA-to-Structure** (done) | Embed/score homologous genes across species with Carbon, visualize evolutionary relationships, then translate to protein and fold both ref and mutant with ESMFold to show structural impact of cross-species divergence | Phylogenetic dendrograms, cross-species similarity heatmaps, embedding UMAP projections, side-by-side 3D protein structures with pLDDT coloring |

## Healthcare AI Tutorials

Ideas for healthcare/clinical AI demos with strong visual reports.

### Medical Imaging

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **Chest X-Ray Triage** | CheXpert or NIH ChestX-ray14 (open) | GradCAM heatmap overlays on X-rays, multi-label classification report, per-pathology ROC curves |
| **Retinal Disease Detection** | APTOS 2019 or Messidor (diabetic retinopathy grading) | Fundus images with severity grading (0-4), confusion matrices, confidence calibration plots |
| **Skin Lesion Classification** | HAM10000 (dermoscopy) | Lesion images with predicted vs actual labels, melanoma vs benign ROC, GradCAM attention maps |

### Clinical NLP

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **Medical Note Summarization** | MIMIC discharge summaries (or synthetic) | Side-by-side original vs structured extraction (diagnoses, meds, procedures), entity highlighting |
| **Adverse Drug Event Detection** | ADE corpus or i2b2 | NER entity spans highlighted on clinical text, drug-event relation graphs |

### Time Series / Signals

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **ECG Arrhythmia Detection** | PTB-XL (21k 12-lead ECGs, open) | 12-lead waveform SVGs, rhythm classification confusion matrix, per-lead attention heatmaps |
| **ICU Patient Deterioration** | MIMIC vitals time series | Sparkline dashboards of vitals, real-time risk score evolution, alert threshold visualizations |

### Drug & Molecular

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **Drug-Drug Interaction Prediction** | DrugBank or TWOSIDES | Molecule pair cards, interaction severity heatmaps, drug interaction network graphs |
| **Drug Repurposing with Embeddings** | DrugBank + disease ontologies | UMAP projections colored by therapeutic area, nearest-neighbor candidate tables, embedding similarity heatmaps |

### Computer Vision — Instance Segmentation

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **Nuclei Segmentation (Mask R-CNN)** | Data Science Bowl 2018 (~670 microscopy images, Kaggle) | Colorful per-nucleus instance masks overlaid on microscopy, mask confidence scores, cell count distributions, before/after segmentation comparisons |

### 3D Medical Imaging

| Tutorial Idea | Dataset | Visuals |
|--------------|---------|---------|
| **Brain Tumor Segmentation (3D MRI)** | BraTS 2023 (multi-modal MRI volumes, open) | 3D tumor volume renderings, axial/sagittal/coronal slice views with mask overlays, tumor subregion breakdowns (enhancing, necrotic, edema), volumetric statistics |
| **3D Organ Segmentation (CT)** | TotalSegmentator (1.2k CT scans, 117 anatomical structures) | 3D organ renderings, slice-by-slice segmentation animations, multi-organ Dice score comparisons |

### Build Queue

1. **Nuclei Segmentation (Mask R-CNN)** — Instance segmentation showpiece, healthcare + biotech crossover
2. **Brain Tumor Segmentation (3D MRI)** — Volumetric data, 3D renderings are visually stunning
3. **Chest X-Ray Triage** — Universally recognized, GradCAM heatmaps
4. **ECG Arrhythmia Detection** — Different modality (time-series), everyone knows an ECG

### Other Strong Picks

- **Retinal Disease Detection** — Fundus images are visually striking, clear grading scale, real screening use case

## Notes

- Phi-4-mini is the best candidate to demonstrate QLoRA's real value (3.8B won't fit full fine-tune on a T4)
- For gated models (LLaMA), need `HF_TOKEN` — good to show that flow
- Different architectures may need different LoRA target modules — see README for the lookup snippet
