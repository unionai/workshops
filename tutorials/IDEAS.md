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

## Vertical Demo Lineup

One hero demo per industry vertical, each a real runnable Flyte pipeline on a real open
dataset, ending in a Flyte Report good enough to embed on the vertical page or record as
a demo video. Reference implementations for the report style:
[brain-tumor-segmentation](brain-tumor-segmentation/) and
[genomic-gene-comparison](genomic-gene-comparison/).

Vertical pages live in `website-content/verticals/`. Several have an explicit
`TODO: tutorial repo` slot waiting to be filled by one of these.

### Coverage audit

| Vertical | Status | Existing tutorials |
|----------|--------|-------------------|
| Biotech & Life Sciences | Covered | genomic-{gene-comparison,variant-effect,dna-generation}, protein-sequence-analysis, drug-molecule-screening, cell-microscopy-classification, nuclei-segmentation, brain-tumor-segmentation |
| Frontier AI | Covered | llm-fine-tuning-{grpo,grpo-code,grpo-math,grpo-countdown,dpo,ppo,lora-qlora} |
| Computer Vision | Covered | detr-object-detection, lance-streaming-vision, brain-tumor-segmentation, nuclei-segmentation |
| Reinforcement Learning | Covered | openenv-{blackjack-rl,maze,snake}, rl-{mujoco,unitree-g1,unitree-go2,unitree-go2-mjx} |
| Agents | Covered | multi-agent-workflows, langgraph_agent_research, claude_agent_research, code-mode-analysis, autoresearch |
| Model training | Covered | bert-fine-tuning-{emotion,sentiment}, llm-fine-tuning-lora-qlora |
| Data Processing | Partial | lance-streaming-vision |
| Inference | Partial | gemma4-chat, detr-object-detection (app) |
| **Geospatial** | **Gap — building now** | — |
| **Media** | **Gap** | — |
| **Logistics** | **Gap** | — |
| **Autonomous Systems** | **Gap** | rl-unitree-* are locomotion, not AV perception |
| **Observability** | **Gap** | — |
| **Context Engineering** | **Gap** | code-mode-analysis is adjacent |
| **Financial Services** | **Weak** | fraud-detection-feast (tabular, no visual payoff) |

### 1. Geospatial — burn scar mapping (IN PROGRESS)

Wildfire burn scar segmentation on Harmonized Landsat–Sentinel-2 imagery with NASA/IBM's
Prithvi geospatial foundation model, plus a live STAC query → chip → segment → mosaic
fan-out over a real AOI.

- **Dataset:** `ibm-nasa-geospatial/hls_burn_scars` — 804 scenes, 512x512, 6 bands
  (Blue, Green, Red, NIR, SWIR1, SWIR2) as reflectance; masks `1`=burn / `0`=unburned /
  `-1`=nodata. Ground truth from MTBS shapefiles, CONUS 2018-2021. **CC-BY-4.0, ungated.**
  Train/val split already provided (2/3 · 1/3).
- **Model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` (base ViT encoder, ships a standalone
  `prithvi_mae.py`) + our own segmentation decoder. Apache-2.0, ungated.
  `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` is the official fine-tuned head and
  ships `splits/{train,val,test}.txt` matching the dataset exactly — useful as a baseline.
- **Live data:** Element84 STAC API (`https://earth-search.aws.element84.com/v1`) — confirmed
  public, no auth. Collections: `sentinel-2-l2a`, `sentinel-1-grd`, `landsat-c2-l2`, `naip`,
  `cop-dem-glo-30`.
- **Money shot:** mosaic assembling tile-by-tile while the Flyte UI fans out across hundreds
  of tile tasks (split-screen: DAG left, map right), then a before/after wipe in SWIR false
  color where burn scars glow neon orange-red against green vegetation. Per-tile IoU.
- **Why this one first:** the geospatial vertical page has an open `TODO: tutorial repo` for
  exactly this, and its code sample is already `query_scenes → chip_scene → segment_tile →
  mosaic_results`. Building to that skeleton turns the page's pseudocode into real code. It's
  also the one demo where the *orchestrator* is the star, not the model.
- **Traps:**
  - `load_dataset("ibm-nasa-geospatial/hls_burn_scars")` **fails** — script-based dataset
    (`hls_burn_scars.py` + 2.6 GB tar.gz), and `datasets>=3.0` dropped script support. Use
    `hf_hub_download` on the tarball and untar.
  - **Avoid `terratorch`** despite the model card's `library_name`. 68 dependencies
    (geopandas, rasterio, lightning, torchgeo, diffusers, pycocotools). Vendor
    `prithvi_mae.py` instead and keep the requirements list to ~6 packages.
  - STAC API needs full RFC3339 datetimes (`2021-06-01T00:00:00Z/...`); the `query`
    extension 400s — use CQL2 `filter` or sort client-side on `eo:cloud_cover`.
- **Video note:** use a large AOI (Dixie Fire 2021, N. California, ~390k ha, bbox roughly
  `[-121.45, 39.95, -121.05, 40.30]`) so the fan-out is visually dense, and re-render the
  mosaic progressively so a screen recording always has motion.

### 2. Autonomous Systems — multi-sensor perception replay (BLOCKED on a licence call)

3D detection on real driving logs: LiDAR + multi-camera.

- **Money shot:** the coolest *motion* of the whole lineup — 6-camera ring with 3D boxes
  tracking across frames, rotating LiDAR point cloud, BEV with boxes sliding along
  trajectories. Inherently video; nothing needs animating.
- **Research verdict — the original premise fails.** There is **no ungated HF mirror of
  nuScenes with actual sensor data**. nuScenes, Argoverse and Waymo have no HF org at all;
  every promising `nuscenes-mini` repo is an empty shell (`.gitattributes` + README, 0 GB).
  There are also **no pretrained 3D detectors on HF** — `mmdetection3d` returns zero
  results, and the only real weights are ONNX inside Autoware.
- **The two viable paths, and the tradeoff:**
  1. **KITTI** — `dpdl-benchmark/kitti` (ungated, 5.65 GB parquet, full 3D labels *and*
     `image/file_name`) + `Chaitanya89/KITTI`'s `data_object_calib.zip` (26.8 MB) joined on
     frame stem for the real `P2` projection. Complete and workable — but KITTI is
     **CC BY-NC-SA 3.0, non-commercial**. Both HF mirrors mislabel it `license: unknown`.
  2. **`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams`** — ungated and
     **CC BY 4.0, commercially clean**, with real LiDAR + 3D boxes sharded per clip
     (~359 MB lidar tar + 5.8 MB object info). Camera video is "coming soon", so it is
     LiDAR/BEV only for now — which is still the best motion visual.
- **Decision needed:** accept KITTI's non-commercial terms, or build LiDAR-only on the
  NVIDIA data. Held back rather than guessed, since ChaBuD and segformer-ADE were both
  rejected on exactly this basis.
- **Also:** don't promise a pretrained detector — render ground-truth boxes, optionally
  adding `depth-anything/Depth-Anything-V2-Small-hf` (24.8M, Apache-2.0) for colored
  pseudo-LiDAR from the mono camera.
- **Cost:** heaviest build — sensor calibration and camera-projection math.

### 3. Media — video to searchable index

Ingest video → shot boundary detection → keyframes → VLM captions + Whisper transcript →
embed → semantic search.

- **Money shot:** filmstrip contact sheet populating shot by shot, timeline ribbon filling,
  then a text query makes matching frames jump out.
- **To check:** Condensed Movies, MovieNet, VidChapters, AudioSet; TransNetV2 for shot
  boundaries; pyannote for diarization (likely gated — verify).
- **Flyte angle:** async fan-out per shot, actors for the VLM.

### 4. Logistics — demand forecast to route optimization

Per-SKU demand forecast feeding an OR-Tools VRP solve.

- **Money shot:** routes drawn on a real basemap, naive vs optimized side by side, cost delta.
  The most instantly legible "this is my industry" image of the set, and needs no GPU.
- **To check:** M5 Forecasting / Walmart, UCI Online Retail II, CVRPLIB/TSPLIB.
- **Alternative:** warehouse CV on the Amazon Bin Image Dataset (count items, flag manifest
  mismatch) — verify license and whether it's mirrored on HF.

### 5. Observability — catch the regression

Eval suite across model versions/prompts → LLM-judge scoring → per-slice regression detection.

- **Money shot:** model x capability heatmap, regression alert cards, trace waterfall.
- **To check:** `lmsys/lmsys-chat-1m` (gated?), HelpSteer, MT-Bench, RewardBench.

### 6. Context Engineering — what actually fits in the window

One question, five strategies: full dump / naive RAG / rerank / compress-summarize /
agentic search. Measure accuracy vs tokens vs latency.

- **Money shot:** accuracy-vs-tokens Pareto scatter, plus a context-window occupancy bar
  showing exactly what got included and what got dropped.
- **To check:** LongBench, RULER, SCROLLS, HotpotQA, MultiHop-RAG.

### 7. Financial Services — 10-K risk factor extraction

Structured risk-factor extraction from real SEC EDGAR filings, compared across companies.

- **Money shot:** highlighted spans on real filing text, risk-category heatmap across
  companies, year-over-year diff of risk language.
- **Why not another fraud model:** `fraud-detection-feast` already covers tabular fraud and
  has no visual payoff. EDGAR is free, public, and unlimited.

### Build order

Geospatial first (in progress), then Autonomous Systems pending dataset confirmation, then
Media, Logistics, Fintech, Observability, Context Engineering. One at a time, each reviewed
before the next so the report template compounds.

## Notes

- Phi-4-mini is the best candidate to demonstrate QLoRA's real value (3.8B won't fit full fine-tune on a T4)
- For gated models (LLaMA), need `HF_TOKEN` — good to show that flow
- Different architectures may need different LoRA target modules — see README for the lookup snippet
