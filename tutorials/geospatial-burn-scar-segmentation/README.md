# Burn Scar Mapping with a Geospatial Foundation Model

Fine-tune NASA/IBM's **[Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)** geospatial foundation model to segment wildfire burn scars in satellite imagery, then apply it at scale — query a live [STAC](https://stacspec.org/) catalog for a real fire, chip the scenes into tiles, segment every tile in parallel, and mosaic the results back into one map. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

A geospatial foundation model is pretrained, self-supervised, on millions of satellite scenes — so it already "knows" what vegetation, water, soil, and cloud look like across seasons and sensors. Prithvi-EO-2.0 was trained on 4.2M Harmonized Landsat–Sentinel-2 (HLS) scenes. That means you can adapt it to a specific task — here, mapping burn scars — with only a few hundred labelled examples and a small decoder, instead of training a segmentation network from scratch on data you don't have.

This tutorial shows both halves of how earth-observation teams actually work:

1. **Adapt the model** on a labelled dataset with ground truth.
2. **Apply it at planetary scale** on imagery it has never seen, fanning out across tiles.

The second half is where the orchestrator becomes the point: hundreds of small, independently retryable tile tasks running at once, with the mosaic assembling as they land.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `prepare_data` | Download HLS Burn Scars from the Hub, inspect class balance and band distributions | SWIR/NIR/Red composites, MTBS ground-truth overlays, per-band reflectance histograms |
| `train` | Fine-tune Prithvi (frozen encoder + segmentation decoder) | Live loss curve, validation IoU per epoch, parameter breakdown |
| `evaluate` | Score the validation split, surface best and hardest scenes | IoU/Dice/precision/recall, prediction-vs-truth overlays, quality bar chart |
| `discover_tiles` | Query a live STAC catalog for a real fire and lay out an AOI tile grid | Pre/post scene table, candidate counts, tile plan |
| `segment_tile` | Range-read one tile from cloud-optimized GeoTIFFs, segment it, compute dNBR | (feeds the mosaic) |
| `mosaic` | Stitch tiles into a map product | **Draggable before/after wipe**, tile fan-out grid, most-burned-tiles table |

## The Dataset

**[ibm-nasa-geospatial/hls_burn_scars](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)** — 804 512×512 scenes of Harmonized Landsat & Sentinel-2 imagery over the contiguous US, 2018–2021. **CC-BY-4.0.**

- **6 bands:** Blue, Green, Red, NIR, SWIR1, SWIR2 (surface reflectance ×10,000)
- **Masks:** `1` burn / `0` unburned / `-1` nodata — the nodata class is excluded from both loss and metrics
- **Ground truth is real:** labels come from [MTBS](https://mtbs.gov/) (Monitoring Trends in Burn Severity) fire perimeters, not synthetic annotations

### Why six bands, not RGB

Burned ground is **dark in near-infrared** (no live vegetation) and **bright in shortwave-infrared** (exposed char and soil). In the SWIR2/NIR/Red composite that becomes an unmistakable **magenta-crimson** signature — SWIR2 drives the red channel, NIR the green — while healthy forest goes vivid green. In true colour the same scar is an ordinary brown smudge, which is exactly why an RGB-only burn dataset would be easier to load and far less useful.

Two things look suspicious on a burn map but are not burn, so the report legend names them explicitly: **water** (NIR collapses, NDWI > 0.8) and **terrain shadow** on north-facing slopes (NIR drops ~60% while the visible bands barely move, because shadows are still lit by diffuse skylight).

## The Model

**[Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M)** — a 300M-parameter Vision Transformer (24 blocks, 1024-dim, 3D patch embedding) pretrained on HLS imagery. **Apache-2.0.**

We vendor IBM's own model definition (`prithvi_mae.py`, Apache-2.0, unmodified) and attach a lightweight FPN-style decoder. By default the encoder stays **frozen** and only the ~1.5M-parameter decoder trains — the honest foundation-model story, and it keeps the run on a single mid-range GPU. Pass `--freeze_encoder False` for a full fine-tune.

> **Note:** the model card lists `library_name: terratorch`. We deliberately don't use TerraTorch — it pulls in 60+ dependencies (geopandas, lightning, torchgeo, diffusers…), a real failure surface for a tutorial. Vendoring the one model file keeps the dependency list to torch + timm + einops + rasterio.

## Setup

```bash
cd tutorials/geospatial-burn-scar-segmentation

# Python 3.12+ (numpy 2.5 / rasterio 1.5 require it)
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

### Local (quick smoke test — CPU-friendly subset)

```bash
flyte run --local --tui workflow.py pipeline --max_scenes 24 --epochs 2 --tile_limit 12
```

### Local (a real training run)

```bash
flyte run --local --tui workflow.py pipeline --epochs 20
```

### Remote (full training + a large AOI mosaic)

```bash
flyte run workflow.py pipeline

# A different fire
flyte run workflow.py pipeline --aoi creek

# Full fine-tune instead of frozen encoder (wants the L40s)
flyte run workflow.py pipeline --freeze_encoder False --epochs 40
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `aoi` | `dixie` | Fire to map: `dixie`, `creek`, or `cameron` |
| `max_scenes` | `0` | Cap training scenes (0 = all 540) |
| `epochs` | `20` | Training epochs |
| `batch_size` | `4` | Scenes per batch |
| `learning_rate` | `1e-3` | AdamW learning rate (decoder) |
| `freeze_encoder` | `True` | Freeze the 300M encoder; train only the decoder |
| `tile_px` | `256` | Tile size for the mosaic stage |
| `tile_limit` | `64` | Max tiles to segment (centred on the AOI) |

## Areas of Interest

Each is a real, large, well-documented wildfire. The pipeline finds a clear pre-fire and post-fire Sentinel-2 scene from the **same MGRS tile** (so their pixel grids align exactly) via the public [Element84 earth-search](https://element84.com/earth-search/) STAC API — no account, no API key.

| AOI | Fire | Year | Scale |
|-----|------|------|-------|
| `dixie` | Dixie Fire, California | 2021 | ~389,000 ha — 2nd largest in CA history |
| `creek` | Creek Fire, California | 2020 | ~154,000 ha, Sierra National Forest |
| `cameron` | Cameron Peak Fire, Colorado | 2020 | ~84,000 ha — largest in CO history |

## Key Concepts

**Foundation model fine-tuning** — Adapt a large pretrained model to a specific task by training a small task head on top of frozen features. With ~500 labelled scenes, the decoder is the part worth training; the encoder's representation already came from millions of scenes.

**IoU vs pixel accuracy** — ~89% of labelled pixels are unburned, so a model that predicts "nothing burned" everywhere still scores ~0.89 pixel accuracy. **IoU on the burn class** is the metric that actually moves, which is why the report leads with it.

**dNBR (differenced Normalized Burn Ratio)** — The standard index-based burn metric, `NBR_pre − NBR_post` where `NBR = (NIR − SWIR2) / (NIR + SWIR2)`. Computed independently of the model, so agreement between a high predicted burn fraction and a high dNBR is genuine corroboration rather than the model marking its own homework.

**STAC + COG + windowed reads** — A tile is a *window* into a cloud-optimized GeoTIFF, fetched with HTTP range requests. Each tile task reads only its own few megabytes instead of pulling a full 110 km scene, which is what makes tile-level fan-out cheap.

**MGRS tile alignment** — Sentinel-2 scenes sharing an MGRS tile id (e.g. `MGRS-10TFK`) share an identical pixel grid. Choosing pre- and post-fire scenes from the same tile makes the comparison valid pixel-for-pixel.

## How It Works

```
HLS Burn Scars ──> Prithvi-EO-2.0 ──> Fine-tuned ──> Evaluate
  (804 scenes,       (frozen 300M      segmenter        (IoU/Dice)
   MTBS labels)       encoder +            │
                      decoder)             │
                                           ▼
STAC query ──> chip AOI ──> segment_tile (fan-out) ──> mosaic
 (live catalog,  into tiles    │  many parallel        │  before/after
  real fire)                   │  tile tasks           │  wipe + map
                               ▼                        ▼
                        window-read COGs         "how much burned,
                        + dNBR cross-check         and where?"
```

## Architecture

- **config.py** — Flyte image (`.with_pip_packages`, per repo convention) + three environments: GPU (L40s) for training, a **reusable T4 pool** for tile fan-out, and CPU for orchestration and reporting
- **model.py** — Vendored Prithvi encoder + FPN decoder, checkpoint split/load, band normalization
- **report_helpers.py** — Self-contained SVG charts and base64 raster rendering (no external assets)
- **workflow.py** — Tasks, the STAC/tiling logic, and the pipeline orchestrator
- **prithvi_mae.py** — IBM's model definition, vendored unmodified (Apache-2.0)

### Caching empties the reports

Worth knowing before you reach for `cache="auto"`: **a cached task does not execute its body.** The reports in this pipeline are written by those bodies, via `flyte.report.replace(...)`, so a cache hit returns entirely correct outputs attached to an *empty report*.

That failure is silent. The run still goes green, the metrics are still right, the DAG looks perfect — and every visual the tutorial exists to produce is blank. It's a genuinely useful accelerator while iterating on a downstream stage (skipping a GPU training run is worth a lot), but it must be off for any run you intend to look at or record. Caching is therefore disabled here by default.

### Sizing the tile fan-out

Three decisions here are worth copying, because getting them wrong produces an OOM loop rather than a slow run:

- **The tile pool needs no GPU at all — measure before reaching for one.** It is tempting to assume a 300M ViT-L must be GPU work. At 256×256 with a 16×16 patch the encoder sees only **256 tokens**, and a forward measures **~0.15 s on 4 CPU threads** — the entire 36-tile grid is a few seconds of compute. A GPU would spend longer queueing for a node than it saved. CPU replicas schedule instantly and cost a fraction.
- **`concurrency` shares a process, so it shares memory.** Concurrent tile tasks run as coroutines in one Python process. That is what makes a cached model useful, but it also means an unguarded cache lets every coroutine build its own copy of a 300M model simultaneously. `workflow.py` guards the cache with an `asyncio.Lock`; the resulting shared model is safe to use concurrently because inference runs in eval mode under `no_grad`.
- **Checkpoint size is multiplied by the fan-out.** With a frozen encoder, 303.9M of the 305.3M parameters are bit-identical to the public Prithvi weights. Saving them produces a 1.2 GB artifact that *every* tile task then downloads. `split_state_dict()` persists only the decoder — **5.8 MB instead of 1,222 MB, a 211× reduction** — and `load_segmenter()` rebuilds the encoder from the Hub cache. Verified lossless: predictions are bit-identical either way.

## References

- Szwarcman et al., "Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation" ([arXiv:2412.02732](https://arxiv.org/abs/2412.02732))
- HLS Burn Scars dataset — [doi:10.57967/hf/0956](https://huggingface.co/datasets/ibm-nasa-geospatial/hls_burn_scars)
- MTBS — Monitoring Trends in Burn Severity: [mtbs.gov](https://mtbs.gov/)
- STAC — SpatioTemporal Asset Catalog: [stacspec.org](https://stacspec.org/)
