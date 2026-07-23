# Long-Tail Scenario Coverage for Autonomous Driving

Survey a synthetic AV scenario dataset to answer the question that actually matters: **which situations do we have, and which are we missing?** Then render the multi-camera surround view for the ones worth looking at. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

The hard problem in AV data is not collecting more miles. A fleet logs thousands of hours of ordinary highway driving and almost no emergency-vehicle interactions in fog — then the model meets one on a Tuesday.

This pipeline builds the coverage picture: scenario family against condition, with the **gaps** called out. A coverage chart that only shows what you have is a vanity metric; the useful information is the zeros.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `index_scenarios` | Sample per-scenario metadata and captions, build coverage matrices | Family × time-of-day / weather / region matrices with gaps flagged |
| `render_scenario` | Decode the camera rig, run open-vocabulary detection, composite a surround view (fans out) | (feeds the player) |
| `pipeline` | | **Playable 7-camera surround with detection boxes**, per-prompt hit rates, coverage matrix, gap count |

## The Data

**[nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios)** — **OpenMDW 1.1, ungated**, README states plainly *"ready for commercial/non-commercial use."*

- **6,072 scenarios** — `lanechange` 4,854 · `emergency` 1,005 · `nudging` 213
- **6,010 carry the full 7-camera rig**: `front_wide` (120°), `front_tele` (30°), `rear_left`/`rear_right` (70°), and three 200° fisheyes (`left`, `right`, `rear`)
- **4K, 24 fps, ~460 frames** (~19 s) per camera; ~118 MB for all seven
- Per-camera **Qwen2.5-7B captions** plus structured `weather` / `time_of_day` / `surface_type` / `region`
- Generated with NVIDIA Omniverse; used in **midtraining of Cosmos 3** world foundation models

### This data is synthetic, and NVIDIA says so plainly

The dataset card states it *"exhibits a sim-to-real appearance gap relative to real driving footage"* and that *"a subset of authored agent behaviors may also appear unnatural — e.g. emergency vehicles cutting through dense traffic when open space is available nearby."* It is built for **long-tail coverage**, not photorealism, and is meant to be paired with real fleet data. The reports repeat this warning rather than burying it.

## Detection: what a model adds over rendering

Boxes on the forward view come from **[google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble)** (Apache-2.0, ungated) — an **open-vocabulary** detector prompted with free text (`"a police car"`, `"an ambulance"`, `"a person"`) rather than a fixed class list. Two things this measures that rendering cannot:

**1. Label verification — and why the whole rig matters.** A clip filed under `emergency` ought to contain an emergency vehicle. Measured on a real run (10 frames per camera, 0.32 score floor):

| Scenario | Detections | Pinhole conf. | Fisheye conf. | Emergency vehicle |
|---|---|---|---|---|
| `lanechange` | 112 | 0.432 | 0.390 | — |
| `nudging` | 72 | 0.406 | 0.375 | — |
| `emergency` | 38 | 0.434 | 0.336 | **found** |

An earlier version ran detection on `front_wide` alone and reported **no** emergency vehicle in the `emergency` clip — a **false negative**. The vehicle is present, just in another camera's field of view. Single-camera verification gave the wrong answer about the label; the full rig corrected it. That is the argument for treating the whole rig as the unit of analysis rather than a nicer picture.

**2. The sim-to-real gap, as a number.** OWLv2 was trained on real photographs. Even keeping only confident detections, mean confidence here is **0.34–0.43**, against 0.5–0.8 typical on real imagery. NVIDIA warns about the gap qualitatively; this measures it.

**3. Thresholds change conclusions, not just counts.** At a permissive 0.12 floor the three 200° fisheyes appeared to score **20–28%** below the rectilinear cameras — a tidy story about distortion hurting detection. Raising the floor collapsed that gap to **8–23%**, and on one scenario the fisheyes briefly scored *higher*. Most of the original "penalty" was marginal detections the wide views produced more of, not a real accuracy difference. Any statistic computed over unfiltered detections describes the noise floor as much as the model.

**Why 0.32.** Measured over 8 camera-clips, raising the floor from 0.12 to 0.32 cuts detections 127 → 58 and removes a false `car` that fires on the ego vehicle's own bonnet (~0.20). It also drops most traffic lights, which genuinely score 0.11–0.17 — a real cost, accepted because a clean frame of confident boxes reads better than a busy one full of maybes. The emergency vehicle measures **0.471**, so label verification is unaffected at any threshold tested.

Detection runs on **every** camera at ~1.8 s/frame on CPU (~7× the single-camera cost) and degrades to plain rendering if it fails. Not every scenario ships all seven cameras, and a transient fetch failure can silently reduce the count.

## Three traps worth knowing

**1. There is no `time_of_day` value called `"Day"`.** The real values are `Mid-day`, `Morning`, `Afternoon`, `Evening`, `Dusk`, `Twilight`, `Daytime`, `Night`. A `startswith("day")` filter matches **nothing** — it silently returned 0 daylight scenarios out of 40 before this was caught. Membership against the real set returns 23/40.

**2. OWLv2 needs `scipy`.** Without it the failure is an opaque `requires_backends` ImportError at *first inference*, not at import — so it survives every import check and dies mid-run.

**3. Metadata is not uniform.** Most scenarios carry the four condition fields, but some campaigns (notably within `emergency`) ship only a `caption_key` instead. Those are reported as **`Unlabelled`**, never folded into the counts — *"we never recorded the condition"* and *"this condition does not occur"* are different findings, and collapsing them makes the coverage matrix lie. Metadata completeness is reported as a first-class statistic.

## Setup

```bash
cd tutorials/av-scenario-coverage
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Local
flyte run --local --tui workflow.py pipeline --n_scenarios 2 --n_sample 80

# Remote
flyte run workflow.py pipeline

# Wider survey, longer clips
flyte run workflow.py pipeline --n_scenarios 4 --n_sample 400 --n_frames 24
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_scenarios` | `3` | Scenarios to render (spread across families) |
| `n_sample` | `150` | Scenarios to sample when indexing |
| `n_frames` | `16` | Frames per surround sequence |
| `tile_w` | `320` | Per-camera tile width |
| `fps` | `8` | Playback rate |
| `full_rig` | `True` | All 7 cameras; `False` uses the forward-biased 4 |

## Architecture

- **config.py** — CPU-only image + a **reusable** scenario pool for the fan-out
- **video.py** — PyAV decoding and surround compositing
- **detect.py** — OWLv2 open-vocabulary detection, box drawing, hit-rate summaries
- **report_helpers.py** — coverage matrix, surround player, charts
- **workflow.py** — tasks and pipeline

**PyAV ships statically-linked FFmpeg wheels**, so no system codec packages are needed in the image.

**The surround view is composited server-side** into one image per timestep. Seven synchronised `<video>` elements drift; one image cannot. It also means the report stays a single self-contained file and scrubs frame-accurately.

**Frames are JPEG, not PNG.** These are photographic: at 320×180 a PNG tile costs roughly 8× a JPEG at q78, and a seven-camera sequence is embedded as base64.

**Caching is off.** A cached task does not execute its body, and these reports are written by those bodies — a cache hit returns correct outputs with an empty report, silently.

## Related

- [av-perception-replay](../av-perception-replay/) — the complementary view: BEV reconstruction from HD map and 3D box annotations. That one is *what the car knows*; this one is *what the car sees*.

## References

- Dataset: [nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios](https://huggingface.co/datasets/nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios)
- Licence: [OpenMDW 1.1](https://openmdw.ai/)
