# AV Scene Replay with Cosmos Drive Dreams

Reconstruct real autonomous-driving scenes from NVIDIA's **[Cosmos Drive Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)** dataset — full 3D HD map, per-frame 3D object boxes with persistent track IDs, ego pose — and render them as playable bird's-eye-view sequences. Orchestrated with [Flyte](https://docs.union.ai/) on [Union](https://www.union.ai/).

## Why This Matters

Perception teams spend most of their time not training models but **looking at labelled data**: which scenes are worth annotating, which are over-represented, where the class balance is wrong. This pipeline is that loop — screen thousands of clips, replay the interesting ones, compare them.

### There is no model here, deliberately

**No pretrained 3D object detectors are published on Hugging Face.** A search for `mmdetection3d` returns nothing usable; `pointpillars` and `bevformer` hits have single-digit downloads; the only credible weights are ONNX inside Autoware as a split two-stage graph requiring Autoware's own preprocessing. Rather than ship a demo that pretends to detect, this renders **ground-truth annotations**. That is the honest framing, and it is also genuinely what the job looks like.

## What This Pipeline Does

| Stage | What it does | Report visuals |
|-------|-------------|----------------|
| `screen_clips` | Score all 5,843 clip captions for scene density, keep the richest | Ranked caption table, score distribution |
| `replay_clip` | Download one clip's annotations, render a BEV sequence (fans out) | (feeds the player) |
| `compare_clips` | Aggregate class mix and crowding across scenes | Class histogram, objects-over-time, per-clip table |
| `pipeline` | | **Playable BEV replays**, legend, summary |

## Screening is the highest-leverage step

Clips are wildly unequal, and the difference decides whether the demo is worth watching:

| Clip | Tracks | Objects/frame | Classes |
|---|---|---|---|
| Randomly chosen | 19 | **8** | cars, trucks, trailers |
| Top-ranked by caption | 156 | **140** | + 7,927 pedestrians, 1,200 riders, buses |

**17× more objects**, and the difference between an empty night motorway and a busy daytime intersection. Captions ship as small text files, so ranking a few hundred costs seconds. Skipping this step is how you end up building a beautiful renderer pointed at nothing.

## The Data

**CC-BY-4.0, ungated.** Per clip (~20 s, ~300 frames, 7-camera rig):

| Layer | Contents |
|-------|----------|
| `all_object_info` | Per-frame 3D boxes: `object_to_world` 4×4, `object_lwh`, class, `object_is_moving`, **persistent track ID** |
| `3d_lanes`, `3d_lanelines`, `3d_road_boundaries`, `3d_road_markings`, `3d_crosswalks`, `3d_wait_lines` | HD map geometry |
| `3d_poles`, `3d_traffic_lights`, `3d_traffic_signs` | Map furniture |
| `pose`, `vehicle_pose` | Per-frame ego and camera poses |
| `captions` | Natural-language scene description |
| `lidar_raw` | Velodyne sweeps — **not used**, see below |

**Annotations are ~9 MB per clip; `lidar_raw` alone is ~370 MB.** Since the BEV is built from the map and the boxes, skipping LiDAR makes the pipeline roughly 40× cheaper for no visual loss. Add it if you want the point cloud.

### Four different geometry schemas

A trap worth knowing, because handling only one yields a near-empty map rather than an error:

| Encoding | Layers |
|---|---|
| `polylines3d.polylines[].vertices` | `3d_lanes` |
| `polyline3d.vertices` | lanelines, road boundaries, wait lines, poles |
| `surface.vertices` | crosswalks, road markings |
| `cuboid3d.vertices` | traffic lights, traffic signs |

Parsing only the first gave **584** map features. Handling all four gives **1,302**.

## Setup

```bash
cd tutorials/av-perception-replay
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage

```bash
# Local
flyte run --local --tui workflow.py pipeline --n_clips 2 --n_frames 40

# Remote
flyte run workflow.py pipeline

# Wider view, more frames
flyte run workflow.py pipeline --n_clips 4 --n_frames 90 --fwd 120 --side 60
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_clips` | `3` | Clips to replay (highest density first) |
| `n_sample` | `200` | Captions to sample when screening |
| `n_frames` | `60` | Frames rendered per clip |
| `fwd` / `side` | `80` / `45` | BEV extent in metres (forward / lateral) |
| `fps` | `10` | Playback rate |

## Reading the BEV

Rig frame, metres: **+x forward, +y left**, drawn with forward up. Range rings every 25 m.

- **Filled box** = moving · **outline only** = stationary — straight from `object_is_moving`
- **Tick** from centre = heading
- **Faint tail** = recent track history, possible because track IDs persist across frames
- **Red triangle** at origin = ego vehicle

## Architecture

- **config.py** — CPU-only image + a **reusable** clip pool for the fan-out
- **bev.py** — annotation parsing (all four schemas) and the BEV renderer
- **report_helpers.py** — self-contained SVG charts and the frame-by-frame player
- **workflow.py** — tasks and the pipeline

The player pre-renders PNG frames and swaps them in JS rather than encoding video: the report stays a single self-contained file, needs no encoder in the image, and scrubs frame-accurately, which a GIF cannot do. Clip tasks are wrapped in `flyte.group("replay-clips")` so the fan-out collapses to one folder in the UI.

**Caching is off.** A cached task does not execute its body, and these reports are written by those bodies — a cache hit returns correct outputs attached to an empty report, silently.

## References

- Cosmos Drive Dreams ([arXiv:2506.09042](https://arxiv.org/abs/2506.09042))
- Dataset: [nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)
