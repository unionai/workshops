"""
Panoptic segmentation with Mask2Former on COCO.

Panoptic segmentation is the most complete image-understanding task in one pass: it labels
*every* pixel, giving each countable object ("thing" — a person, a truck) its own instance
mask and each amorphous region ("stuff" — sky, road, grass) a single mask. It subsumes both
object detection and semantic segmentation.

  1. Prepare.  Pull the busiest COCO val images (most ground-truth segments) — a crowded
     scene shows far more than an empty one.
  2. Segment.  Run Mask2Former per image (fans out), render the coloured overlay with
     labelled boxes, and build a scene inventory.
  3. Report.   Input / predicted / ground-truth panels side by side, plus the object and
     region inventory and pixel coverage.

The model is applied zero-shot — it is the pretrained COCO panoptic checkpoint, run as-is.

Usage:
    flyte run --local --tui workflow.py pipeline --n_images 4
    flyte run workflow.py pipeline
"""

import asyncio
import io
import json
import logging
import os
import tempfile

import flyte
import flyte.io
import flyte.report

import report_helpers as rh
import segment as seg
from config import cpu_env, segment_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATASET_REPO = "nielsr/coco-panoptic-val2017"
DATASET_FILE = "data/train-00000-of-00002-ac9a9b049ea19ce7.parquet"
PIPELINE_STEPS = ["Prepare Images", "Segment", "Report"]


# ------------------------------------------------------------------
# Task 1: prepare images
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def prepare_data(n_images: int = 6, scan: int = 300) -> flyte.io.Dir:
    """
    Pull COCO val images, preferring the busiest scenes.

    Busy-ness is read from `segments_info` (no model needed), so a crowded banquet or
    street beats an empty landscape — panoptic segmentation has far more to show on a
    dense scene.
    """
    from huggingface_hub import hf_hub_download
    from PIL import Image
    import pyarrow.parquet as pq

    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Preparing COCO val2017</h2><p>Scanning for the busiest scenes…</p>"
    ), do_flush=True)

    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    table = pq.read_table(path)
    n = min(scan, table.num_rows)
    head = table.slice(0, n).to_pydict()

    # Rank by ground-truth segment count.
    order = sorted(range(n), key=lambda i: -len(head["segments_info"][i]))
    picked = order[:n_images]

    out_dir = tempfile.mkdtemp(prefix="panoptic_data_")
    index, previews = [], []
    for k, i in enumerate(picked):
        img = Image.open(io.BytesIO(head["image"][i]["bytes"])).convert("RGB")
        img.save(os.path.join(out_dir, f"{k:03d}_rgb.jpg"), quality=92)
        with open(os.path.join(out_dir, f"{k:03d}_gt.png"), "wb") as f:
            f.write(head["label"][i]["bytes"])
        index.append({"rgb": f"{k:03d}_rgb.jpg", "gt": f"{k:03d}_gt.png",
                      "gt_segments": len(head["segments_info"][i])})
        if k < 4:
            previews.append(f'<figure style="margin:0;"><img src="{rh.jpeg_uri(img)}" '
                            f'style="width:100%;border-radius:6px;">'
                            f'<figcaption style="font-size:.78em;color:#64748b;text-align:center;">'
                            f'{len(head["segments_info"][i])} GT segments</figcaption></figure>')

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"images": index}, f)

    html = f"""
    <h2>COCO val2017 — panoptic ground truth</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(index)}</div><div class="label">Scenes selected</div></div>
      <div class="stat"><div class="value">{table.num_rows:,}</div><div class="label">Val images available</div></div>
      <div class="stat"><div class="value">{max(x['gt_segments'] for x in index)}</div><div class="label">Segments in busiest</div></div>
      <div class="stat"><div class="value">133</div><div class="label">Panoptic classes</div></div>
    </div>
    <div class="note">
      Scenes are ranked by ground-truth segment count, so the report shows dense,
      information-rich images rather than empty ones. Every pixel in COCO panoptic is
      labelled as either a countable <b>thing</b> or an amorphous <b>stuff</b> region.
    </div>
    <div class="chart-container">
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;">{''.join(previews)}</div>
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: segment one image  (fans out)
# ------------------------------------------------------------------

@segment_env.task(retries=2)
async def segment_image(data_dir: flyte.io.Dir, rgb_name: str, gt_name: str) -> flyte.io.Dir:
    """Run Mask2Former on one image, render overlay + GT comparison + inventory."""
    from PIL import Image

    local = await data_dir.download()
    img = Image.open(os.path.join(local, rgb_name)).convert("RGB")

    segments = seg.segment(img)
    ov = seg.overlay(img, segments)
    inv = seg.inventory(segments)
    cov = seg.coverage(segments, (img.size[1], img.size[0]))

    _, gt_col, n_gt = seg.decode_gt(Image.open(os.path.join(local, gt_name)))

    out_dir = tempfile.mkdtemp(prefix="panoptic_out_")
    img.save(os.path.join(out_dir, "rgb.jpg"), quality=90)
    Image.fromarray(ov).save(os.path.join(out_dir, "pred.jpg"), quality=90)
    Image.fromarray(gt_col).save(os.path.join(out_dir, "gt.jpg"), quality=90)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({"rgb": rgb_name, "inventory": inv, "coverage": cov,
                   "n_pred": len(segments), "n_gt": n_gt,
                   "mean_score": (sum(s["score"] for s in segments if not s["is_stuff"])
                                  / max(inv["n_things"], 1))}, f)

    log.info(f"{rgb_name}: {len(segments)} segments ({inv['n_things']} things), "
             f"coverage {cov:.1%}, GT segments {n_gt}")
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(n_images: int = 6, scan: int = 300) -> str:
    """Segment a set of COCO scenes and report predicted vs ground-truth panoptic."""
    from collections import Counter

    async def step(n, note):
        await flyte.report.replace.aio(
            rh.wrap_report(f"<h2>Panoptic Segmentation</h2>{rh.progress_html(PIPELINE_STEPS, n, note)}"),
            do_flush=True,
        )

    await step(1, "Selecting the busiest COCO scenes…")
    data_dir = await prepare_data(n_images=n_images, scan=scan)
    with open(os.path.join(await data_dir.download(), "index.json")) as f:
        images = json.load(f)["images"]

    await step(2, f"Segmenting {len(images)} scenes…")
    with flyte.group("segment-scenes"):
        results = await asyncio.gather(*[
            segment_image(data_dir=data_dir, rgb_name=im["rgb"], gt_name=im["gt"])
            for im in images
        ], return_exceptions=True)
    dirs = [r for r in results if not isinstance(r, Exception)]
    for r in results:
        if isinstance(r, Exception):
            log.warning(f"segment failed: {r}")
    if not dirs:
        raise RuntimeError("Every scene failed to segment.")

    await step(3, "Assembling the report…")
    scored = []
    for d in dirs:
        local = await d.download()
        with open(os.path.join(local, "stats.json")) as f:
            st = json.load(f)
        st["dir"] = local
        scored.append(st)
    scored.sort(key=lambda s: -s["n_pred"])

    total_things = Counter()
    for st in scored:
        total_things.update(st["inventory"]["things"])
    mean_cov = sum(s["coverage"] for s in scored) / len(scored)
    total_objects = sum(s["inventory"]["n_things"] for s in scored)

    def _uri(d, name):
        with open(os.path.join(d, name), "rb") as fh:
            return "data:image/jpeg;base64," + __import__("base64").b64encode(fh.read()).decode()

    blocks = ""
    for st in scored:
        d = st["dir"]
        blocks += (
            f"<h3>{st['n_pred']} segments &nbsp;"
            f"<span style='font-size:.7em;color:#64748b;'>"
            f"{st['inventory']['n_things']} objects · {st['inventory']['n_stuff']} regions · "
            f"{st['coverage']:.0%} pixel coverage · GT has {st['n_gt']}</span></h3>"
            + rh.triptych(_uri(d, "rgb.jpg"), _uri(d, "pred.jpg"), _uri(d, "gt.jpg"))
            + f"<div class='card'>{rh.inventory_chips(st['inventory'])}</div>"
        )

    labels = [k for k, _ in total_things.most_common(10)]
    values = [v for _, v in total_things.most_common(10)]

    html = f"""
    <h2>Panoptic Segmentation — Mask2Former on COCO</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(scored)}</div><div class="label">Scenes segmented</div></div>
      <div class="stat"><div class="value">{total_objects}</div><div class="label">Objects detected</div></div>
      <div class="stat"><div class="value">{len(total_things)}</div><div class="label">Distinct classes</div></div>
      <div class="stat"><div class="value">{mean_cov:.0%}</div><div class="label">Mean pixel coverage</div></div>
      <div class="stat"><div class="value">0</div><div class="label">Params trained</div></div>
    </div>
    <div class="note">
      <b>Every pixel is labelled.</b> Panoptic segmentation gives each object its own
      instance mask (with a bounding box and confidence) and every background region a
      class — detection and semantic segmentation in a single forward pass. The model is the
      pretrained COCO checkpoint, applied zero-shot; near-100% pixel coverage is the
      signature of a panoptic result, where nothing is left unlabelled.
    </div>
    <div class="chart-container">
      {rh.make_bar_chart(labels, values, title="Objects detected by class (all scenes)")}
    </div>
    {blocks}
    <div class="note">
      The predicted and ground-truth panels use independent colours per segment, so match
      the <i>regions</i>, not the hues. The model typically recovers the salient objects and
      large regions cleanly; ground truth carries more segments because it labels every
      distant background instance a crowded scene contains.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({"n_scenes": len(scored), "objects": total_objects,
                       "classes": len(total_things), "mean_coverage": mean_cov,
                       "top_classes": dict(total_things.most_common(8))})
