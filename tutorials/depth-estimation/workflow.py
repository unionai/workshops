"""
Monocular depth estimation with Depth Anything V2, evaluated against a real depth sensor.

Turn a single flat photo into per-pixel depth — then hold the prediction to the truth. The
input is NYU Depth V2: indoor RGB frames each paired with a Kinect depth capture, so every
prediction can be scored against what a physical sensor actually measured.

  1. Prepare.  Pull real RGB + Kinect-depth pairs from the dataset.
  2. Estimate. For each image, run Depth Anything V2 (fans out), align the scaleless
     prediction to metric ground truth, score it, and render the panels — including a
     depth-driven parallax that turns the flat photo back into 3D.
  3. Evaluate. Aggregate the standard depth metrics and surface best and worst cases.

Depth Anything V2 is a foundation model applied **zero-shot**: it was never trained on
this dataset, yet predicts usable relative depth on scenes it has never seen.

Usage:
    flyte run --local --tui workflow.py pipeline --n_images 4
    flyte run workflow.py pipeline
"""

import asyncio
import json
import logging
import os
import tarfile
import tempfile

import flyte
import flyte.io
import flyte.report

import depth
import report_helpers as rh
from config import cpu_env, estimate_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATASET_REPO = "sayakpaul/nyu_depth_v2"
DATASET_FILE = "data/train-000000.tar"
PIPELINE_STEPS = ["Prepare Data", "Estimate Depth", "Evaluate"]


# ------------------------------------------------------------------
# Task 1: prepare data
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def prepare_data(n_images: int = 6, stride: int = 40) -> flyte.io.Dir:
    """
    Pull RGB + Kinect-depth pairs from NYU Depth V2.

    The dataset ships as a loading script plus webdataset tars. Loading scripts were
    removed in `datasets>=4.0`, so we fetch a tar directly and read its HDF5 samples —
    both the working path and the simpler one.
    """
    import h5py
    import numpy as np
    from huggingface_hub import hf_hub_download
    from PIL import Image

    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Preparing NYU Depth V2</h2><p>Fetching RGB + depth pairs…</p>"
    ), do_flush=True)

    tar_path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")

    out_dir = tempfile.mkdtemp(prefix="depth_data_")
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if m.name.endswith(".h5")]
        picked = members[::stride][:n_images]
        extract_dir = tempfile.mkdtemp()
        tar.extractall(extract_dir, members=picked, filter="data")

    h5_files = sorted(
        os.path.join(dp, f)
        for dp, _, fs in os.walk(extract_dir) for f in fs if f.endswith(".h5")
    )[:n_images]

    index = []
    previews = []
    for i, h5 in enumerate(h5_files):
        with h5py.File(h5) as f:
            rgb = np.transpose(np.asarray(f["rgb"]), (1, 2, 0)).astype("uint8")
            gt = np.asarray(f["depth"], dtype="float32")
        np.savez_compressed(os.path.join(out_dir, f"{i:03d}.npz"), rgb=rgb, depth=gt)
        index.append(f"{i:03d}.npz")
        if i < 4:
            previews.append(
                f'<figure><img src="{rh.jpeg_uri(rgb)}">'
                f'<figcaption>scene {i} · {gt[gt>0].min():.1f}–{gt[gt>0].max():.1f} m</figcaption></figure>'
            )

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump({"files": index}, f)

    html = f"""
    <h2>NYU Depth V2 — RGB &amp; sensor depth</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(index)}</div><div class="label">Scenes loaded</div></div>
      <div class="stat"><div class="value">640&times;480</div><div class="label">Resolution</div></div>
      <div class="stat"><div class="value">Kinect</div><div class="label">Depth source</div></div>
      <div class="stat"><div class="value">Apache-2.0</div><div class="label">License</div></div>
    </div>
    <div class="note">
      Every RGB frame is paired with a real depth capture from a Microsoft Kinect. That
      ground truth is what makes this an <b>evaluation</b> rather than a gallery: the
      model's guess is scored against what a physical sensor measured, pixel by pixel.
    </div>
    <div class="chart-container"><div class="panels">{''.join(previews)}</div></div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: estimate depth  (fans out)
# ------------------------------------------------------------------

@estimate_env.task(retries=2)
async def estimate_depth(data_dir: flyte.io.Dir, sample: str,
                         parallax_frames: int = 12) -> flyte.io.Dir:
    """Predict depth for one scene, align to ground truth, score, and render."""
    import numpy as np
    from PIL import Image

    local = await data_dir.download()
    arr = np.load(os.path.join(local, sample))
    rgb, gt = arr["rgb"], arr["depth"]

    pred = depth.predict(Image.fromarray(rgb))
    aligned, mask, a, b = depth.align_to_metric(pred, gt)
    if aligned is None:
        raise RuntimeError(f"{sample}: too few valid ground-truth pixels to align")
    m = depth.metrics(aligned, gt, mask)

    out_dir = tempfile.mkdtemp(prefix="depth_out_")
    from PIL import Image as PILImage

    PILImage.fromarray(rgb).save(os.path.join(out_dir, "rgb.jpg"), quality=88)
    PILImage.fromarray(depth.colorize(pred, invert=False)).save(
        os.path.join(out_dir, "pred.jpg"), quality=88)
    PILImage.fromarray(depth.colorize(gt, mask=mask, invert=True)).save(
        os.path.join(out_dir, "gt.jpg"), quality=88)
    PILImage.fromarray(depth.error_map(aligned, gt, mask)).save(
        os.path.join(out_dir, "err.jpg"), quality=88)

    # Parallax frames (downscaled — smoothness matters more than resolution here).
    small_rgb = np.asarray(PILImage.fromarray(rgb).resize((512, 384)))
    small_pred = np.asarray(PILImage.fromarray(pred).resize((512, 384)))
    frames_dir = os.path.join(out_dir, "parallax")
    os.makedirs(frames_dir, exist_ok=True)
    for i, fr in enumerate(depth.parallax_frames(small_rgb, small_pred, n_frames=parallax_frames)):
        PILImage.fromarray(fr).save(os.path.join(frames_dir, f"{i:02d}.jpg"), quality=82)

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump({"sample": sample, "metrics": m, "scale": a, "shift": b,
                   "depth_min": float(gt[gt > 0].min()), "depth_max": float(gt[gt > 0].max())}, f)

    log.info(f"{sample}: AbsRel={m['abs_rel']:.3f} delta1={m['delta1']:.3f}")
    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(n_images: int = 6, stride: int = 40, parallax_frames: int = 12,
                   fps: int = 12) -> str:
    """Estimate depth for a handful of scenes and score against the Kinect ground truth."""
    import numpy as np

    async def step(n, note):
        await flyte.report.replace.aio(
            rh.wrap_report(f"<h2>Depth Estimation</h2>{rh.progress_html(PIPELINE_STEPS, n, note)}"),
            do_flush=True,
        )

    await step(1, "Loading RGB + sensor-depth pairs…")
    data_dir = await prepare_data(n_images=n_images, stride=stride)
    with open(os.path.join(await data_dir.download(), "index.json")) as f:
        samples = json.load(f)["files"]

    await step(2, f"Estimating depth for {len(samples)} scenes…")
    with flyte.group("estimate-depth"):
        results = await asyncio.gather(*[
            estimate_depth(data_dir=data_dir, sample=s, parallax_frames=parallax_frames)
            for s in samples
        ], return_exceptions=True)
    dirs = [r for r in results if not isinstance(r, Exception)]
    for r in results:
        if isinstance(r, Exception):
            log.warning(f"estimate failed: {r}")
    if not dirs:
        raise RuntimeError("Every scene failed to estimate.")

    await step(3, "Scoring against ground truth…")
    scored = []
    for d in dirs:
        local = await d.download()
        with open(os.path.join(local, "stats.json")) as f:
            st = json.load(f)
        st["dir"] = local
        scored.append(st)
    scored.sort(key=lambda s: s["metrics"]["abs_rel"])

    mean = {k: float(np.mean([s["metrics"][k] for s in scored]))
            for k in scored[0]["metrics"]}

    def _panels_block(st):
        d = st["dir"]

        def u(name):
            with open(os.path.join(d, name), "rb") as fh:
                return "data:image/jpeg;base64," + __import__("base64").b64encode(fh.read()).decode()

        frames_dir = os.path.join(d, "parallax")
        frame_uris = [u(os.path.join("parallax", n)) for n in sorted(os.listdir(frames_dir))]
        m = st["metrics"]
        return (
            f"<h3>Scene {st['sample'].split('.')[0]} &nbsp;"
            f"<span style='font-size:.7em;color:#78716c;'>AbsRel {m['abs_rel']:.3f} · "
            f"&delta;1 {m['delta1']:.1%} · range {st['depth_min']:.1f}–{st['depth_max']:.1f} m</span></h3>"
            + rh.panels_html(u("rgb.jpg"), u("pred.jpg"), u("gt.jpg"), u("err.jpg"))
            + rh.parallax_player(frame_uris, slug=st["sample"].split(".")[0], fps=fps,
                                 caption="3D parallax reconstructed from the single photo above, "
                                         "using the predicted depth")
        )

    best = _panels_block(scored[0])
    worst = _panels_block(scored[-1]) if len(scored) > 1 else ""

    html = f"""
    <h2>Monocular Depth — Depth Anything V2 vs Kinect</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(scored)}</div><div class="label">Scenes scored</div></div>
      <div class="stat"><div class="value">{mean['abs_rel']:.3f}</div><div class="label">Mean AbsRel &darr;</div></div>
      <div class="stat"><div class="value">{mean['delta1']:.1%}</div><div class="label">&delta;&lt;1.25 &uarr;</div></div>
      <div class="stat"><div class="value">{mean['rmse']:.2f} m</div><div class="label">RMSE</div></div>
      <div class="stat"><div class="value">0</div><div class="label">Params trained</div></div>
    </div>
    <div class="note">
      <b>Nothing was trained.</b> Depth Anything V2 is a foundation model applied zero-shot;
      it never saw NYU Depth V2. It predicts <i>relative</i> depth with no absolute scale, so
      each prediction is aligned to the sensor with one least-squares scale and shift — the
      standard scale-invariant protocol — before scoring. <b>&delta;&lt;1.25</b> is the
      fraction of pixels within 25% of the true depth; higher is better.
    </div>
    {rh.turbo_legend()}
    <div class="chart-container">
      {rh.make_bar_chart(["δ<1.25", "δ<1.25²", "δ<1.25³"],
                         [mean['delta1'], mean['delta2'], mean['delta3']],
                         colors=["#16a34a", "#65a30d", "#a3a300"],
                         title="Depth accuracy (fraction of pixels within threshold)", y_max=1.0)}
    </div>

    <h2>Best scene</h2>
    {best}
    {"<h2>Hardest scene</h2>" + worst if worst else ""}

    <div class="note">
      Depth Anything struggles where a single image is genuinely ambiguous — reflective
      surfaces, glass, and untextured walls have no monocular depth cue, and the error map
      lights up exactly there. The parallax view is built purely from the predicted depth:
      a flat photo, given geometry by the model, viewed from angles the camera never saw.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({"n_scenes": len(scored), "mean_metrics": mean,
                       "best": scored[0]["sample"], "worst": scored[-1]["sample"]})
