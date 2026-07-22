"""
Burn scar mapping with NASA/IBM's Prithvi geospatial foundation model.

Two halves that mirror how earth-observation teams actually work:

  1. Adapt a foundation model. Fine-tune Prithvi-EO-2.0-300M on HLS Burn Scars (804
     labelled Harmonized Landsat/Sentinel-2 scenes, ground truth from MTBS fire
     perimeters), then evaluate it.

  2. Apply it at scale. Query a live STAC catalog for a real area of interest, chip the
     matching scenes into tiles, segment every tile in parallel, and mosaic the results
     back into one map product.

The second half is the part that makes the orchestrator visible: hundreds of small,
independently retryable tile tasks fanning out at once, with the mosaic assembling in the
report as they land.

Usage:
    # Local smoke test (CPU, tiny subset)
    flyte run --local --tui workflow.py pipeline --max_scenes 24 --epochs 2 --tile_limit 12

    # Remote, full training + a real AOI mosaic
    flyte run workflow.py pipeline

    # Full fine-tune instead of frozen encoder (needs the L40S)
    flyte run workflow.py pipeline --freeze_encoder False --epochs 40
"""

import asyncio
import json
import logging
import os
import tarfile
import tempfile
import urllib.request

import flyte
import flyte.io
import flyte.report

import report_helpers as rh
from config import cpu_env, gpu_env, tile_env

# IMPORTANT: these are imported at *module* scope rather than lazily inside the tasks.
# Flyte's code bundler ships the module-level import closure of this file, so a local
# helper that is only imported inside a function body never makes it into the bundle and
# the task dies remotely with `ModuleNotFoundError` — while working perfectly locally.
# This line is what pulls in model.py and, transitively, prithvi_mae.py.
from model import (
    PrithviSegmenter,
    load_pretrained_encoder,
    load_segmenter,
    normalize,
    split_state_dict,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATASET_REPO = "ibm-nasa-geospatial/hls_burn_scars"
DATASET_FILE = "hls_burn_scars.tar.gz"
MODEL_REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
MODEL_FILE = "Prithvi_EO_V2_300M.pt"

BAND_NAMES = ["B02 Blue", "B03 Green", "B04 Red", "B8A NIR", "B11 SWIR1", "B12 SWIR2"]

# Live STAC catalog. Public, no auth, no key.
STAC_URL = "https://earth-search.aws.element84.com/v1/search"
STAC_COLLECTION = "sentinel-2-l2a"

# Sentinel-2 L2A asset keys on Element84, in Prithvi's HLS band order.
S2_ASSETS = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# Areas of interest for the mosaic half. Each is a real, large, well-documented fire.
AOIS = {
    "dixie": {
        "name": "Dixie Fire, California",
        "year": 2021,
        "bbox": [-121.45, 39.95, -121.05, 40.30],
        "pre": ("2021-06-01", "2021-07-10"),
        "post": ("2021-10-15", "2021-11-30"),
        "blurb": "Second-largest wildfire in California history — roughly 389,000 hectares "
                 "across five counties between July and October 2021.",
    },
    "creek": {
        "name": "Creek Fire, California",
        "year": 2020,
        "bbox": [-119.45, 37.05, -119.05, 37.40],
        "pre": ("2020-06-01", "2020-08-20"),
        "post": ("2020-11-01", "2020-12-20"),
        "blurb": "Burned roughly 154,000 hectares of the Sierra National Forest after "
                 "igniting in September 2020.",
    },
    "cameron": {
        "name": "Cameron Peak Fire, Colorado",
        "year": 2020,
        "bbox": [-105.90, 40.50, -105.45, 40.80],
        "pre": ("2020-06-01", "2020-08-01"),
        "post": ("2020-10-20", "2020-12-01"),
        "blurb": "Largest wildfire in Colorado history — about 84,000 hectares in the "
                 "Arapaho and Roosevelt National Forests.",
    },
}

PIPELINE_STEPS = [
    "Prepare Data",
    "Fine-tune Prithvi",
    "Evaluate",
    "STAC Discovery",
    "Segment Tiles",
    "Mosaic",
]


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _open_thumb(local_dir: str):
    """Small preview written by segment_tile; fall back to the full-size render."""
    from PIL import Image

    for name in ("thumb.png", "burn.png"):
        p = os.path.join(local_dir, name)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    raise FileNotFoundError(f"no tile preview in {local_dir}")


def _read_scene(path: str):
    """Read a 6-band HLS GeoTIFF as (6, H, W) float32."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def _read_mask(path: str):
    """Read a single-band burn mask. Values: 1 burn, 0 unburned, -1 nodata."""
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.int16)
    return arr


def _scene_pairs(split_dir: str) -> list[tuple[str, str]]:
    """Pair each `*_merged.tif` scene with its `*.mask.tif` label."""
    from glob import glob

    pairs = []
    for scene in sorted(glob(os.path.join(split_dir, "*_merged.tif"))):
        mask = scene.replace("_merged.tif", ".mask.tif")
        if os.path.exists(mask):
            pairs.append((scene, mask))
    return pairs


class BurnScarDataset:
    """Minimal dataset — deliberately not torch.utils.data, so the tutorial has one less
    abstraction between the reader and the pixels."""

    def __init__(self, pairs, augment: bool = False):
        self.pairs = pairs
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        import numpy as np


        scene_path, mask_path = self.pairs[i]
        scene = normalize(_read_scene(scene_path))
        mask = _read_mask(mask_path)

        if self.augment:
            if np.random.rand() < 0.5:
                scene, mask = scene[:, :, ::-1].copy(), mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                scene, mask = scene[:, ::-1, :].copy(), mask[::-1, :].copy()

        return scene, mask


def _batches(dataset, batch_size: int, shuffle: bool = False):
    """Yield (x, y) torch tensors. `-1` nodata is carried through as a validity mask."""
    import numpy as np
    import torch

    idx = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        scenes, masks = [], []
        for i in chunk:
            s, m = dataset[int(i)]
            scenes.append(s)
            masks.append(m)
        x = torch.from_numpy(np.stack(scenes)).float()
        y = torch.from_numpy(np.stack(masks)).long()
        yield x, y


def _masked_losses(logits, target):
    """Dice + BCE over valid pixels only (nodata is -1 and must not train the model)."""
    import torch
    import torch.nn.functional as F

    logits = logits.squeeze(1)
    valid = (target >= 0).float()
    y = (target == 1).float()

    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1.0)

    probs = torch.sigmoid(logits) * valid
    y_v = y * valid
    inter = (probs * y_v).sum()
    denom = probs.sum() + y_v.sum()
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)

    return bce + dice


def _confusion(pred, target):
    """Return (tp, fp, fn, tn) over valid pixels."""
    valid = target >= 0
    p = (pred == 1) & valid
    t = (target == 1) & valid
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t & valid).sum())
    return tp, fp, fn, tn


def _metrics(tp, fp, fn, tn):
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "accuracy": acc}


# ------------------------------------------------------------------
# Task 1: Prepare data
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def prepare_data(max_scenes: int = 0) -> flyte.io.Dir:
    """
    Download and unpack HLS Burn Scars.

    Note this deliberately does NOT use `load_dataset()`. The repo is a script-based
    dataset (`hls_burn_scars.py`), and `datasets` removed loading-script support entirely
    in 4.0 — so the documented call fails on any current install. Pulling the tarball
    directly is both the working path and the simpler one.
    """
    import numpy as np
    from huggingface_hub import hf_hub_download

    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Preparing HLS Burn Scars</h2>"
        "<p>Downloading dataset archive (~2.6 GB) from the Hugging Face Hub…</p>"
    ), do_flush=True)

    archive = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset")

    work = tempfile.mkdtemp(prefix="burn_scars_")
    await flyte.report.replace.aio(rh.wrap_report(
        "<h2>Preparing HLS Burn Scars</h2><p>Extracting archive…</p>"
    ), do_flush=True)
    with tarfile.open(archive) as tar:
        tar.extractall(work, filter="data")

    # The archive lays out `training/` and `validation/`, sometimes nested one level deep.
    def _find(name):
        for root, dirs, _ in os.walk(work):
            if name in dirs:
                return os.path.join(root, name)
        return None

    train_dir, val_dir = _find("training"), _find("validation")
    if not train_dir or not val_dir:
        raise RuntimeError(f"Could not locate training/validation dirs under {work}")

    train_pairs = _scene_pairs(train_dir)
    val_pairs = _scene_pairs(val_dir)
    if max_scenes:
        train_pairs = train_pairs[:max_scenes]
        val_pairs = val_pairs[:max(2, max_scenes // 3)]

    log.info(f"Scenes: {len(train_pairs)} train / {len(val_pairs)} val")

    # Copy the selected scenes into the task's output Dir and index them by *relative*
    # path. Storing the extraction tempdir's absolute paths would work locally and then
    # fail on every remote run, because that directory only exists in this pod.
    import shutil

    out_dir = tempfile.mkdtemp(prefix="burn_scars_data_")
    index = {"train": [], "val": []}
    for split, pairs in (("train", train_pairs), ("val", val_pairs)):
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for scene_path, mask_path in pairs:
            scene_rel = os.path.join(split, os.path.basename(scene_path))
            mask_rel = os.path.join(split, os.path.basename(mask_path))
            shutil.copy2(scene_path, os.path.join(out_dir, scene_rel))
            shutil.copy2(mask_path, os.path.join(out_dir, mask_rel))
            index[split].append([scene_rel, mask_rel])

    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f)

    # ---- Report: class balance, band distributions, sample scenes ----
    sample = train_pairs[:6]
    burn_px = valid_px = 0
    band_values = [[] for _ in range(6)]
    cards = []

    for scene_path, mask_path in sample:
        scene = _read_scene(scene_path)
        mask = _read_mask(mask_path)
        burn_px += int((mask == 1).sum())
        valid_px += int((mask >= 0).sum())
        for b in range(6):
            band_values[b].append(scene[b][np.isfinite(scene[b])].ravel()[::37])

        fc = rh.scene_uri(scene, bands=(rh.SWIR2, rh.NIR, rh.RED))
        ov = rh.scene_uri(scene, mask=(mask == 1), bands=(rh.SWIR2, rh.NIR, rh.RED))
        frac = (mask == 1).sum() / max((mask >= 0).sum(), 1)
        cards.append(
            f'<div class="scene"><img src="{fc}">'
            f'<div class="cap">{os.path.basename(scene_path)[:26]}…</div></div>'
            f'<div class="scene"><img src="{ov}">'
            f'<div class="cap">MTBS ground truth · {frac:.1%} burned</div></div>'
        )

    hists = []
    for b in range(6):
        vals = np.concatenate(band_values[b]) if band_values[b] else np.array([0.0])
        counts, edges = np.histogram(vals, bins=48)
        hists.append(f'<div>{rh.make_histogram(list(counts), list(edges), title=BAND_NAMES[b])}</div>')

    burn_frac = burn_px / max(valid_px, 1)
    html = f"""
    <h2>HLS Burn Scars — Dataset</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_pairs)}</div><div class="label">Training scenes</div></div>
      <div class="stat"><div class="value">{len(val_pairs)}</div><div class="label">Validation scenes</div></div>
      <div class="stat"><div class="value">512&times;512</div><div class="label">Scene size</div></div>
      <div class="stat"><div class="value">6</div><div class="label">Spectral bands</div></div>
      <div class="stat"><div class="value">{burn_frac:.1%}</div><div class="label">Burned pixels (sample)</div></div>
      <div class="stat"><div class="value">CC-BY-4.0</div><div class="label">License</div></div>
    </div>

    <div class="note">
      Harmonized Landsat &amp; Sentinel-2 imagery over the contiguous US, 2018&ndash;2021.
      Labels come from <b>MTBS</b> (Monitoring Trends in Burn Severity) fire perimeters —
      these are real mapped fires, not synthetic annotations. Masks are
      <code>1</code> burn / <code>0</code> unburned / <code>-1</code> nodata; the nodata
      class is excluded from both the loss and every metric below.
    </div>

    <h3>Why six bands and not RGB</h3>
    <div class="card">
      Burned ground is <b>dark in near-infrared</b> (no live vegetation) and
      <b>bright in shortwave-infrared</b> (exposed char and bare soil). The
      SWIR2/NIR/Red composite below turns that contrast into an unmistakable orange-red
      signature, while healthy forest goes vivid green. In true colour the same scar is an
      ordinary brown smudge — which is why an RGB-only burn-scar dataset would be both
      easier to load and far less useful.
    </div>
    {rh.burn_legend_html()}
    <div class="scene-grid">{''.join(cards)}</div>

    <h3>Band reflectance distributions</h3>
    <div class="note">
      Surface reflectance scaled by 10,000. These distributions are what the Prithvi
      normalization constants assume, so a wildly different range here is the first sign
      the inputs need rescaling before they reach the encoder.
    </div>
    <div class="scene-grid">{''.join(hists)}</div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Fine-tune
# ------------------------------------------------------------------

# NOTE ON CACHING — deliberately off.
#
# `cache="auto"` here makes iteration much faster: a rerun that only touches downstream
# tasks skips training entirely. But a cached task does not execute its body, and this
# pipeline's reports are written *by* those bodies (`flyte.report.replace(...)`). A cache
# hit therefore returns the right outputs with an EMPTY report — and the reports are the
# product here, not a side effect. The failure is silent and easy to miss, because the run
# still goes green and the numbers are still correct.
#
# Turn caching on while debugging downstream stages, off for anything you intend to look
# at or record.
@gpu_env.task(report=True)
async def train(
    data_dir: flyte.io.Dir,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    freeze_encoder: bool = True,
) -> flyte.io.File:
    """
    Adapt Prithvi to burn-scar segmentation.

    By default the 300M-parameter encoder stays frozen and only the decoder trains. That
    is the honest foundation-model story — the representation was learned from 4.2M HLS
    scenes, and roughly 500 labelled ones are nowhere near enough to improve on it — and
    it keeps the run on a single mid-range GPU. Pass `freeze_encoder=False` for a full
    fine-tune when you have the headroom.
    """
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download


    local = await data_dir.download()
    with open(os.path.join(local, "index.json")) as f:
        index = json.load(f)

    def _abs(pairs):
        return [(os.path.join(local, s), os.path.join(local, m)) for s, m in pairs]

    train_ds = BurnScarDataset(_abs(index["train"]), augment=True)
    val_ds = BurnScarDataset(_abs(index["val"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device} | frozen encoder: {freeze_encoder}")

    await flyte.report.replace.aio(rh.wrap_report(
        f"<h2>Fine-tuning Prithvi-EO-2.0-300M</h2>"
        f"<p>Downloading pretrained encoder from <code>{MODEL_REPO}</code>…</p>"
    ), do_flush=True)

    ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)

    model = PrithviSegmenter(img_size=512, freeze_encoder=freeze_encoder)
    load_info = load_pretrained_encoder(model, ckpt_path)
    model = model.to(device)

    log.info(f"Encoder tensors loaded: {load_info['loaded']}/{load_info['target_tensors']}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    train_losses, val_losses, val_ious = [], [], []

    def _render(epoch_note: str):
        charts = ""
        if train_losses:
            charts += rh.make_line_chart(
                {"train loss": ("#ea580c", train_losses), "val loss": ("#0284c7", val_losses)},
                title="Loss (Dice + BCE)", y_label="loss",
            )
        if val_ious:
            charts += rh.make_line_chart(
                {"val IoU": ("#16a34a", val_ious)},
                title="Validation IoU (burn class)", y_label="IoU",
            )
        warn = ""
        if load_info["n_missing"]:
            warn = (f'<div class="note"><b>{load_info["n_missing"]}</b> encoder tensors were not '
                    f'found in the checkpoint and kept their random init — e.g. '
                    f'<code>{", ".join(load_info["missing"][:3])}</code>.</div>')
        return rh.wrap_report(f"""
          <h2>Fine-tuning Prithvi-EO-2.0-300M</h2>
          <div class="stat-grid">
            <div class="stat"><div class="value">{total/1e6:.0f}M</div><div class="label">Total params</div></div>
            <div class="stat"><div class="value">{trainable/1e6:.1f}M</div><div class="label">Trainable</div></div>
            <div class="stat"><div class="value">{'frozen' if freeze_encoder else 'full'}</div><div class="label">Encoder</div></div>
            <div class="stat"><div class="value">{len(train_ds)}</div><div class="label">Train scenes</div></div>
            <div class="stat"><div class="value">{load_info['loaded']}</div><div class="label">Pretrained tensors</div></div>
            <div class="stat"><div class="value">{device}</div><div class="label">Device</div></div>
          </div>
          {warn}
          <p>{epoch_note}</p>
          <div class="chart-container">{charts}</div>
          <div class="note">
            Only <b>{trainable/1e6:.1f}M</b> of <b>{total/1e6:.0f}M</b> parameters are being
            updated. The encoder's representation came from self-supervised pretraining on
            millions of HLS scenes; with a few hundred labelled examples the decoder is the
            part worth training.
          </div>
        """)

    await flyte.report.replace.aio(_render("Starting training…"), do_flush=True)

    best_iou, best_state = -1.0, None
    for epoch in range(epochs):
        model.train()
        # BatchNorm in the decoder is fine, but a frozen encoder should stay in eval mode
        # so its (unused) dropout/attention behave deterministically.
        if freeze_encoder:
            model.encoder.eval()

        running, nb = 0.0, 0
        for x, y in _batches(train_ds, batch_size, shuffle=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = _masked_losses(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            nb += 1
        scheduler.step()
        train_losses.append(running / max(nb, 1))

        # ---- validation ----
        model.eval()
        v_running, v_nb = 0.0, 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for x, y in _batches(val_ds, batch_size):
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = _masked_losses(logits, y)
                v_running += float(loss.item())
                v_nb += 1
                pred = (torch.sigmoid(logits.squeeze(1)) > 0.5).long().cpu().numpy()
                a, b, c, d = _confusion(pred, y.cpu().numpy())
                tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d

        val_losses.append(v_running / max(v_nb, 1))
        m = _metrics(tp, fp, fn, tn)
        val_ious.append(m["iou"])

        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in split_state_dict(model, freeze_encoder).items()
            }

        log.info(f"epoch {epoch+1}/{epochs} train={train_losses[-1]:.4f} "
                 f"val={val_losses[-1]:.4f} IoU={m['iou']:.4f}")
        await flyte.report.replace.aio(
            _render(f"Epoch {epoch+1}/{epochs} — val IoU {m['iou']:.4f} (best {best_iou:.4f})"),
            do_flush=True,
        )

    out = os.path.join(tempfile.mkdtemp(prefix="burn_ckpt_"), "prithvi_burnscar.pt")
    torch.save(
        {
            # Decoder-only when the encoder is frozen: ~6 MB instead of ~1.2 GB. Every
            # downstream tile task pulls this artifact, so its size is multiplied by the
            # width of the fan-out.
            "state_dict": (best_state if best_state is not None
                           else split_state_dict(model, freeze_encoder)),
            "freeze_encoder": freeze_encoder,
            "best_iou": best_iou,
            "epochs": epochs,
            "history": {"train_loss": train_losses, "val_loss": val_losses, "val_iou": val_ious},
        },
        out,
    )

    await flyte.report.replace.aio(
        _render(f"Training complete — best validation IoU <b>{best_iou:.4f}</b>"), do_flush=True
    )
    return await flyte.io.File.from_local(out)


# ------------------------------------------------------------------
# Task 3: Evaluate
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    data_dir: flyte.io.Dir,
    checkpoint: flyte.io.File,
    demo_scenes: int = 4,
) -> str:
    """Score the tuned model on the validation split and show where it wins and loses."""
    import numpy as np
    import torch


    local = await data_dir.download()
    with open(os.path.join(local, "index.json")) as f:
        index = json.load(f)
    val_pairs = [(os.path.join(local, s), os.path.join(local, m)) for s, m in index["val"]]

    ckpt_local = await checkpoint.download()
    ckpt = torch.load(ckpt_local, map_location="cpu", weights_only=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Rebuilds the frozen encoder from the Hub when the checkpoint is decoder-only.
    model = load_segmenter(ckpt, img_size=512).to(device).eval()

    await flyte.report.replace.aio(rh.wrap_report(
        f"<h2>Evaluation</h2><p>Scoring {len(val_pairs)} validation scenes…</p>"
    ), do_flush=True)


    per_scene = []
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for scene_path, mask_path in val_pairs:
            scene = _read_scene(scene_path)
            mask = _read_mask(mask_path)
            x = torch.from_numpy(normalize(scene)).float().unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits.squeeze()).cpu().numpy() > 0.5).astype(np.int16)

            a, b, c, d = _confusion(pred, mask)
            tp, fp, fn, tn = tp + a, fp + b, fn + c, tn + d
            sm = _metrics(a, b, c, d)
            per_scene.append({
                "scene": scene_path,
                "mask": mask_path,
                "iou": sm["iou"],
                "truth_frac": float((mask == 1).sum() / max((mask >= 0).sum(), 1)),
            })

    overall = _metrics(tp, fp, fn, tn)
    scored = [s for s in per_scene if s["truth_frac"] > 0.001]
    scored.sort(key=lambda s: s["iou"], reverse=True)
    best = scored[:demo_scenes]
    worst = scored[-demo_scenes:][::-1] if len(scored) > demo_scenes else []

    def _cards(items, heading):
        if not items:
            return ""
        out = []
        with torch.no_grad():
            for s in items:
                scene = _read_scene(s["scene"])
                mask = _read_mask(s["mask"])
                x = torch.from_numpy(normalize(scene)).float().unsqueeze(0).to(device)
                pred = (torch.sigmoid(model(x).squeeze()).cpu().numpy() > 0.5)

                fc = rh.composite(scene)
                pred_img = rh.overlay(fc, pred, color=rh.BURN_COLOR)
                pred_img = rh.outline(pred_img, mask == 1, color=rh.TRUTH_COLOR)
                badge = ("badge-success" if s["iou"] > 0.6
                         else "badge-warning" if s["iou"] > 0.3 else "badge-danger")
                out.append(
                    f'<div class="scene"><img src="{rh.to_png_uri(fc)}">'
                    f'<div class="cap">SWIR2/NIR/Red composite</div></div>'
                    f'<div class="scene"><img src="{rh.to_png_uri(pred_img)}">'
                    f'<div class="cap">Prediction (fill) vs truth (outline) &nbsp;'
                    f'<span class="badge {badge}">IoU {s["iou"]:.3f}</span></div></div>'
                )
        return f"<h3>{heading}</h3>{rh.burn_legend_html()}<div class='scene-grid'>{''.join(out)}</div>"

    history = ckpt.get("history", {})
    curve = ""
    if history.get("val_iou"):
        curve = rh.make_line_chart(
            {"val IoU": ("#16a34a", history["val_iou"])},
            title="Validation IoU across training", y_label="IoU",
        )

    html = f"""
    <h2>Evaluation — Burn Scar Segmentation</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{overall['iou']:.3f}</div><div class="label">IoU (burn)</div></div>
      <div class="stat"><div class="value">{overall['dice']:.3f}</div><div class="label">Dice</div></div>
      <div class="stat"><div class="value">{overall['precision']:.3f}</div><div class="label">Precision</div></div>
      <div class="stat"><div class="value">{overall['recall']:.3f}</div><div class="label">Recall</div></div>
      <div class="stat"><div class="value">{overall['accuracy']:.3f}</div><div class="label">Pixel accuracy</div></div>
      <div class="stat"><div class="value">{len(val_pairs)}</div><div class="label">Scenes scored</div></div>
    </div>

    <div class="note">
      Pixel accuracy is the least useful number here and is shown only to make that point:
      roughly 89% of labelled pixels are unburned, so a model that predicts "nothing burned"
      everywhere still scores ~0.89. <b>IoU on the burn class</b> is the metric that moves.
    </div>

    <div class="chart-container">
      {rh.make_bar_chart(
        ["IoU", "Dice", "Precision", "Recall"],
        [overall['iou'], overall['dice'], overall['precision'], overall['recall']],
        colors=["#ea580c", "#f97316", "#0284c7", "#16a34a"],
        title="Segmentation quality (burn class)", y_max=1.0)}
    </div>
    {f'<div class="chart-container">{curve}</div>' if curve else ''}

    {_cards(best, "Best scenes")}
    {_cards(worst, "Hardest scenes")}

    <div class="note">
      Failure modes worth noticing in the hard cases: recently harvested farmland and bare
      rock share the low-NIR / high-SWIR signature of a burn scar, and cloud shadow can
      mimic it too. This is exactly where the multi-temporal comparison in the mosaic stage
      earns its keep — a real scar is present after the fire and absent before it.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    return json.dumps({
        "overall": overall,
        "n_scenes": len(val_pairs),
        "best_iou": ckpt.get("best_iou"),
    })


# ------------------------------------------------------------------
# Task 4: STAC discovery
# ------------------------------------------------------------------

def _mgrs(feature) -> str:
    """MGRS tile id, e.g. 'MGRS-10TFK'. Scenes sharing this share an identical pixel grid."""
    return feature["properties"].get("grid:code", "")


def _stac_search(bbox, start, end, limit=60):
    """Query the public Element84 catalog. No auth, no key."""
    body = json.dumps({
        "collections": [STAC_COLLECTION],
        "bbox": bbox,
        # Full RFC3339 is required — a bare `YYYY-MM-DD` range returns HTTP 400.
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": limit,
    }).encode()
    req = urllib.request.Request(
        STAC_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.load(resp)
    feats = payload.get("features", [])
    # The `query` extension 400s on this endpoint, so filter cloud cover client-side.
    return sorted(feats, key=lambda f: f["properties"].get("eo:cloud_cover", 100.0))


@cpu_env.task(report=True)
async def discover_tiles(
    aoi: str = "dixie",
    tile_px: int = 256,
    tile_limit: int = 64,
) -> str:
    """
    Query a live STAC catalog for the AOI and lay out a tile grid.

    This is the step that turns a named place into a work list. Everything downstream is
    per-tile and embarrassingly parallel.
    """
    if aoi not in AOIS:
        raise ValueError(f"Unknown AOI '{aoi}'. Available: {', '.join(AOIS)}")
    spec = AOIS[aoi]

    await flyte.report.replace.aio(rh.wrap_report(
        f"<h2>STAC Discovery — {spec['name']}</h2><p>Querying the Element84 catalog…</p>"
    ), do_flush=True)

    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    pre = _stac_search(spec["bbox"], *spec["pre"])
    post = _stac_search(spec["bbox"], *spec["post"])
    if not pre or not post:
        raise RuntimeError(
            f"STAC returned {len(pre)} pre-fire and {len(post)} post-fire scenes for {aoi}."
        )

    # Pick the clearest post-fire scene, then pick the clearest pre-fire scene from the
    # SAME MGRS tile. Same MGRS tile => identical pixel grid, so a window read at (x, y)
    # refers to the exact same ground in both scenes and dNBR lines up pixel-for-pixel.
    post_scene = post[0]
    tile_id = _mgrs(post_scene)
    pre_same = [f for f in pre if _mgrs(f) == tile_id]
    if not pre_same:
        raise RuntimeError(
            f"No pre-fire scene shares MGRS tile {tile_id} with the post-fire scene. "
            f"Pre-fire tiles available: {sorted({_mgrs(f) for f in pre})}"
        )
    pre_scene = pre_same[0]

    def _hrefs(feature):
        assets = feature["assets"]
        missing = [b for b in S2_ASSETS if b not in assets]
        if missing:
            raise RuntimeError(f"Scene {feature['id']} is missing bands: {missing}")
        return [assets[b]["href"] for b in S2_ASSETS]

    # Restrict tiling to the AOI, not the whole 110km granule. Reproject the lon/lat bbox
    # into the scene's UTM CRS and turn it into a pixel window; the fire lives inside it.
    # Tiling the granule centre instead would usually miss the burn entirely.
    with rasterio.open(_hrefs(post_scene)[0]) as src:
        width, height = src.width, src.height
        l, b, r_, t = transform_bounds("EPSG:4326", src.crs, *spec["bbox"])
        win = from_bounds(l, b, r_, t, src.transform)

    x0 = max(0, int(win.col_off))
    y0 = max(0, int(win.row_off))
    x1 = min(width, int(win.col_off + win.width))
    y1 = min(height, int(win.row_off + win.height))
    cols = max(1, (x1 - x0) // tile_px)
    rows = max(1, (y1 - y0) // tile_px)

    # Tiles are defined in WORLD coordinates, not pixel offsets.
    #
    # Sentinel-2 bands do not share a pixel grid: blue/green/red are 10 m (10980x10980)
    # while NIR/SWIR1/SWIR2 are 20 m (5490x5490). Reusing one pixel window across all six
    # silently reads a 2x larger ground footprint from the 20 m bands, and runs off their
    # edge entirely past row 5490 — which fills NIR and SWIR with zeros while the 10 m
    # bands still return data. Storing CRS bounds lets each band resolve its own window and
    # resample onto a common grid, which is the only way the six bands line up.
    with rasterio.open(_hrefs(post_scene)[0]) as src:
        transform = src.transform
        crs_name = src.crs.to_string()

    tiles = []
    for r in range(rows):
        for c in range(cols):
            px, py = x0 + c * tile_px, y0 + r * tile_px
            left, top = transform * (px, py)
            right, bottom = transform * (px + tile_px, py + tile_px)
            tiles.append({
                "row": r, "col": c, "size": tile_px,
                "bounds": [left, bottom, right, top],
            })

    # If limiting, keep a centred square block of the AOI grid — the burn is usually near
    # the middle of a well-chosen AOI, and a square reads best in the mosaic.
    if tile_limit and len(tiles) > tile_limit:
        side = max(1, int(tile_limit ** 0.5))
        r0, c0 = max(0, rows // 2 - side // 2), max(0, cols // 2 - side // 2)
        tiles = [t for t in tiles if r0 <= t["row"] < r0 + side and c0 <= t["col"] < c0 + side]
        for t in tiles:
            t["row"] -= r0
            t["col"] -= c0
        cols = side

    # One contrast stretch for the whole AOI, computed from a coarse overview of each band.
    # Every tile then renders with the same mapping, so the stitched mosaic reads as a
    # single image. Stretching per tile instead normalises each one independently — a
    # shaded tile and a sunlit tile both come out mid-grey — and the tile grid shows up as
    # seams across the mosaic.
    def _stretch_ranges(hrefs):
        import numpy as np

        # Coarse read of the whole AOI: cheap over HTTP, and plenty for percentiles.
        left, top = transform * (x0, y0)
        right, bottom = transform * (x1, y1)

        ranges = []
        for href in hrefs:
            with rasterio.open(href) as src:
                win = from_bounds(left, bottom, right, top, src.transform)
                arr = src.read(1, window=win, out_shape=(256, 256),
                               boundless=True, fill_value=0).astype("float32")
            valid = arr[arr > 0]
            if valid.size < 32:
                ranges.append([0.0, 1.0])
            else:
                lo, hi = np.percentile(valid, [2, 98])
                ranges.append([float(lo), float(max(hi, lo + 1))])
        return ranges

    post_stretch = _stretch_ranges(_hrefs(post_scene))
    pre_stretch = _stretch_ranges(_hrefs(pre_scene))

    plan = {
        "aoi": aoi,
        "name": spec["name"],
        "post_stretch": post_stretch,
        "pre_stretch": pre_stretch,
        "blurb": spec["blurb"],
        "bbox": spec["bbox"],
        "mgrs": tile_id,
        "pre": {"id": pre_scene["id"], "date": pre_scene["properties"]["datetime"][:10],
                "cloud": pre_scene["properties"].get("eo:cloud_cover", 0.0),
                "hrefs": _hrefs(pre_scene)},
        "post": {"id": post_scene["id"], "date": post_scene["properties"]["datetime"][:10],
                 "cloud": post_scene["properties"].get("eo:cloud_cover", 0.0),
                 "hrefs": _hrefs(post_scene)},
        "tiles": tiles,
        "cols": cols,
        "tile_px": tile_px,
        "crs": crs_name,
        "aoi_window": [x0, y0, x1 - x0, y1 - y0],
        "scene_size": [width, height],
        "n_candidates": {"pre": len(pre), "post": len(post)},
    }

    html = f"""
    <h2>STAC Discovery — {spec['name']}</h2>
    <div class="card">{spec['blurb']}</div>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(pre)}</div><div class="label">Pre-fire candidates</div></div>
      <div class="stat"><div class="value">{len(post)}</div><div class="label">Post-fire candidates</div></div>
      <div class="stat"><div class="value">{len(tiles)}</div><div class="label">Tiles to segment</div></div>
      <div class="stat"><div class="value">{tile_px}px</div><div class="label">Tile size</div></div>
      <div class="stat"><div class="value">{width}&times;{height}</div><div class="label">Scene raster</div></div>
    </div>
    <table>
      <tr><th>Role</th><th>Scene ID</th><th>Date</th><th>Cloud cover</th><th>MGRS tile</th></tr>
      <tr><td>Pre-fire</td><td><code>{pre_scene['id']}</code></td>
          <td>{plan['pre']['date']}</td><td>{plan['pre']['cloud']:.1f}%</td><td>{plan['mgrs']}</td></tr>
      <tr><td>Post-fire</td><td><code>{post_scene['id']}</code></td>
          <td>{plan['post']['date']}</td><td>{plan['post']['cloud']:.1f}%</td><td>{plan['mgrs']}</td></tr>
    </table>
    <div class="note">
      Scenes come from the public <b>Element84 earth-search</b> catalog — no account, no API
      key. Both scenes are the same MGRS tile (<b>{plan['mgrs']}</b>), so their pixel grids
      align exactly and the pre/post comparison is valid pixel-for-pixel. Tiles are
      <i>windows</i> into the AOI region of cloud-optimized GeoTIFFs, so each downstream
      task range-reads only its own few megabytes instead of pulling a full 110&nbsp;km scene.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)
    return json.dumps(plan)


# ------------------------------------------------------------------
# Task 5: Segment one tile  (this is what fans out)
# ------------------------------------------------------------------

# Cache the loaded segmenter at module scope. Under `ReusePolicy` the container — and this
# Python process — is reused across tiles, so the model is built once per replica instead
# of once per tile. Keyed by input size because the ViT builds its position embedding grid
# from it.
#
# The lock is load-bearing, not defensive. `concurrency > 1` means several tile coroutines
# run in this same process; without it they all fail the `not in cache` check together and
# each builds its own 300M model, which is an immediate OOM rather than a slow path. The
# built model is then shared across coroutines, which is safe: inference is in eval mode
# under no_grad, so the forward pass mutates no module state.
_MODEL_CACHE: dict = {}
_MODEL_LOCK = asyncio.Lock()


async def _get_tile_model(ckpt_path: str, size: int):
    import torch

    from model import load_segmenter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (ckpt_path, size, device)

    async with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = load_segmenter(ckpt, img_size=size).to(device)
            _MODEL_CACHE.clear()  # only ever keep one — these are large
            _MODEL_CACHE[key] = (model, device)
        return _MODEL_CACHE[key]


@tile_env.task(retries=3)
async def segment_tile(
    tile_index: int, plan_json: str, checkpoint: flyte.io.File
) -> flyte.io.Dir:
    """
    Read one tile window from the COGs, segment it, return a thumbnail plus stats.

    Small, cheap, independently retryable — hundreds of these run at once. Interruption of
    a single tile costs one retry, not the run.
    """
    import numpy as np
    import rasterio
    import torch
    from rasterio.windows import from_bounds

    plan = json.loads(plan_json)
    tile = plan["tiles"][tile_index]
    size = tile["size"]
    bounds = tile["bounds"]

    def _read(hrefs):
        """Read the same patch of *ground* from every band, onto a common size x size grid.

        The window is derived per band from world bounds rather than shared as pixel
        offsets, because Sentinel-2 bands do not share a pixel grid: blue/green/red are
        10 m (10980x10980) and NIR/SWIR1/SWIR2 are 20 m (5490x5490). Reusing one pixel
        window reads a 2x larger ground footprint from the 20 m bands and runs off their
        edge past row 5490, filling NIR and SWIR with zeros while the 10 m bands still
        return data — which renders as a solid blue lower half and feeds the model
        garbage. `out_shape` resamples every band onto the same grid so the six align.
        """
        bands = []
        for href in hrefs:
            with rasterio.open(href) as src:
                win = from_bounds(*bounds, src.transform)
                arr = src.read(
                    1, window=win, out_shape=(size, size),
                    boundless=True, fill_value=0,
                ).astype("float32")
            bands.append(arr)
        return np.stack(bands)

    post = _read(plan["post"]["hrefs"])
    pre = _read(plan["pre"]["hrefs"])

    ckpt_local = await checkpoint.download()
    model, device = await _get_tile_model(ckpt_local, size)

    with torch.no_grad():
        x = torch.from_numpy(normalize(post)).float().unsqueeze(0).to(device)
        pred = (torch.sigmoid(model(x).squeeze()).cpu().numpy() > 0.5)

    # dNBR — the standard pre/post burn index — as an independent cross-check on the model.
    def _nbr(arr):
        nir, swir2 = arr[3], arr[5]
        return (nir - swir2) / np.clip(nir + swir2, 1e-6, None)

    dnbr = float(np.nanmean(_nbr(pre) - _nbr(post)))

    # Shared AOI-wide stretch so tiles stitch seamlessly (see discover_tiles).
    post_rng = plan.get("post_stretch")
    pre_rng = plan.get("pre_stretch")
    fc = rh.composite(post, ranges=post_rng)
    thumb = rh.overlay(fc, pred, color=rh.BURN_COLOR, alpha=0.5)

    # Return rendered rasters as *files*, not as base64 in the return value.
    #
    # Task inputs/outputs are passed inline and capped (10 MB by default). Three base64
    # PNGs per tile across a 36-tile fan-out is ~16 MB — base64 alone adds 33% — and the
    # mosaic task fails with InlineIOMaxBytesBreached *after* every tile has already done
    # its work. Writing to a Dir hands the blobs to object storage and passes a reference,
    # which is both the fix and the only thing that scales: the whole point of this
    # pipeline is that the tile count can grow without the plumbing changing.
    from PIL import Image

    out_dir = tempfile.mkdtemp(prefix=f"tile_{tile['row']}_{tile['col']}_")
    burn_im = Image.fromarray(thumb)
    burn_im.save(os.path.join(out_dir, "burn.png"), optimize=True)
    Image.fromarray(rh.composite(pre, ranges=pre_rng)).save(
        os.path.join(out_dir, "pre.png"), optimize=True)
    # A small copy for the live progress grid. The pipeline re-flushes its report on every
    # completed tile, so the per-flush payload is what matters: at full resolution the
    # growing grid would ship tens of megabytes over the course of a run.
    burn_im.resize((96, 96)).save(os.path.join(out_dir, "thumb.png"), optimize=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump({
            "row": tile["row"],
            "col": tile["col"],
            "state": "done",
            "burn_frac": float(pred.mean()),
            "dnbr": dnbr,
        }, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 6: Mosaic
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def mosaic(plan_json: str, tile_dirs: list[flyte.io.Dir]) -> flyte.io.Dir:
    """Stitch tile results into one map product plus the before/after wipe."""
    import numpy as np
    from PIL import Image

    plan = json.loads(plan_json)

    # Pull each tile's rasters down from object storage. They arrived as references, so
    # the fan-out width never inflates the task's inline payload.
    results = []
    for d in tile_dirs:
        local = await d.download()
        with open(os.path.join(local, "meta.json")) as f:
            meta = json.load(f)
        meta["burn_img"] = Image.open(os.path.join(local, "burn.png")).convert("RGB")
        meta["pre_img"] = Image.open(os.path.join(local, "pre.png")).convert("RGB")
        meta["uri"] = rh.image_to_uri(meta["burn_img"])
        results.append(meta)

    cols = plan["cols"]
    rows = max((r["row"] for r in results), default=0) + 1

    burn_fracs = [r["burn_frac"] for r in results]
    burned_tiles = [r for r in results if r["burn_frac"] > 0.02]
    mean_burn = float(np.mean(burn_fracs)) if burn_fracs else 0.0

    # Each Sentinel-2 pixel is 10x10 m.
    px_per_tile = plan["tile_px"] ** 2
    burned_px = sum(r["burn_frac"] * px_per_tile for r in results)
    burned_ha = burned_px * 100 / 10_000

    grid = rh.mosaic_grid_html(results, cols=cols)

    # Assemble full-AOI mosaics for the wipe by stitching the per-tile thumbnails back
    # into their grid positions.
    def _stitch(key):
        canvas = None
        for r in results:
            img = r[key]
            if canvas is None:
                cell = img.size[0]
                canvas = Image.new("RGB", (cols * cell, rows * cell))
            canvas.paste(img, (r["col"] * img.size[0], r["row"] * img.size[1]))
        return canvas

    pre_img = _stitch("pre_img")
    burn_img = _stitch("burn_img")

    wipe = ""
    if pre_img is not None and burn_img is not None:
        wipe = rh.wipe_html(
            rh.image_to_uri(pre_img), rh.image_to_uri(burn_img),
            f"Pre-fire · {plan['pre']['date']}",
            f"Post-fire + prediction · {plan['post']['date']}",
            slug="mosaic",
        )

    top = sorted(results, key=lambda r: r["burn_frac"], reverse=True)[:6]
    rows_html = "".join(
        f"<tr><td>r{r['row']}c{r['col']}</td><td>{r['burn_frac']:.1%}</td>"
        f"<td>{r['dnbr']:+.3f}</td></tr>" for r in top
    )

    html = f"""
    <h2>Mosaic — {plan['name']}</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(results)}</div><div class="label">Tiles segmented</div></div>
      <div class="stat"><div class="value">{len(burned_tiles)}</div><div class="label">Tiles with burn</div></div>
      <div class="stat"><div class="value">{mean_burn:.1%}</div><div class="label">Mean burned area</div></div>
      <div class="stat"><div class="value">{burned_ha:,.0f}</div><div class="label">Hectares burned (est.)</div></div>
    </div>

    <h3>Before / after</h3>
    <div class="note">
      Left is the pre-fire scene, right is post-fire with the model's burn mask in red.
      Both are SWIR2/NIR/Red composites built from the same Sentinel-2 tiles, rendered with
      a single AOI-wide contrast stretch so the tiles stitch without visible seams.
    </div>
    {rh.burn_legend_html()}
    {wipe}

    <h3>Tile fan-out</h3>
    {grid}

    <h3>Most-burned tiles</h3>
    <table>
      <tr><th>Tile</th><th>Burned fraction</th><th>dNBR</th></tr>
      {rows_html}
    </table>
    <div class="note">
      <b>dNBR</b> (differenced Normalized Burn Ratio) is the standard index-based burn
      metric, computed here from the pre- and post-fire NIR/SWIR bands. It is calculated
      independently of the model, so agreement between a high burned fraction and a high
      dNBR is genuine corroboration rather than the model marking its own homework.
      Hectares are estimated from 10&nbsp;m Sentinel-2 pixels and are indicative only —
      the tile grid is clipped to a subset of the scene.
    </div>
    """
    await flyte.report.replace.aio(rh.wrap_report(html), do_flush=True)

    # Hand the stitched rasters back as a Dir so the *pipeline* can show the finished
    # mosaic in its own report. Without this the result only exists inside this task's
    # report, and anyone who sat watching the fan-out fill in the pipeline report sees it
    # replaced by summary statistics and a note telling them to go look somewhere else.
    # (A Dir, not inline strings — full-AOI PNGs would blow the 10 MB inline IO limit.)
    out_dir = tempfile.mkdtemp(prefix="mosaic_")
    summary = {
        "aoi": plan["aoi"],
        "name": plan["name"],
        "n_tiles": len(results),
        "burned_tiles": len(burned_tiles),
        "mean_burn_frac": mean_burn,
        "burned_hectares": burned_ha,
        "pre_date": plan["pre"]["date"],
        "post_date": plan["post"]["date"],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f)
    if pre_img is not None:
        pre_img.save(os.path.join(out_dir, "pre.png"), optimize=True)
    if burn_img is not None:
        burn_img.save(os.path.join(out_dir, "burn.png"), optimize=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    aoi: str = "dixie",
    max_scenes: int = 0,
    epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    freeze_encoder: bool = True,
    tile_px: int = 256,
    tile_limit: int = 64,
) -> tuple[str, str]:
    """
    End-to-end: fine-tune Prithvi on labelled burn scars, then map a real fire.

    Returns (evaluation JSON, mosaic JSON).
    """
    async def step(n: int, note: str):
        await flyte.report.replace.aio(
            rh.wrap_report(f"<h2>Burn Scar Mapping</h2>{rh.progress_html(PIPELINE_STEPS, n, note)}"),
            do_flush=True,
        )

    await step(1, "Downloading and inspecting HLS Burn Scars…")
    data_dir = await prepare_data(max_scenes=max_scenes)

    await step(2, "Fine-tuning Prithvi-EO-2.0-300M…")
    checkpoint = await train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_encoder=freeze_encoder,
    )

    await step(3, "Scoring the validation split…")
    eval_json = await evaluate(data_dir=data_dir, checkpoint=checkpoint)

    await step(4, f"Querying STAC for the {AOIS[aoi]['name']} area of interest…")
    plan_json = await discover_tiles(aoi=aoi, tile_px=tile_px, tile_limit=tile_limit)
    plan = json.loads(plan_json)

    n_tiles = len(plan["tiles"])
    await step(5, f"Segmenting {n_tiles} tiles in parallel…")

    # ---- live mosaic: render tiles into the report as they land ----
    #
    # `asyncio.as_completed` rather than `flyte.map` specifically for the visual.
    # `flyte.map` yields "in the order of the inputs, regardless of the order in which the
    # individual actions finish", which fills the grid in a tidy raster scan and makes a
    # heavily parallel run look sequential. `as_completed` streams results the moment each
    # action finishes, so the mosaic pops in out of order — which is what genuinely
    # parallel work looks like. We give up flyte.map's built-in exception capture, so each
    # await is guarded individually to keep one bad tile from killing the run.
    #
    # Re-flushing the report inside the loop is what makes it live: reports stream to the
    # UI whenever `do_flush=True`, so the grid visibly fills during the run instead of
    # appearing all at once at the end.
    grid = [
        {"row": t["row"], "col": t["col"], "state": "pending"} for t in plan["tiles"]
    ]
    by_pos = {(g["row"], g["col"]): g for g in grid}

    async def _render_live(done: int, failed: int):
        # The tile grid goes FIRST, above everything else.
        #
        # This report re-renders on every completed tile. Anything tall placed above the
        # grid — a heading, the six-step progress strip, stat tiles — pushes it below the
        # fold, and each refresh snaps the reader back to the top, so the fill is never
        # actually watchable. Keeping the grid at the top means you can leave it open and
        # watch the mosaic assemble; the narrative detail sits underneath.
        pct = (done + failed) / max(n_tiles, 1) * 100
        status = (
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
            f'margin:0 0 8px;font-size:.92em;">'
            f'<b style="color:#9a3412;">Segmenting tiles — {plan["name"]}</b>'
            f'<span style="color:#6c757d;">{done}/{n_tiles} done'
            + (f' · <span style="color:#b91c1c;">{failed} failed</span>' if failed else "")
            + f' · {pct:.0f}%</span></div>'
        )
        await flyte.report.replace.aio(
            rh.wrap_report(
                status
                + rh.mosaic_grid_html(grid, cols=plan["cols"])
                + rh.progress_html(PIPELINE_STEPS, 5)
            ),
            do_flush=True,
        )

    await _render_live(0, 0)

    tile_results, failed, done = [], 0, 0

    # `flyte.group` collapses the fan-out into a single expandable folder in the run tree.
    # At 144 tiles the ungrouped view is 144 sibling nodes, which buries the four tasks
    # that actually differ from one another. Grouping is presentation-only — the tiles
    # still run in parallel exactly as before — but it also gives an aggregated
    # success/failure count on hover, which is the thing you want at a glance.
    #
    # The whole loop is inside the context, not just the coroutine construction, because
    # the task invocation is what gets tagged.
    with flyte.group("segment-tiles"):
        coros = [
            segment_tile(tile_index=i, plan_json=plan_json, checkpoint=checkpoint)
            for i in range(n_tiles)
        ]
        for fut in asyncio.as_completed(coros):
            try:
                tile_dir = await fut
            except Exception as e:  # noqa: BLE001 — one tile must not sink the mosaic
                failed += 1
                log.warning(f"tile failed: {e}")
                await _render_live(done, failed)
                continue

            tile_results.append(tile_dir)
            done += 1
            # Pull just the small thumbnail so the growing report stays light.
            try:
                local = await tile_dir.download()
                with open(os.path.join(local, "meta.json")) as f:
                    meta = json.load(f)
                cell = by_pos.get((meta["row"], meta["col"]))
                if cell is not None:
                    cell.update(
                        state="done",
                        burn_frac=meta["burn_frac"],
                        uri=rh.image_to_uri(_open_thumb(local)),
                    )
            except Exception as e:  # noqa: BLE001 — preview only; never fail the run
                log.warning(f"tile preview unavailable: {e}")

            await _render_live(done, failed)

    if not tile_results:
        raise RuntimeError(f"All {n_tiles} tile tasks failed; nothing to mosaic.")
    if failed:
        log.warning(f"{failed}/{n_tiles} tiles failed; mosaicking the remaining "
                    f"{len(tile_results)}.")

    # Keep the completed grid on screen while the mosaic runs, rather than blanking it.
    await _render_live(done, failed)
    mosaic_dir = await mosaic(plan_json=plan_json, tile_dirs=tile_results)

    # Pull the stitched result back and show it *here* — the fan-out filled in on this
    # report, so the finished map belongs on this report too.
    local = await mosaic_dir.download()
    with open(os.path.join(local, "summary.json")) as f:
        mo = json.load(f)

    wipe = ""
    pre_p, burn_p = os.path.join(local, "pre.png"), os.path.join(local, "burn.png")
    if os.path.exists(pre_p) and os.path.exists(burn_p):
        from PIL import Image

        wipe = rh.wipe_html(
            rh.image_to_uri(Image.open(pre_p).convert("RGB")),
            rh.image_to_uri(Image.open(burn_p).convert("RGB")),
            f"Pre-fire · {mo['pre_date']}",
            f"Post-fire + prediction · {mo['post_date']}",
            slug="final",
            height=520,
        )

    ev = json.loads(eval_json)
    await flyte.report.replace.aio(rh.wrap_report(f"""
      <h3 style="margin-top:0;">{mo['name']} — burn scar map</h3>
      {rh.burn_legend_html()}
      {wipe}
      <div class="stat-grid">
        <div class="stat"><div class="value">{mo['burned_hectares']:,.0f}</div><div class="label">Hectares burned (est.)</div></div>
        <div class="stat"><div class="value">{mo['burned_tiles']}/{mo['n_tiles']}</div><div class="label">Tiles with burn</div></div>
        <div class="stat"><div class="value">{mo['mean_burn_frac']:.1%}</div><div class="label">Mean burned area</div></div>
        <div class="stat"><div class="value">{ev['overall']['iou']:.3f}</div><div class="label">Validation IoU</div></div>
        <div class="stat"><div class="value">{ev['overall']['dice']:.3f}</div><div class="label">Dice</div></div>
        <div class="stat"><div class="value">{AOIS[aoi]['year']}</div><div class="label">Fire year</div></div>
      </div>
      <h3>Tile fan-out</h3>
      {rh.mosaic_grid_html(grid, cols=plan["cols"])}
      <div class="card"><b>AOI:</b> {plan['name']} &nbsp;|&nbsp;
        <b>Pre-fire:</b> {mo['pre_date']} &nbsp;|&nbsp;
        <b>Post-fire:</b> {mo['post_date']} &nbsp;|&nbsp;
        <b>Encoder:</b> {'frozen' if freeze_encoder else 'fully fine-tuned'}</div>
      <div class="note">
        Individual task reports have the dataset explorer, training curves, best/worst
        scene comparisons, and the per-tile dNBR table.
      </div>
    """), do_flush=True)

    return eval_json, json.dumps(mo)
