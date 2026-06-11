"""
3D Brain Tumor Segmentation — Train SegResNet on BraTS 2023 MRI volumes.

Pipeline: download the BraTS 2023 GLI dataset (1,251 multi-modal MRI scans),
train a SegResNet for 3D brain tumor segmentation using MONAI, evaluate with
Dice scores and Hausdorff distance, and render multi-plane tumor overlays
(axial/sagittal/coronal) with color-coded tumor subregions.

Tumor subregions:
  - Enhancing Tumor (ET) — active tumor growth (red)
  - Necrotic Core (NCR) — dead tissue at tumor center (yellow)
  - Peritumoral Edema (ED) — swelling around tumor (green)

Usage:
    # Default (SegResNet on BraTS 2023, subset of 100 cases)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --max_cases 20 --epochs 5

    # Remote (full dataset)
    flyte run workflow.py pipeline --max_cases 0 --epochs 50

    # Just prepare data (cached)
    flyte run --local --tui workflow.py prepare_data
"""

import asyncio
import json
import logging
import os
import random
import shutil
import tempfile

import flyte
import flyte.io
import flyte.report
from config import cpu_env, gpu_env
from report_helpers import (
    EVAL_REGIONS,
    LABEL_MAP,
    MRI_DISPLAY_NAMES,
    MRI_MODALITIES,
    REGION_COLORS,
    REGION_COLORS_HEX,
    dual_slice_viewer_html,
    make_bar_chart,
    make_line_chart,
    render_slice_gif,
    slice_to_data_uri,
    slice_viewer_html,
    three_plane_html,
    tumor_legend_html,
    wrap_report,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Task 1: Prepare data — download BraTS 2023 and organize
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_repo: str = "Angelou0516/brats2023-gli-dataset",
    max_cases: int = 100,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> flyte.io.Dir:
    """Download BraTS 2023 GLI dataset from HuggingFace and split train/val.

    Each case has 4 MRI modalities (T1, T1ce, T2, FLAIR) and a segmentation
    label, all as NIfTI files (.nii.gz). Data is already preprocessed
    (co-registered, 1mm isotropic, skull-stripped).

    Output directory:
      train/<case_id>/{t1n,t1c,t2w,t2f,seg}.nii.gz
      val/<case_id>/{t1n,t1c,t2w,t2f,seg}.nii.gz
      meta.json
    """
    import numpy as np
    from huggingface_hub import snapshot_download

    log.info(f"Downloading dataset: {dataset_repo}")
    dataset_path = snapshot_download(repo_id=dataset_repo, repo_type="dataset")

    # Find all case directories (contain .nii.gz files)
    all_cases = []
    for root, dirs, files in os.walk(dataset_path):
        nii_files = [f for f in files if f.endswith(".nii.gz")]
        if nii_files and any("seg" in f for f in nii_files):
            all_cases.append(root)

    all_cases.sort()
    log.info(f"Found {len(all_cases)} cases with segmentation labels")

    # Subset if requested
    if max_cases > 0 and max_cases < len(all_cases):
        rng = random.Random(seed)
        rng.shuffle(all_cases)
        all_cases = all_cases[:max_cases]
        all_cases.sort()
        log.info(f"Using subset of {max_cases} cases")

    # Split train/val
    rng = random.Random(seed)
    rng.shuffle(all_cases)
    n_val = max(1, int(len(all_cases) * val_fraction))
    val_cases = all_cases[:n_val]
    train_cases = all_cases[n_val:]

    log.info(f"Split: {len(train_cases)} train / {len(val_cases)} val")

    # Pack output directory
    out_dir = tempfile.mkdtemp(prefix="brats_seg_")

    tumor_stats = []

    for split_name, cases in [("train", train_cases), ("val", val_cases)]:
        for case_path in cases:
            case_id = os.path.basename(case_path)
            case_out = os.path.join(out_dir, split_name, case_id)
            os.makedirs(case_out, exist_ok=True)

            # Find and copy NIfTI files, normalizing names
            nii_files = [f for f in os.listdir(case_path) if f.endswith(".nii.gz")]
            for f in nii_files:
                src = os.path.join(case_path, f)
                # Normalize filename: look for modality suffixes
                f_lower = f.lower()
                if "seg" in f_lower:
                    dst_name = "seg.nii.gz"
                elif "t1c" in f_lower or "t1ce" in f_lower or "t1gd" in f_lower:
                    dst_name = "t1c.nii.gz"
                elif "t2f" in f_lower or "flair" in f_lower:
                    dst_name = "t2f.nii.gz"
                elif "t2w" in f_lower or ("t2" in f_lower and "f" not in f_lower):
                    dst_name = "t2w.nii.gz"
                elif "t1n" in f_lower or ("t1" in f_lower and "c" not in f_lower):
                    dst_name = "t1n.nii.gz"
                else:
                    dst_name = f
                shutil.copy2(src, os.path.join(case_out, dst_name))

            # Collect tumor stats from segmentation
            seg_path = os.path.join(case_out, "seg.nii.gz")
            if os.path.exists(seg_path):
                import nibabel as nib
                seg = nib.load(seg_path).get_fdata().astype(int)
                voxel_counts = {
                    "NCR": int((seg == 1).sum()),
                    "ED": int((seg == 2).sum()),
                    "ET": int((seg == 4).sum()),
                }
                total_tumor = sum(voxel_counts.values())
                tumor_stats.append({
                    "case_id": case_id,
                    "split": split_name,
                    "total_tumor_voxels": total_tumor,
                    **voxel_counts,
                })

    # Save metadata
    meta = {
        "dataset_repo": dataset_repo,
        "num_train": len(train_cases),
        "num_val": len(val_cases),
        "total_cases": len(all_cases),
        "modalities": MRI_MODALITIES,
        "labels": LABEL_MAP,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Data saved to {out_dir}")

    # -- Build report with sample MRI slices and tumor overlays --
    import nibabel as nib

    sample_cases = [s for s in tumor_stats if s["split"] == "train" and s["total_tumor_voxels"] > 1000][:3]

    samples_html = ""
    for stat in sample_cases:
        case_id = stat["case_id"]
        case_dir = os.path.join(out_dir, "train", case_id)

        # Load FLAIR (best for seeing edema) and segmentation
        flair_path = os.path.join(case_dir, "t2f.nii.gz")
        seg_path = os.path.join(case_dir, "seg.nii.gz")

        if not os.path.exists(flair_path) or not os.path.exists(seg_path):
            continue

        flair = nib.load(flair_path).get_fdata()
        seg = nib.load(seg_path).get_fdata().astype(int)

        samples_html += three_plane_html(
            flair, seg,
            label=f"Case: {case_id} — NCR:{stat['NCR']:,} ED:{stat['ED']:,} ET:{stat['ET']:,} voxels",
        )

    # Show 4-modality panel for first case
    modality_html = ""
    if sample_cases:
        case_dir = os.path.join(out_dir, "train", sample_cases[0]["case_id"])
        seg = nib.load(os.path.join(case_dir, "seg.nii.gz")).get_fdata().astype(int)

        # Find best axial slice (most tumor)
        tumor_per_slice = [(seg[:, :, z] > 0).sum() for z in range(seg.shape[2])]
        best_z = int(np.argmax(tumor_per_slice))

        modality_html = '<h3>4 MRI Modalities (Same Slice)</h3><div class="image-grid">'
        for mod in MRI_MODALITIES:
            mod_path = os.path.join(case_dir, f"{mod}.nii.gz")
            if os.path.exists(mod_path):
                vol = nib.load(mod_path).get_fdata()
                uri = slice_to_data_uri(vol[:, :, best_z])
                modality_html += f'''
                <div class="image-card">
                  <img src="{uri}" />
                  <div class="caption">{MRI_DISPLAY_NAMES[mod]}</div>
                </div>'''
        modality_html += "</div>"

    # Tumor size distribution
    tumor_volumes = [s["total_tumor_voxels"] for s in tumor_stats if s["total_tumor_voxels"] > 0]
    avg_tumor = float(np.mean(tumor_volumes)) if tumor_volumes else 0
    has_tumor_pct = len(tumor_volumes) / len(tumor_stats) * 100 if tumor_stats else 0

    report_html = f"""
    <h2>Dataset: BraTS 2023 GLI</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{meta['total_cases']}</div><div class="label">Total Cases</div></div>
      <div class="stat"><div class="value">{meta['num_train']}</div><div class="label">Training</div></div>
      <div class="stat"><div class="value">{meta['num_val']}</div><div class="label">Validation</div></div>
      <div class="stat"><div class="value">4</div><div class="label">MRI Modalities</div></div>
      <div class="stat"><div class="value">3</div><div class="label">Tumor Subregions</div></div>
      <div class="stat"><div class="value">{has_tumor_pct:.0f}%</div><div class="label">With Tumor</div></div>
    </div>
    {tumor_legend_html()}
    {modality_html}
    <h3>Sample Cases with Tumor Overlays (FLAIR)</h3>
    <p>Axial, coronal, and sagittal views centered on tumor mass.</p>
    {samples_html}
    <div class="note">
      <b>BraTS 2023:</b> Multi-modal MRI brain scans with glioma annotations.
      Each case has T1, T1-contrast, T2, and FLAIR volumes (240x240x155 voxels, 1mm isotropic).
      Tumors are segmented into 3 subregions:
      <span style="color:#ffdc32;">necrotic core (NCR)</span>,
      <span style="color:#32cd32;">peritumoral edema (ED)</span>, and
      <span style="color:#dc3232;">enhancing tumor (ET)</span>.
    </div>
    """

    await flyte.report.replace.aio(wrap_report(report_html), do_flush=True)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Task 2: Train — SegResNet via MONAI
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    data_dir: flyte.io.Dir,
    epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patch_size: int = 128,
) -> flyte.io.Dir:
    """Train a SegResNet for 3D brain tumor segmentation using MONAI.

    Uses patch-based training (128x128x128 crops from 240x240x155 volumes)
    to fit in GPU memory. Reports training loss and validation Dice live.
    """
    import numpy as np
    import torch
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.losses import DiceLoss
    from monai.metrics import DiceMetric
    from monai.networks.nets import SegResNet
    from monai.transforms import (
        CastToTyped,
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        CropForegroundd,
        DivisiblePadd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        RandFlipd,
        RandScaleIntensityd,
        RandShiftIntensityd,
        RandSpatialCropd,
        Activationsd,
        AsDiscreted,
    )

    log.info("Setting up SegResNet training...")
    await flyte.report.replace.aio(wrap_report(
        "<h2>Loading Model...</h2>"
        "<p>SegResNet for 3D brain tumor segmentation</p>"
        "<p>Preparing MONAI data pipeline with patch-based training...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    with open(os.path.join(data_path, "meta.json")) as f:
        meta = json.load(f)

    # Build file lists for MONAI
    def _get_data_dicts(split: str) -> list[dict]:
        split_dir = os.path.join(data_path, split)
        if not os.path.isdir(split_dir):
            return []
        dicts = []
        for case_id in sorted(os.listdir(split_dir)):
            case_dir = os.path.join(split_dir, case_id)
            if not os.path.isdir(case_dir):
                continue
            entry = {"label": os.path.join(case_dir, "seg.nii.gz")}
            # Stack all 4 modalities as channels
            images = []
            for mod in MRI_MODALITIES:
                p = os.path.join(case_dir, f"{mod}.nii.gz")
                if os.path.exists(p):
                    images.append(p)
            if len(images) == 4 and os.path.exists(entry["label"]):
                entry["image"] = images
                dicts.append(entry)
        return dicts

    train_dicts = _get_data_dicts("train")
    val_dicts = _get_data_dicts("val")
    log.info(f"Train: {len(train_dicts)} cases | Val: {len(val_dicts)} cases")

    # MONAI transforms
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CastToTyped(keys="label", dtype="float32"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandSpatialCropd(
            keys=["image", "label"],
            roi_size=[patch_size, patch_size, patch_size],
            random_size=False,
        ),
        DivisiblePadd(keys=["image", "label"], k=16),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CastToTyped(keys="label", dtype="float32"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=16),
    ])

    # For large datasets, disable caching to avoid OOM (each 3D volume is ~56MB).
    # For small runs (<100 cases), cache_rate=0.5 is fine.
    cache_rate = 0.5 if len(train_dicts) <= 100 else 0.0
    val_cache_rate = 1.0 if len(val_dicts) <= 50 else 0.0
    train_ds = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=cache_rate)
    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=val_cache_rate)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # SegResNet: 4 input channels (T1, T1ce, T2, FLAIR), 3 output channels
    # (WT, TC, ET — the standard BraTS composite regions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"SegResNet parameters: {trainable_params:,} / {total_params:,}")

    loss_fn = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])

    # Training loop with live reporting
    training_log: list[dict] = []
    val_log: list[dict] = []
    loop = asyncio.get_running_loop()
    total_steps = epochs * len(train_loader)

    def _build_training_report() -> str:
        stats_html = f"""
        <h2>Training in Progress...</h2>
        <h3>SegResNet — 3D Brain Tumor Segmentation</h3>
        <div class="stat-grid">
          <div class="stat"><div class="value">{len(train_dicts)}</div><div class="label">Train Cases</div></div>
          <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
          <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
          <div class="stat"><div class="value">{patch_size}³</div><div class="label">Patch Size</div></div>
          <div class="stat"><div class="value">{learning_rate}</div><div class="label">Learning Rate</div></div>
          <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>
        </div>
        {tumor_legend_html()}
        """

        charts_html = ""
        if training_log:
            current = training_log[-1]
            progress_pct = current["epoch"] / epochs * 100

            charts_html += f"""
            <div class="card">
              <b>Epoch {current['epoch']}/{epochs}</b>
              ({progress_pct:.0f}%) |
              Train Loss: <span class="highlight">{current['train_loss']:.4f}</span>
              <div style="background:#bfdbfe;border-radius:4px;height:8px;margin-top:8px;">
                <div style="background:#3b82f6;width:{progress_pct:.1f}%;height:100%;border-radius:4px;"></div>
              </div>
            </div>
            """

            loss_chart = make_line_chart(
                data=training_log,
                x_key="epoch",
                y_keys=["train_loss"],
                title="Training Loss (Dice)",
                x_label="Epoch",
                y_label="Loss",
                colors=["#3b82f6"],
                x_range_override=(0, epochs),
            )
            charts_html += f'<div class="chart-container">{loss_chart}</div>'

        if val_log:
            dice_chart = make_line_chart(
                data=val_log,
                x_key="epoch",
                y_keys=["dice_wt", "dice_tc", "dice_et"],
                title="Validation Dice Score",
                x_label="Epoch",
                y_label="Dice",
                colors=["#06d6a0", "#f59e0b", "#dc3232"],
                y_max_cap=1.0,
                x_range_override=(0, epochs),
                y_display_names={
                    "dice_wt": "Whole Tumor",
                    "dice_tc": "Tumor Core",
                    "dice_et": "Enhancing",
                },
            )
            charts_html += f'<div class="chart-container">{dice_chart}</div>'

        return wrap_report(stats_html + charts_html)

    def _train_loop():
        best_dice = 0
        val_interval = max(1, epochs // 10)

        log_interval = max(1, len(train_dicts) // 10)  # log ~10 times per epoch

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_losses = []

            for batch_idx, batch_data in enumerate(train_loader):
                inputs = batch_data["image"].to(device)
                labels = batch_data["label"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

                if (batch_idx + 1) % log_interval == 0:
                    log.info(
                        f"  step={batch_idx + 1}/{len(train_loader)} "
                        f"epoch={epoch} loss={loss.item():.4f}"
                    )

            lr_scheduler.step()
            avg_loss = np.mean(epoch_losses)

            entry = {
                "epoch": epoch,
                "train_loss": round(float(avg_loss), 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
            training_log.append(entry)
            log.info(f"Epoch {epoch}/{epochs} — train loss: {avg_loss:.4f}")

            # Validation at intervals
            if epoch % val_interval == 0 or epoch == epochs:
                model.eval()
                dice_metric.reset()

                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs = val_data["image"].to(device)
                        val_labels = val_data["label"].to(device)

                        val_outputs = sliding_window_inference(
                            val_inputs,
                            roi_size=(patch_size, patch_size, patch_size),
                            sw_batch_size=4,
                            predictor=model,
                            overlap=0.5,
                        )

                        val_data["pred"] = val_outputs
                        val_data = [post_trans(i) for i in decollate_batch(val_data)]
                        val_outputs_post = [d["pred"].cpu() for d in val_data]
                        val_labels_post = [d["label"].cpu() for d in val_data]

                        dice_metric(y_pred=val_outputs_post, y=val_labels_post)

                dice_values = dice_metric.aggregate()
                dice_wt = float(dice_values[0])
                dice_tc = float(dice_values[1])
                dice_et = float(dice_values[2])
                mean_dice = (dice_wt + dice_tc + dice_et) / 3

                val_log.append({
                    "epoch": epoch,
                    "dice_wt": round(dice_wt, 4),
                    "dice_tc": round(dice_tc, 4),
                    "dice_et": round(dice_et, 4),
                    "mean_dice": round(mean_dice, 4),
                })

                log.info(f"  Val Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} "
                         f"ET:{dice_et:.3f} Mean:{mean_dice:.3f}")

                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), os.path.join(tempfile.gettempdir(), "best_model.pth"))
                    log.info(f"  New best model (mean dice: {best_dice:.3f})")

            asyncio.run_coroutine_threadsafe(
                flyte.report.replace.aio(_build_training_report(), do_flush=True),
                loop,
            )

    log.info("Starting training...")
    await asyncio.to_thread(_train_loop)
    log.info("Training complete.")

    # Save best model
    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_segresnet")
    os.makedirs(save_dir, exist_ok=True)

    best_path = os.path.join(tempfile.gettempdir(), "best_model.pth")
    if os.path.exists(best_path):
        shutil.copy2(best_path, os.path.join(save_dir, "model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))

    model_config = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "patch_size": patch_size,
        "in_channels": 4,
        "out_channels": 3,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    shutil.copy2(os.path.join(data_path, "meta.json"), os.path.join(save_dir, "meta.json"))

    # Final training report
    stats_html = f"""
    <h2>Training Complete</h2>
    <h3>SegResNet — 3D Brain Tumor Segmentation</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(train_dicts)}</div><div class="label">Train Cases</div></div>
      <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value">{epochs}</div><div class="label">Epochs</div></div>
      <div class="stat"><div class="value">{trainable_params:,}</div><div class="label">Parameters</div></div>
    </div>
    {tumor_legend_html()}
    """

    charts_html = ""
    if training_log:
        loss_chart = make_line_chart(
            data=training_log,
            x_key="epoch",
            y_keys=["train_loss"],
            title="Training Loss (Dice)",
            x_label="Epoch",
            y_label="Loss",
            colors=["#3b82f6"],
            x_range_override=(0, epochs),
        )
        charts_html += f'<div class="chart-container">{loss_chart}</div>'

    if val_log:
        best = max(val_log, key=lambda d: d["mean_dice"])
        charts_html += f"""
        <div class="card">
          <b>Best Validation (Epoch {best['epoch']}):</b>
          WT: <span class="highlight">{best['dice_wt']:.3f}</span> |
          TC: <span class="highlight">{best['dice_tc']:.3f}</span> |
          ET: <span class="highlight">{best['dice_et']:.3f}</span> |
          Mean: <span class="highlight">{best['mean_dice']:.3f}</span>
        </div>
        """

        dice_chart = make_line_chart(
            data=val_log,
            x_key="epoch",
            y_keys=["dice_wt", "dice_tc", "dice_et"],
            title="Validation Dice Score",
            x_label="Epoch",
            y_label="Dice",
            colors=["#06d6a0", "#f59e0b", "#dc3232"],
            y_max_cap=1.0,
            x_range_override=(0, epochs),
            y_display_names={
                "dice_wt": "Whole Tumor",
                "dice_tc": "Tumor Core",
                "dice_et": "Enhancing",
            },
        )
        charts_html += f'<div class="chart-container">{dice_chart}</div>'

    await flyte.report.replace.aio(wrap_report(stats_html + charts_html), do_flush=True)

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Helper: load trained model
# ------------------------------------------------------------------

def _load_segresnet(model_path: str, device):
    """Load a trained SegResNet from saved state dict."""
    import torch
    from monai.networks.nets import SegResNet

    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    model = SegResNet(
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        init_filters=16,
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        dropout_prob=0.2,
    )
    model.load_state_dict(torch.load(
        os.path.join(model_path, "model.pth"),
        map_location=device,
        weights_only=True,
    ))
    model.to(device)
    model.eval()
    return model, config


# ------------------------------------------------------------------
# Task 3: Evaluate — Dice scores per tumor region
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
) -> str:
    """Evaluate the trained SegResNet on validation cases.

    Computes Dice score for each composite tumor region:
      WT (Whole Tumor) = labels 1+2+4
      TC (Tumor Core) = labels 1+4
      ET (Enhancing Tumor) = label 4
    """
    import numpy as np
    import torch
    from monai.data import CacheDataset, DataLoader, decollate_batch
    from monai.inferers import sliding_window_inference
    from monai.metrics import DiceMetric
    from monai.transforms import (
        Activationsd,
        AsDiscreted,
        CastToTyped,
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        CropForegroundd,
        DivisiblePadd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
    )

    log.info("Starting evaluation...")
    await flyte.report.replace.aio(wrap_report(
        "<h2>Evaluation</h2><p>Running sliding window inference on validation set...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_segresnet(ft_path, device)
    patch_size = config["patch_size"]

    # Build val data
    val_dir = os.path.join(data_path, "val")
    val_dicts = []
    for case_id in sorted(os.listdir(val_dir)):
        case_dir = os.path.join(val_dir, case_id)
        if not os.path.isdir(case_dir):
            continue
        images = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in MRI_MODALITIES]
        label = os.path.join(case_dir, "seg.nii.gz")
        if all(os.path.exists(p) for p in images) and os.path.exists(label):
            val_dicts.append({"image": images, "label": label})

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CastToTyped(keys="label", dtype="float32"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        DivisiblePadd(keys=["image", "label"], k=16),
    ])

    val_ds = CacheDataset(data=val_dicts, transform=val_transforms, cache_rate=1.0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activationsd(keys="pred", sigmoid=True), AsDiscreted(keys="pred", threshold=0.5)])

    per_case_dice = []

    with torch.no_grad():
        for val_data in val_loader:
            val_inputs = val_data["image"].to(device)
            val_labels = val_data["label"].to(device)

            val_outputs = sliding_window_inference(
                val_inputs,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            val_data["pred"] = val_outputs
            val_data = [post_trans(i) for i in decollate_batch(val_data)]
            val_outputs_post = [d["pred"].cpu() for d in val_data]
            val_labels_post = [d["label"].cpu() for d in val_data]

            dice_metric(y_pred=val_outputs_post, y=val_labels_post)

            # Per-case dice
            case_dice = DiceMetric(include_background=True, reduction="mean_batch")
            case_dice(y_pred=val_outputs_post, y=val_labels_post)
            cd = case_dice.aggregate()
            per_case_dice.append({
                "wt": float(cd[0]),
                "tc": float(cd[1]),
                "et": float(cd[2]),
            })

    dice_values = dice_metric.aggregate()
    dice_wt = float(dice_values[0])
    dice_tc = float(dice_values[1])
    dice_et = float(dice_values[2])
    mean_dice = (dice_wt + dice_tc + dice_et) / 3

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log.info(f"Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} ET:{dice_et:.3f} Mean:{mean_dice:.3f}")

    # Build report
    bar_chart = make_bar_chart(
        labels=["Whole Tumor", "Tumor Core", "Enhancing"],
        series={"Dice": [dice_wt, dice_tc, dice_et]},
        title="Dice Score by Tumor Region",
        colors=["#06d6a0", "#f59e0b", "#dc3232"],
        y_max_cap=1.0,
    )

    eval_html = f"""
    <h2>Evaluation Results</h2>
    <div class="stat-grid">
      <div class="stat"><div class="value">{len(val_dicts)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
      <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
      <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
      <div class="stat"><div class="value highlight">{mean_dice:.3f}</div><div class="label">Mean Dice</div></div>
    </div>
    {tumor_legend_html()}
    <div class="chart-container">{bar_chart}</div>
    <table>
      <tr><th>Region</th><th>Dice Score</th><th>Description</th></tr>
      <tr><td><b>Whole Tumor (WT)</b></td><td class="highlight">{dice_wt:.3f}</td><td>All tumor subregions (NCR+ED+ET)</td></tr>
      <tr><td><b>Tumor Core (TC)</b></td><td class="highlight">{dice_tc:.3f}</td><td>Core without edema (NCR+ET)</td></tr>
      <tr><td><b>Enhancing (ET)</b></td><td class="highlight">{dice_et:.3f}</td><td>Active tumor growth only</td></tr>
    </table>
    <div class="note">
      <b>Dice Score</b> measures overlap between predicted and ground truth segmentation (0=no overlap, 1=perfect).
      WT is typically easiest (largest region), ET is hardest (smallest, most variable).
      BraTS challenge winners achieve ~0.90 WT, ~0.85 TC, ~0.80 ET.
    </div>
    """

    await flyte.report.replace.aio(wrap_report(eval_html), do_flush=True)

    return json.dumps({
        "dice_wt": dice_wt,
        "dice_tc": dice_tc,
        "dice_et": dice_et,
        "mean_dice": mean_dice,
        "num_val_cases": len(val_dicts),
        "per_case_dice": per_case_dice,
    })


# ------------------------------------------------------------------
# Task 4: Inference — multi-plane tumor overlay visualizations
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def explore_results(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    max_cases: int = 4,
    metrics_json: str = "{}",
) -> str:
    """Run inference on validation cases and render multi-plane tumor overlays.

    Shows axial/coronal/sagittal views with ground truth vs predicted tumor
    overlays, color-coded by subregion.
    """
    import nibabel as nib
    import numpy as np
    import torch
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        CastToTyped,
        Compose,
        ConvertToMultiChannelBasedOnBratsClassesd,
        DivisiblePadd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
    )

    log.info("Starting inference visualization...")
    await flyte.report.replace.aio(wrap_report(
        "<h2>Inference</h2><p>Generating multi-plane tumor overlays...</p>"
    ), do_flush=True)

    data_path = await data_dir.download()
    ft_path = await finetuned_dir.download()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = _load_segresnet(ft_path, device)
    patch_size = config["patch_size"]

    # Load val cases
    val_dir = os.path.join(data_path, "val")
    case_ids = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

    rng = random.Random(42)
    rng.shuffle(case_ids)
    case_ids = case_ids[:max_cases]

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="label"),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CastToTyped(keys="label", dtype="float32"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        DivisiblePadd(keys=["image", "label"], k=16),
    ])

    html_blocks = []

    for case_id in case_ids:
        case_dir = os.path.join(val_dir, case_id)
        images = [os.path.join(case_dir, f"{mod}.nii.gz") for mod in MRI_MODALITIES]
        label_path = os.path.join(case_dir, "seg.nii.gz")

        if not all(os.path.exists(p) for p in images) or not os.path.exists(label_path):
            continue

        data_dict = {"image": images, "label": label_path}
        data_dict = val_transforms(data_dict)

        # Run inference
        input_tensor = data_dict["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = sliding_window_inference(
                input_tensor,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )
            pred = (torch.sigmoid(output) > 0.5).squeeze(0).cpu().numpy()

        # Ground truth and FLAIR at original resolution
        gt_seg = nib.load(label_path).get_fdata().astype(int)
        flair = nib.load(os.path.join(case_dir, "t2f.nii.gz")).get_fdata()
        orig_shape = flair.shape

        # Convert multi-channel pred back to label map
        # Channel 0=WT (all tumor), 1=TC (core=NCR+ET), 2=ET (enhancing)
        # Use set differences to reconstruct exclusive regions:
        wt = pred[0] > 0   # whole tumor
        tc = pred[1] > 0   # tumor core
        et = pred[2] > 0   # enhancing tumor
        pred_seg = np.zeros(pred.shape[1:], dtype=int)
        pred_seg[wt & ~tc] = 2    # ED: in whole tumor but NOT in core
        pred_seg[tc & ~et] = 1    # NCR: in core but NOT enhancing
        pred_seg[et] = 4          # ET: enhancing tumor

        # Crop pred back to original volume size (DivisiblePadd may have padded)
        pred_seg = pred_seg[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

        # Mask predictions to brain region only (no tumor outside the skull)
        brain_mask = flair > 0
        pred_seg[~brain_mask] = 0

        # Count voxels per region
        gt_counts = {
            "NCR": int((gt_seg == 1).sum()),
            "ED": int((gt_seg == 2).sum()),
            "ET": int((gt_seg == 4).sum()),
        }
        pred_counts = {
            "NCR": int((pred_seg == 1).sum()),
            "ED": int((pred_seg == 2).sum()),
            "ET": int((pred_seg == 4).sum()),
        }

        # Interactive slice viewer with slider + play/pause (GT vs Predicted)
        viewer_html = dual_slice_viewer_html(
            flair, gt_seg, pred_seg,
            axis="axial", n_frames=24,
            label=f"Case: {case_id} — Axial Slice Viewer",
        )

        # Static 3-plane overview
        static_html = f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
          <div>{three_plane_html(flair, gt_seg, label="Ground Truth")}</div>
          <div>{three_plane_html(flair, pred_seg, label="Predicted")}</div>
        </div>
        """

        html_blocks.append(f"""
        <div class="card">
          <b>Case: {case_id}</b>
          <span style="font-size:0.8em;color:#6c757d;">
            GT: NCR={gt_counts['NCR']:,} ED={gt_counts['ED']:,} ET={gt_counts['ET']:,} |
            Pred: NCR={pred_counts['NCR']:,} ED={pred_counts['ED']:,} ET={pred_counts['ET']:,}
          </span>
          {viewer_html}
          {static_html}
        </div>
        """)

    # Parse metrics
    metrics = json.loads(metrics_json)
    dice_wt = metrics.get("dice_wt")
    dice_tc = metrics.get("dice_tc")
    dice_et = metrics.get("dice_et")

    metric_stats = ""
    if dice_wt is not None:
        metric_stats = f"""
        <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
        <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
        <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
        """

    demo_html = f"""
    <h2>Brain Tumor Segmentation Results</h2>
    <div class="stat-grid">
      {metric_stats}
      <div class="stat"><div class="value">{len(case_ids)}</div><div class="label">Cases Shown</div></div>
    </div>
    {tumor_legend_html()}
    <p>Use the <b>slider</b> to scrub through axial slices or hit <b>Play</b> to animate.
    Ground truth (left) and predictions (right) are synced.
    Static 3-plane views (axial/coronal/sagittal) are shown below each viewer.
    <span style="color:#ffdc32;">Yellow=NCR</span>
    <span style="color:#32cd32;">Green=ED</span>
    <span style="color:#dc3232;">Red=ET</span></p>
    {"".join(html_blocks)}
    """

    await flyte.report.replace.aio(wrap_report(demo_html), do_flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return json.dumps({
        "num_cases": len(case_ids),
    })


# ------------------------------------------------------------------
# Pipeline — chains all 4 tasks
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    dataset_repo: str = "Angelou0516/brats2023-gli-dataset",
    max_cases: int = 100,
    epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    patch_size: int = 128,
    val_fraction: float = 0.15,
    demo_cases: int = 4,
) -> tuple[flyte.io.Dir, str]:
    """
    End-to-end 3D brain tumor segmentation pipeline.

    1. Download BraTS 2023 MRI volumes from HuggingFace
    2. Train SegResNet for 3D tumor segmentation using MONAI
    3. Evaluate with Dice scores (WT, TC, ET)
    4. Render multi-plane tumor overlays on validation cases
    """
    log.info(f"Pipeline: SegResNet | dataset={dataset_repo}")

    def _pipeline_progress(step: int, label: str) -> str:
        steps = ["Preparing Data", "Training SegResNet", "Evaluating", "Explore Results"]
        dots = ""
        for i, s in enumerate(steps):
            if i + 1 < step:
                icon = '<span style="color:#059669;">&#10003;</span>'
            elif i + 1 == step:
                icon = '<span style="color:#3b82f6;">&#9679;</span>'
            else:
                icon = '<span style="color:#93c5fd;">&#9675;</span>'
            dots += f"<span style='margin:0 8px;'>{icon} {s}</span>"
        return f"""
        <h2>Brain Tumor Segmentation Pipeline</h2>
        <p><b>Model:</b> SegResNet | <b>Dataset:</b> BraTS 2023 GLI</p>
        <div class="card" style="text-align:center;">{dots}</div>
        <p>{label}</p>
        """

    # Step 1: Prepare data
    await flyte.report.replace.aio(
        wrap_report(_pipeline_progress(1, "Downloading BraTS 2023 from HuggingFace...")),
        do_flush=True,
    )
    data_dir = await prepare_data(
        dataset_repo=dataset_repo,
        max_cases=max_cases,
        val_fraction=val_fraction,
    )

    # Step 2: Train
    await flyte.report.replace.aio(
        wrap_report(_pipeline_progress(2, "Training SegResNet on 3D MRI volumes...")),
        do_flush=True,
    )
    finetuned_dir = await train(
        data_dir=data_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        patch_size=patch_size,
    )

    # Step 3: Evaluate
    await flyte.report.replace.aio(
        wrap_report(_pipeline_progress(3, "Computing Dice scores (WT, TC, ET)...")),
        do_flush=True,
    )
    metrics_json = await evaluate(finetuned_dir, data_dir)
    metrics = json.loads(metrics_json)

    # Step 4: Inference
    await flyte.report.replace.aio(
        wrap_report(_pipeline_progress(4, "Generating multi-plane tumor overlays...")),
        do_flush=True,
    )
    inference_json = await explore_results(
        finetuned_dir, data_dir, demo_cases,
        metrics_json=metrics_json,
    )

    # Final pipeline report
    dice_wt = metrics.get("dice_wt", 0)
    dice_tc = metrics.get("dice_tc", 0)
    dice_et = metrics.get("dice_et", 0)
    mean_dice = metrics.get("mean_dice", 0)

    final_html = f"""
    <h2>Pipeline Complete</h2>
    <h3>SegResNet on BraTS 2023 GLI</h3>
    <div class="stat-grid">
      <div class="stat"><div class="value">{metrics.get('num_val_cases', 0)}</div><div class="label">Val Cases</div></div>
      <div class="stat"><div class="value highlight">{dice_wt:.3f}</div><div class="label">Dice WT</div></div>
      <div class="stat"><div class="value highlight">{dice_tc:.3f}</div><div class="label">Dice TC</div></div>
      <div class="stat"><div class="value highlight">{dice_et:.3f}</div><div class="label">Dice ET</div></div>
      <div class="stat"><div class="value highlight">{mean_dice:.3f}</div><div class="label">Mean Dice</div></div>
    </div>
    {tumor_legend_html()}
    <div class="card">
      <b>Configuration:</b> {epochs} epochs | LR {learning_rate} | Patch {patch_size}³ |
      {max_cases} cases | Val fraction {val_fraction}
    </div>
    <div class="note">
      Pipeline ran: prepare_data &#8594; train &#8594; evaluate &#8594; inference.
      Check individual task reports for loss curves, Dice progression,
      and side-by-side tumor overlay visualizations.
    </div>
    """

    await flyte.report.replace.aio(wrap_report(final_html), do_flush=True)

    log.info(f"Pipeline complete. Dice — WT:{dice_wt:.3f} TC:{dice_tc:.3f} ET:{dice_et:.3f}")
    return finetuned_dir, json.dumps({"metrics": metrics, "inference": json.loads(inference_json)})
