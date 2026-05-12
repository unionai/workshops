"""
RT-DETRv2 Object Detection — Fine-tune on a custom COCO dataset.

Pipeline: download a COCO-format dataset from HuggingFace, fine-tune RT-DETRv2
for object detection, evaluate with COCO mAP, and render an inference demo
with bounding boxes drawn on held-out images.

Usage:
    # Default (RT-DETRv2-R18 on Union swag stickers)
    flyte run --local --tui workflow.py pipeline

    # Quick local test
    flyte run --local --tui workflow.py pipeline --epochs 1 --batch_size 2

    # Remote
    flyte run workflow.py pipeline --epochs 30

    # Swap model
    flyte run workflow.py pipeline --model_name "PekingU/rtdetr_v2_r50vd"
"""

import asyncio
import base64
import io
import json
import logging
import os
import random
import shutil
import tempfile

import flyte
import flyte.report
from config import cpu_env, gpu_env, HF_TOKEN

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------------------------------------------
# Task 1: Prepare dataset — download COCO JSON + images, split train/val
# ------------------------------------------------------------------

@cpu_env.task(cache="auto")
async def prepare_data(
    dataset_repo: str = "sagecodes/union_swag_coco",
    annotations_path: str = "swag/train.json",
    images_subdir: str = "swag/images",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> flyte.io.Dir:
    """Download a COCO-format dataset from HF and split into train/val."""
    from huggingface_hub import snapshot_download

    log.info(f"Downloading dataset: {dataset_repo}")
    local_repo = snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        token=HF_TOKEN,
    )

    ann_file = os.path.join(local_repo, annotations_path)
    img_root = os.path.join(local_repo, images_subdir)

    with open(ann_file) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    log.info(
        f"Loaded {len(images)} images, {len(annotations)} annotations, "
        f"{len(categories)} categories"
    )
    log.info(f"Raw category ids: {sorted({c['id'] for c in categories})}")
    log.info(
        f"Raw annotation category_ids (unique): "
        f"{sorted({a['category_id'] for a in annotations})}"
    )

    # Remap category ids to contiguous 0..N-1 — required because HF object
    # detection models size their classifier head to len(id2label) and treat
    # class labels as direct indices into that head. Any gap or 1-indexed id
    # causes an IndexKernel OOB inside the focal-loss scatter.
    #
    # Build the remap from the UNION of ids declared in `categories` and ids
    # actually used in `annotations` — some datasets have orphaned annotations
    # referencing categories that aren't declared (this one does).
    declared_ids = {c["id"] for c in categories}
    used_ids = {a["category_id"] for a in annotations}
    orphans = used_ids - declared_ids
    if orphans:
        log.warning(
            f"Annotations reference undeclared category ids {sorted(orphans)} — "
            f"adding stub categories."
        )

    all_cat_ids = sorted(declared_ids | used_ids)
    id_remap = {old: new for new, old in enumerate(all_cat_ids)}
    existing_names = {c["id"]: c["name"] for c in categories}
    categories = [
        {"id": id_remap[old], "name": existing_names.get(old, f"category_{old}")}
        for old in all_cat_ids
    ]
    annotations = [
        {**a, "category_id": id_remap[a["category_id"]]} for a in annotations
    ]
    log.info(f"Remapped category ids: {id_remap}")
    log.info(f"Final categories: {categories}")

    # Split by image id
    rng = random.Random(seed)
    img_ids = [im["id"] for im in images]
    rng.shuffle(img_ids)
    n_val = max(1, int(len(img_ids) * val_fraction))
    val_ids = set(img_ids[:n_val])
    train_ids = set(img_ids[n_val:])

    def filter_coco(keep_ids: set) -> dict:
        return {
            "info": coco.get("info", {}),
            "categories": categories,
            "images": [im for im in images if im["id"] in keep_ids],
            "annotations": [a for a in annotations if a["image_id"] in keep_ids],
        }

    train_coco = filter_coco(train_ids)
    val_coco = filter_coco(val_ids)

    log.info(
        f"Split: {len(train_coco['images'])} train / {len(val_coco['images'])} val images"
    )

    # Pack output dir: images/ + train.json + val.json
    out_dir = tempfile.mkdtemp(prefix="coco_split_")
    out_img = os.path.join(out_dir, "images")
    shutil.copytree(img_root, out_img)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(train_coco, f)
    with open(os.path.join(out_dir, "val.json"), "w") as f:
        json.dump(val_coco, f)

    return await flyte.io.Dir.from_local(out_dir)


# ------------------------------------------------------------------
# Helpers — torch Dataset wrapping COCO JSON
# ------------------------------------------------------------------

def _build_torch_dataset(coco_path: str, images_root: str, augment: bool):
    """Build a torch Dataset that yields {image, target} for the HF image processor."""
    import albumentations as A
    import numpy as np
    from PIL import Image
    from torch.utils.data import Dataset

    with open(coco_path) as f:
        coco = json.load(f)

    images_by_id = {im["id"]: im for im in coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    image_ids = list(images_by_id.keys())

    # NOTE: we deliberately don't resize here — the HF image processor handles
    # resize+pad. Augmentation only.
    if augment:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category"],
                min_area=4,
                min_visibility=0.1,
                clip=True,
            ),
        )
    else:
        transform = A.Compose(
            [A.NoOp()],
            bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
        )

    class CocoDataset(Dataset):
        def __len__(self) -> int:
            return len(image_ids)

        def __getitem__(self, idx: int):
            img_id = image_ids[idx]
            meta = images_by_id[img_id]
            img_path = os.path.join(images_root, os.path.basename(meta["file_name"]))
            if not os.path.exists(img_path):
                img_path = os.path.join(images_root, meta["file_name"])
            image = np.array(Image.open(img_path).convert("RGB"))

            anns = anns_by_image.get(img_id, [])
            bboxes = [a["bbox"] for a in anns]
            categories = [a["category_id"] for a in anns]

            out = transform(image=image, bboxes=bboxes, category=categories)
            image_t = out["image"]
            bboxes_t = out["bboxes"]
            categories_t = out["category"]

            target_anns = []
            for bb, cat in zip(bboxes_t, categories_t):
                x, y, w, h = bb
                target_anns.append(
                    {
                        "image_id": img_id,
                        "category_id": int(cat),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0,
                    }
                )

            return {
                "image": image_t,
                "target": {"image_id": img_id, "annotations": target_anns},
            }

    return CocoDataset(), coco["categories"]


# ------------------------------------------------------------------
# Task 2: Train
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def train(
    model_name: str,
    data_dir: flyte.io.Dir,
    epochs: int = 30,
    lr: float = 5e-5,
    batch_size: int = 4,
    weight_decay: float = 1e-4,
) -> flyte.io.Dir:
    """Fine-tune RT-DETR (or any HuggingFace object-detection model) on COCO data."""
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForObjectDetection,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    log.info(f"Training: model={model_name}")
    await flyte.report.replace.aio(f"<h2>Loading model: {model_name}</h2>")
    await flyte.report.flush.aio()

    # -- Load data --
    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    train_json = os.path.join(data_path, "train.json")

    with open(train_json) as f:
        categories = json.load(f)["categories"]
    id2label = {c["id"]: c["name"] for c in categories}
    label2id = {v: k for k, v in id2label.items()}

    train_ds, _ = _build_torch_dataset(train_json, images_root, augment=True)
    log.info(f"Train examples: {len(train_ds)} | Categories: {id2label}")

    # -- Processor + model --
    processor = AutoImageProcessor.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForObjectDetection.from_pretrained(
        model_name,
        token=HF_TOKEN,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        f"Parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.1f}%)"
    )

    # -- Collator — runs the image processor on each batch --
    def collate_fn(batch):
        images = [b["image"] for b in batch]
        targets = [b["target"] for b in batch]
        enc = processor(images=images, annotations=targets, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"], "labels": enc["labels"]}

    # -- Sanity check: peek at one batch and verify class_labels fit --
    sample = collate_fn([train_ds[i] for i in range(min(2, len(train_ds)))])
    all_labels = []
    for lbl in sample["labels"]:
        all_labels.extend(lbl["class_labels"].tolist())
    log.info(
        f"Sanity check — class_labels in first batch: {sorted(set(all_labels))} | "
        f"model num_labels: {model.config.num_labels} | "
        f"id2label: {model.config.id2label}"
    )
    if all_labels and max(all_labels) >= model.config.num_labels:
        raise ValueError(
            f"class_label {max(all_labels)} out of range for num_labels="
            f"{model.config.num_labels}. Check category id remapping in prepare_data."
        )

    # -- Log-to-stdout callback only. We deliberately do NOT call
    # flyte.report.replace from inside the callback: trainer.train() blocks
    # the asyncio event loop, and the sync `replace` wrapper would deadlock
    # against the syncify bridge that needs that loop to schedule work.
    class StdoutProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or "loss" not in logs:
                return
            log.info(
                f"step={state.global_step}/{state.max_steps} "
                f"epoch={logs.get('epoch', 0):.2f} "
                f"loss={logs['loss']:.4f}"
            )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output_dir = os.path.join(tempfile.mkdtemp(), "checkpoints")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_steps=5,
        save_strategy="no",
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        warmup_ratio=0.1,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collate_fn,
        callbacks=[StdoutProgressCallback()],
    )

    log.info("Starting training...")
    # Run the sync HF training loop in a thread so the asyncio event loop
    # stays free for Flyte's syncify bridge.
    await asyncio.to_thread(trainer.train)
    log.info("Training complete.")

    save_dir = os.path.join(tempfile.mkdtemp(), "finetuned_model")
    trainer.save_model(save_dir)
    processor.save_pretrained(save_dir)
    log.info(f"Model saved to {save_dir}")

    await flyte.report.replace.aio(
        f"<h2>Training Complete — {model_name}</h2>"
        f"<p><b>Train examples:</b> {len(train_ds)}</p>"
        f"<p><b>Epochs:</b> {epochs} | <b>LR:</b> {lr} | <b>Batch size:</b> {batch_size}</p>"
        f"<p><b>Categories:</b> {', '.join(id2label.values())}</p>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(save_dir)


# ------------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------------

def _run_inference(model, processor, images, device, threshold: float = 0.3):
    """Run object detection on a list of PIL images. Returns list of dicts."""
    import torch

    results = []
    model.eval()
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        target_size = torch.tensor([img.size[::-1]], device=device)  # (h, w)
        post = processor.post_process_object_detection(
            outputs, target_sizes=target_size, threshold=threshold
        )[0]
        results.append(
            {
                "scores": post["scores"].cpu(),
                "labels": post["labels"].cpu(),
                "boxes": post["boxes"].cpu(),  # xyxy in original image coords
            }
        )
    return results


def _draw_boxes(image, boxes, labels, scores, id2label, color: str = "lime"):
    """Draw bounding boxes on a PIL image. Returns a new PIL image."""
    from PIL import ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=max(14, img.width // 60))
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes.tolist(), labels.tolist(), scores.tolist()):
        x0, y0, x1, y1 = box
        width = max(2, img.width // 400)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        name = id2label.get(int(label), str(int(label)))
        caption = f"{name} {score:.2f}"
        text_bg = draw.textbbox((x0, y0), caption, font=font)
        draw.rectangle(text_bg, fill=color)
        draw.text((x0, y0), caption, fill="black", font=font)
    return img


def _img_to_data_uri(img, max_dim: int = 800) -> str:
    """PIL image → base64 data URI, downscaled for the report."""
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------
# Task 3: Evaluate — COCO mAP, base vs fine-tuned
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def evaluate(
    model_name: str,
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    threshold: float = 0.3,
) -> str:
    """Compute COCO mAP for base and fine-tuned models on the val split."""
    import torch
    from PIL import Image
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    log.info("Starting evaluation...")
    await flyte.report.replace.aio("<h2>Evaluation</h2><p>Loading val split...</p>")
    await flyte.report.flush.aio()

    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    val_json = os.path.join(data_path, "val.json")

    with open(val_json) as f:
        val_coco = json.load(f)

    images_by_id = {im["id"]: im for im in val_coco["images"]}
    anns_by_image: dict[int, list] = {}
    for a in val_coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    id2label = {c["id"]: c["name"] for c in val_coco["categories"]}

    pil_images = []
    targets = []
    for img_id, meta in images_by_id.items():
        path = os.path.join(images_root, os.path.basename(meta["file_name"]))
        if not os.path.exists(path):
            path = os.path.join(images_root, meta["file_name"])
        pil_images.append(Image.open(path).convert("RGB"))
        boxes_xyxy = []
        labels = []
        for a in anns_by_image.get(img_id, []):
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        targets.append(
            {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def score_model(name: str, model_path: str, use_pretrained_labels: bool):
        log.info(f"Scoring: {name} ({model_path})")
        processor = AutoImageProcessor.from_pretrained(model_path, token=HF_TOKEN)
        kwargs = {"token": HF_TOKEN}
        if not use_pretrained_labels:
            kwargs.update(
                id2label=id2label,
                label2id={v: k for k, v in id2label.items()},
                ignore_mismatched_sizes=True,
            )
        model = AutoModelForObjectDetection.from_pretrained(model_path, **kwargs).to(device)
        preds = _run_inference(model, processor, pil_images, device, threshold=threshold)

        formatted_preds = [
            {"boxes": p["boxes"], "scores": p["scores"], "labels": p["labels"]}
            for p in preds
        ]

        # When scoring the pretrained base against custom labels its predictions
        # are in COCO label space (80 classes), so mAP against our 2 classes is
        # near-zero by construction. We still report it to show the lift.
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(formatted_preds, targets)

        def to_python(v):
            # torchmetrics returns per-class arrays for keys like `map_per_class`;
            # only call .item() on true scalars, tolist() otherwise.
            if hasattr(v, "numel"):
                return v.item() if v.numel() == 1 else v.tolist()
            return v

        result = {k: to_python(v) for k, v in metric.compute().items()}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result, preds

    base_metrics, base_preds = score_model(
        "base", model_name, use_pretrained_labels=True
    )

    ft_path = await finetuned_dir.download()
    ft_metrics, ft_preds = score_model("finetuned", ft_path, use_pretrained_labels=False)

    log.info(f"Base mAP: {base_metrics.get('map', 0):.3f}")
    log.info(f"Fine-tuned mAP: {ft_metrics.get('map', 0):.3f}")

    rows = []
    for key in ["map", "map_50", "map_75", "mar_1", "mar_10"]:
        rows.append(
            f"<tr><td>{key}</td>"
            f"<td>{base_metrics.get(key, 0):.3f}</td>"
            f"<td>{ft_metrics.get(key, 0):.3f}</td></tr>"
        )
    table = (
        "<table><tr><th>Metric</th><th>Base</th><th>Fine-tuned</th></tr>"
        + "".join(rows)
        + "</table>"
    )

    await flyte.report.replace.aio(
        f"<h2>Evaluation — COCO mAP</h2>"
        f"<p>Val images: {len(pil_images)} | Threshold: {threshold}</p>"
        f"{table}"
        f"<p><i>Note: the base model was pretrained on COCO's 80 classes, so it "
        f"emits boxes labelled with COCO categories rather than ours — "
        f"hence the near-zero base mAP. The lift comes from teaching the "
        f"decoder to predict our category ids.</i></p>"
    )
    await flyte.report.flush.aio()

    return json.dumps(
        {
            "base": {k: round(v, 4) for k, v in base_metrics.items() if isinstance(v, (int, float))},
            "finetuned": {k: round(v, 4) for k, v in ft_metrics.items() if isinstance(v, (int, float))},
            "num_val_images": len(pil_images),
        }
    )


# ------------------------------------------------------------------
# Task 4: Inference demo — render bboxes on val images
# ------------------------------------------------------------------

@gpu_env.task(report=True)
async def inference_demo(
    finetuned_dir: flyte.io.Dir,
    data_dir: flyte.io.Dir,
    threshold: float = 0.3,
    max_images: int = 8,
) -> str:
    """Run the fine-tuned model on val images, render bboxes, embed in the report."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    data_path = await data_dir.download()
    images_root = os.path.join(data_path, "images")
    val_json = os.path.join(data_path, "val.json")

    with open(val_json) as f:
        val_coco = json.load(f)

    id2label = {c["id"]: c["name"] for c in val_coco["categories"]}
    metas = val_coco["images"][:max_images]
    anns_by_image: dict[int, list] = {}
    for a in val_coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    pil_images = []
    gt_per_image = []
    for meta in metas:
        path = os.path.join(images_root, os.path.basename(meta["file_name"]))
        if not os.path.exists(path):
            path = os.path.join(images_root, meta["file_name"])
        pil_images.append(Image.open(path).convert("RGB"))

        boxes_xyxy = []
        labels = []
        for a in anns_by_image.get(meta["id"], []):
            x, y, w, h = a["bbox"]
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
        gt_per_image.append(
            {
                "boxes": torch.tensor(boxes_xyxy, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.long),
                "scores": torch.ones(len(labels)),
            }
        )

    ft_path = await finetuned_dir.download()
    processor = AutoImageProcessor.from_pretrained(ft_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForObjectDetection.from_pretrained(ft_path).to(device)

    preds = _run_inference(model, processor, pil_images, device, threshold=threshold)

    html_blocks = []
    for img, pred, gt in zip(pil_images, preds, gt_per_image):
        pred_img = _draw_boxes(img, pred["boxes"], pred["labels"], pred["scores"], id2label, color="lime")
        gt_img = _draw_boxes(img, gt["boxes"], gt["labels"], gt["scores"], id2label, color="red")
        html_blocks.append(
            f"""
<div style="display:flex; gap:8px; margin:12px 0;">
  <div><p><b>Ground truth</b> ({len(gt['labels'])} boxes)</p>
    <img src="{_img_to_data_uri(gt_img)}" style="max-width:380px;" /></div>
  <div><p><b>Predictions</b> ({len(pred['labels'])} boxes, threshold={threshold})</p>
    <img src="{_img_to_data_uri(pred_img)}" style="max-width:380px;" /></div>
</div>"""
        )

    await flyte.report.replace.aio(
        f"<h2>Inference Demo — fine-tuned RT-DETR</h2>"
        f"<p>Showing {len(pil_images)} val image(s). Green = predictions, red = ground truth.</p>"
        + "".join(html_blocks)
    )
    await flyte.report.flush.aio()

    return json.dumps(
        {
            "num_images": len(pil_images),
            "predictions_per_image": [len(p["labels"]) for p in preds],
        }
    )


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@cpu_env.task(report=True)
async def pipeline(
    model_name: str = "PekingU/rtdetr_v2_r18vd",
    dataset_repo: str = "sagecodes/union_swag_coco",
    annotations_path: str = "swag/train.json",
    images_subdir: str = "swag/images",
    epochs: int = 30,
    lr: float = 5e-5,
    batch_size: int = 4,
    val_fraction: float = 0.2,
    threshold: float = 0.3,
    demo_images: int = 8,
) -> str:
    """
    End-to-end RT-DETRv2 fine-tuning pipeline.

    1. Download COCO dataset from HuggingFace and split train/val
    2. Fine-tune RT-DETRv2 on the train split
    3. Evaluate: COCO mAP comparison (base vs fine-tuned)
    4. Inference demo: render bounding boxes on val images
    """
    log.info(f"Pipeline: {model_name} | dataset={dataset_repo}")

    await flyte.report.replace.aio(
        f"<h2>RT-DETRv2 Object Detection Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Dataset:</b> {dataset_repo}</p>"
        f"<p>Step 1/4: Preparing data...</p>"
    )
    await flyte.report.flush.aio()

    data_dir = await prepare_data(
        dataset_repo=dataset_repo,
        annotations_path=annotations_path,
        images_subdir=images_subdir,
        val_fraction=val_fraction,
    )

    await flyte.report.replace.aio(
        f"<h2>RT-DETRv2 Object Detection Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 2/4: Fine-tuning...</p>"
    )
    await flyte.report.flush.aio()

    finetuned_dir = await train(model_name, data_dir, epochs, lr, batch_size)

    await flyte.report.replace.aio(
        f"<h2>RT-DETRv2 Object Detection Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 3/4: Evaluating...</p>"
    )
    await flyte.report.flush.aio()

    metrics_json = await evaluate(model_name, finetuned_dir, data_dir, threshold)
    metrics = json.loads(metrics_json)

    await flyte.report.replace.aio(
        f"<h2>RT-DETRv2 Object Detection Pipeline</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p>Step 4/4: Rendering inference demo...</p>"
    )
    await flyte.report.flush.aio()

    demo_json = await inference_demo(finetuned_dir, data_dir, threshold, demo_images)

    base_map = metrics["base"].get("map", 0)
    ft_map = metrics["finetuned"].get("map", 0)
    ft_map50 = metrics["finetuned"].get("map_50", 0)

    await flyte.report.replace.aio(
        f"<h2>Pipeline Complete</h2>"
        f"<p><b>Model:</b> {model_name}</p>"
        f"<p><b>Val images:</b> {metrics['num_val_images']}</p>"
        f"<p><b>Base mAP:</b> {base_map:.3f}</p>"
        f"<p><b>Fine-tuned mAP:</b> {ft_map:.3f} (mAP@50: {ft_map50:.3f})</p>"
    )
    await flyte.report.flush.aio()

    log.info(f"Pipeline complete. Fine-tuned mAP: {ft_map:.3f}")
    return json.dumps({"metrics": metrics, "demo": json.loads(demo_json)})
