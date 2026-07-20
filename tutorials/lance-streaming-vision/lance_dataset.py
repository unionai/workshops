"""Synthetic vision dataset, Lance conversion, and streaming dataloader.

This module has no Flyte dependency so you can exercise it directly:

    python lance_dataset.py

It models a common computer-vision data-loading problem: a dataset stored as a
huge number of tiny image files. Each sample is one small image PNG *plus* a
label mask PNG, and samples are multiplied by a per-scene variant dimension
(multiple views / crops / augmentations). A modest dataset therefore explodes
into thousands of tiny objects in storage, and training burns most of its
wall-clock opening a fresh object-store connection per file.

The fix modeled here: convert that pile of tiny files **once** into a single
Lance dataset (columnar, streaming-optimized, multimodal — image bytes, mask
bytes, and structured labels live together in one row), then stream it row-group
by row-group across many training runs.
"""

from __future__ import annotations

import io
import json
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    import lance

# ----------------------------------------------------------------------------
# Arrow schema for the Lance dataset
# ----------------------------------------------------------------------------
# One row == one (scene, view) sample. The image and mask are stored as encoded
# PNG bytes (Lance streams these lazily), and the labels ride along in the same
# row — this is the "multimodal in one row" property that makes Lance a good fit
# for training data.
LANCE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("group_id", pa.string()),
        ("scene_idx", pa.int32()),
        ("view_idx", pa.int32()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("image", pa.large_binary()),  # encoded PNG bytes
        ("mask", pa.large_binary()),   # encoded PNG label mask
        ("bboxes", pa.list_(pa.list_(pa.float32(), 4))),  # [x, y, w, h] per object
        ("object_classes", pa.list_(pa.int32())),          # class id per object
    ]
)

_OBJECT_COLORS = [(198, 40, 40), (21, 101, 192), (245, 245, 245), (255, 179, 0)]
NUM_CLASSES = len(_OBJECT_COLORS)


@dataclass
class Sample:
    """A single generated (scene, view) sample, in memory."""

    sample_id: str
    group_id: str
    scene_idx: int
    view_idx: int
    width: int
    height: int
    image_png: bytes
    mask_png: bytes
    bboxes: list[list[float]]
    object_classes: list[int]


# ----------------------------------------------------------------------------
# Synthetic scene generation
# ----------------------------------------------------------------------------

def render_scene(rng: random.Random, img_size: int) -> Sample:
    """Draw one synthetic scene: a textured background, 1–4 objects (colored
    boxes with a class label), and a matching instance mask. Returns encoded PNG
    bytes + labels.
    """
    bg = (46 + rng.randint(-6, 6), 125 + rng.randint(-8, 8), 50 + rng.randint(-6, 6))
    img = Image.new("RGB", (img_size, img_size), bg)
    mask = Image.new("L", (img_size, img_size), 0)
    draw = ImageDraw.Draw(img)
    mdraw = ImageDraw.Draw(mask)

    # A few background lines, so different views look distinct from each other.
    for _ in range(3):
        y = rng.randint(0, img_size)
        draw.line([(0, y), (img_size, y)], fill=(230, 230, 230), width=1)

    n_objects = rng.randint(1, 4)
    bboxes: list[list[float]] = []
    classes: list[int] = []
    for i in range(n_objects):
        w = rng.randint(img_size // 10, img_size // 5)
        h = rng.randint(img_size // 6, img_size // 3)
        x = rng.randint(0, max(1, img_size - w))
        y = rng.randint(0, max(1, img_size - h))
        cls = rng.randint(0, NUM_CLASSES - 1)
        draw.rectangle([x, y, x + w, y + h], fill=_OBJECT_COLORS[cls])
        mdraw.rectangle([x, y, x + w, y + h], fill=i + 1)
        draw.text((x + 2, y + 2), str(cls), fill=(20, 20, 20))
        bboxes.append([float(x), float(y), float(w), float(h)])
        classes.append(cls)

    img_buf, mask_buf = io.BytesIO(), io.BytesIO()
    img.save(img_buf, format="PNG")
    mask.save(mask_buf, format="PNG")
    return Sample(
        sample_id="",  # filled in by the caller
        group_id="",
        scene_idx=0,
        view_idx=0,
        width=img_size,
        height=img_size,
        image_png=img_buf.getvalue(),
        mask_png=mask_buf.getvalue(),
        bboxes=bboxes,
        object_classes=classes,
    )


def write_tiny_files(
    root: str,
    num_groups: int,
    scenes_per_group: int,
    views_per_scene: int,
    img_size: int,
    seed: int = 0,
) -> dict:
    """Write the dataset the way it commonly lives today: a deep tree of tiny
    per-(group, scene, view) files. Layout:

        {root}/{group_id}/scene_{idx:04d}/view_{v:02d}.png       # image
        {root}/{group_id}/scene_{idx:04d}/view_{v:02d}.mask.png  # label mask
        {root}/{group_id}/scene_{idx:04d}/view_{v:02d}.json      # bboxes + classes

    Returns a stats dict (file counts, byte size).
    """
    rng = random.Random(seed)
    n_files = 0
    n_bytes = 0
    n_samples = 0
    for g in range(num_groups):
        group_id = f"group_{g:02d}"
        for s in range(scenes_per_group):
            scene_dir = os.path.join(root, group_id, f"scene_{s:04d}")
            os.makedirs(scene_dir, exist_ok=True)
            for v in range(views_per_scene):
                scene = render_scene(rng, img_size)
                base = os.path.join(scene_dir, f"view_{v:02d}")
                label = {
                    "group_id": group_id,
                    "scene_idx": s,
                    "view_idx": v,
                    "bboxes": scene.bboxes,
                    "object_classes": scene.object_classes,
                }
                with open(f"{base}.png", "wb") as fh:
                    fh.write(scene.image_png)
                with open(f"{base}.mask.png", "wb") as fh:
                    fh.write(scene.mask_png)
                with open(f"{base}.json", "w") as fh:
                    json.dump(label, fh)
                n_files += 3
                n_bytes += len(scene.image_png) + len(scene.mask_png)
                n_samples += 1

    return {
        "num_samples": n_samples,
        "num_files": n_files,
        "num_bytes": n_bytes,
        "source": "synthetic",
        "num_classes": NUM_CLASSES,
        "detail": f"{num_groups} groups × {scenes_per_group} scenes × {views_per_scene} views @ {img_size}px",
    }


# ----------------------------------------------------------------------------
# Real dataset: CPPE-5 (medical PPE detection) written to the tiny-file layout
# ----------------------------------------------------------------------------
# CPPE-5: ~1k images, 5 classes, openly licensed, no auth. We download it once
# and *materialize each image as its own tiny file* — deliberately recreating
# the "swarm of tiny objects in storage" problem on real data, so the rest of
# the pipeline (convert → benchmark → train) runs unchanged.
CPPE5_REPO = "rishitdagli/cppe-5"
CPPE5_CLASSES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]


def write_cppe5_tiny_files(
    root: str,
    split: str = "train",
    max_images: int | None = 500,
    resize_max: int = 480,
    shard_size: int = 200,
) -> dict:
    """Download CPPE-5 and write each image as an individual tiny file (image
    PNG + label JSON) under a sharded folder tree. Images larger than
    `resize_max` on their long side are downscaled (boxes scaled to match) to
    keep the demo snappy.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(CPPE5_REPO, split=split)
    except Exception:
        ds = load_dataset(CPPE5_REPO, split=split, trust_remote_code=True)
    if max_images is not None:
        ds = ds.select(range(min(max_images, len(ds))))

    n_files = 0
    n_bytes = 0
    n_samples = 0
    for i, ex in enumerate(ds):
        img = ex["image"].convert("RGB")
        w, h = img.size
        boxes = ex["objects"]["bbox"]        # COCO [x, y, w, h], absolute px
        cats = ex["objects"]["category"]     # ints 0..4

        scale = 1.0
        if resize_max and max(w, h) > resize_max:
            scale = resize_max / max(w, h)
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
        sboxes = [[float(b[0]) * scale, float(b[1]) * scale,
                   float(b[2]) * scale, float(b[3]) * scale] for b in boxes]

        group = f"shard_{i // shard_size:03d}"
        scene_dir = os.path.join(root, group, f"scene_{i:05d}")
        os.makedirs(scene_dir, exist_ok=True)
        base = os.path.join(scene_dir, "view_00")
        label = {
            "group_id": group,
            "scene_idx": i,
            "view_idx": 0,
            "bboxes": sboxes,
            "object_classes": [int(c) for c in cats],
        }
        img.save(f"{base}.png", format="PNG")
        with open(f"{base}.json", "w") as fh:
            json.dump(label, fh)
        n_files += 2  # image + label (CPPE-5 ships no masks)
        n_bytes += os.path.getsize(f"{base}.png")
        n_samples += 1

    return {
        "num_samples": n_samples,
        "num_files": n_files,
        "num_bytes": n_bytes,
        "source": "cppe-5",
        "num_classes": len(CPPE5_CLASSES),
        "detail": f"CPPE-5 {split}, {n_samples} images, {len(CPPE5_CLASSES)} classes",
    }


# ----------------------------------------------------------------------------
# Convert tiny files -> a single Lance dataset
# ----------------------------------------------------------------------------

def build_lance_dataset(tiny_root: str, lance_uri: str, write_chunk: int = 512) -> dict:
    """Walk the tiny-file tree and write a single Lance dataset. Rows are
    flushed in chunks so memory stays bounded on large datasets.
    """
    import lance

    def _flush(rows: list[dict], mode: str):
        table = pa.Table.from_pylist(rows, schema=LANCE_SCHEMA)
        lance.write_dataset(table, lance_uri, mode=mode)

    rows: list[dict] = []
    n_rows = 0
    mode = "create"
    for dirpath, _dirs, files in os.walk(tiny_root):
        for name in sorted(files):
            if not name.endswith(".png") or name.endswith(".mask.png"):
                continue
            base = os.path.join(dirpath, name[: -len(".png")])
            rel = os.path.relpath(base, tiny_root)
            with open(base + ".png", "rb") as fh:
                image_bytes = fh.read()
            mask_bytes = b""
            if os.path.exists(base + ".mask.png"):
                with open(base + ".mask.png", "rb") as fh:
                    mask_bytes = fh.read()
            with open(base + ".json") as fh:
                label = json.load(fh)  # carries all metadata (layout-independent)

            with Image.open(io.BytesIO(image_bytes)) as im:
                width, height = im.width, im.height
            rows.append(
                {
                    "sample_id": rel.replace(os.sep, "/"),
                    "group_id": label.get("group_id", "group_00"),
                    "scene_idx": int(label.get("scene_idx", 0)),
                    "view_idx": int(label.get("view_idx", 0)),
                    "width": width,
                    "height": height,
                    "image": image_bytes,
                    "mask": mask_bytes,
                    "bboxes": [[float(v) for v in b] for b in label["bboxes"]],
                    "object_classes": [int(v) for v in label["object_classes"]],
                }
            )
            if len(rows) >= write_chunk:
                _flush(rows, mode)
                mode = "append"
                n_rows += len(rows)
                rows = []

    if rows:
        _flush(rows, mode)
        n_rows += len(rows)

    ds = lance.dataset(lance_uri)
    n_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _d, fs in os.walk(lance_uri)
        for f in fs
    )
    return {"num_rows": n_rows, "num_bytes": n_bytes, "num_fragments": len(ds.get_fragments())}


# ----------------------------------------------------------------------------
# Stream from Lance
# ----------------------------------------------------------------------------


def stream_batches(
    ds: "lance.LanceDataset", batch_size: int, columns: list[str] | None = None
):
    """Yield Arrow RecordBatches from the Lance dataset. This is the streaming
    read: Lance pulls only the requested columns, row-group by row-group, with a
    single dataset handle — no per-sample connection setup.
    """

    yield from ds.scanner(columns=columns, batch_size=batch_size).to_batches()


def read_batches(
    ds: "lance.LanceDataset", batch_size: int, columns: list[str], shuffle: bool
):
    """Yield Arrow batches from the Lance dataset. `shuffle=False` streams
    sequentially (fast, for eval); `shuffle=True` draws a fresh random row order
    each epoch and fetches it with `Dataset.take` — Lance's random access, which
    is what SGD needs and what sequential-only formats can't do."""

    if not shuffle:
        yield from ds.scanner(columns=columns, batch_size=batch_size).to_batches()
        return

    order = list(range(ds.count_rows()))
    random.shuffle(order)
    for i in range(0, len(order), batch_size):
        yield ds.take(order[i : i + batch_size], columns=columns)


def decode_image(png_bytes: bytes) -> "Image.Image":
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def num_detector_labels(ds: "lance.LanceDataset") -> int:
    """Number of labels for a torchvision detector = max class id + 2
    (+1 because we shift class ids off background=0, +1 for background itself).
    Derived by scanning the `object_classes` column so it fits either dataset.
    """
    max_cls = 0
    for batch in stream_batches(ds, batch_size=1024, columns=["object_classes"]):
        for classes in batch.column("object_classes").to_pylist():
            if classes:
                max_cls = max(max_cls, max(classes))
    return max_cls + 2


class LanceImageStream:
    """A torch `IterableDataset` over the Lance dataset.

    Yields (image_tensor CxHxW float32 in [0,1], target) where `target` is a
    torchvision-detection dict: {"boxes": FloatTensor[N,4] in xyxy, "labels":
    Int64Tensor[N]}. Class ids are shifted by +1 because torchvision detection
    reserves label 0 for background.
    """

    def __init__(
        self, ds: "lance.LanceDataset", batch_size: int = 64, shuffle: bool = False
    ):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        import numpy as np
        import torch

        cols = ["image", "bboxes", "object_classes", "width", "height"]
        for batch in read_batches(self.ds, self.batch_size, cols, self.shuffle):
            images = batch.column("image").to_pylist()
            bboxes = batch.column("bboxes").to_pylist()
            classes = batch.column("object_classes").to_pylist()
            widths = batch.column("width").to_pylist()
            heights = batch.column("height").to_pylist()
            for png, boxes, cls, w, h in zip(images, bboxes, classes, widths, heights):
                arr = np.asarray(decode_image(png), dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()

                xyxy, labels = [], []
                for (bx, by, bw, bh), c in zip(boxes, cls):
                    x1, y1 = bx, by
                    x2, y2 = min(bx + bw, float(w)), min(by + bh, float(h))
                    if x2 > x1 and y2 > y1:
                        xyxy.append([x1, y1, x2, y2])
                        labels.append(int(c) + 1)  # +1: reserve 0 for background
                if not xyxy:  # torchvision needs at least one valid box
                    xyxy, labels = [[0.0, 0.0, float(w), float(h)]], [1]

                target = {
                    "boxes": torch.tensor(xyxy, dtype=torch.float32),
                    "labels": torch.tensor(labels, dtype=torch.int64),
                }
                yield tensor, target


def detection_collate(batch):
    """Collate for torchvision detection: keep images and targets as lists,
    since each image has a variable number of objects."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_torch_dataset(
    ds: "lance.LanceDataset", batch_size: int = 64, shuffle: bool = False
):
    """Wrap `LanceImageStream` as a real `torch.utils.data.IterableDataset`."""
    import torch

    stream = LanceImageStream(ds, batch_size=batch_size, shuffle=shuffle)

    class _DS(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(stream)

    return _DS()


if __name__ == "__main__":
    # Tiny smoke test: generate -> convert -> stream, all locally.
    import shutil
    import tempfile

    import lance

    tmp = tempfile.mkdtemp(prefix="lance_smoke_")
    tiny = os.path.join(tmp, "tiny")
    lance_uri = os.path.join(tmp, "dataset.lance")
    print("Generating tiny files...")
    stats = write_tiny_files(tiny, num_groups=1, scenes_per_group=3, views_per_scene=4, img_size=64)
    print("  ", stats)
    print("Converting to Lance...")
    conv = build_lance_dataset(tiny, lance_uri)
    print("  ", conv)
    print("Streaming batches...")
    ds = lance.dataset(lance_uri)
    total = 0
    for batch in stream_batches(ds, batch_size=8, columns=["image", "bboxes"]):
        total += batch.num_rows
    print(f"  streamed {total} rows")
    shutil.rmtree(tmp)
    print("OK")
