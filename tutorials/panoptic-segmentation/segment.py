"""
Panoptic segmentation with Mask2Former, plus overlay rendering and a scene inventory.

Panoptic segmentation labels *every* pixel: "things" (countable objects — a truck, three
people) get individual instance masks, and "stuff" (amorphous regions — sky, road, grass)
get one mask each. It is the most complete form of image understanding in one pass, and it
subsumes both detection and semantic segmentation.

Mask2Former returns one entry per segment as {score, label, mask}. Everything here turns
those into a coloured overlay, labelled boxes around the things, and a per-class inventory.
"""

# COCO's own thing/stuff split, by label name suffix. Panoptic "stuff" classes are the
# merged/background regions; everything else is a countable object.
STUFF_HINTS = ("-merged", "-other", "-stuff", "wall", "sky", "road", "pavement", "floor",
               "ceiling", "grass", "dirt", "sand", "water", "sea", "river", "snow",
               "mountain", "gravel", "platform", "railroad", "sidewalk", "building",
               "house", "fence", "bridge", "tree", "rock", "roof", "field", "clouds",
               "playingfield", "banner", "curtain", "rug", "food-other", "textile")

MODEL_ID = "facebook/mask2former-swin-base-coco-panoptic"
_CACHE: dict = {}


def is_stuff(label: str) -> bool:
    lab = label.lower()
    return any(h in lab for h in STUFF_HINTS)


def load_pipeline(model_id: str = MODEL_ID):
    """Load and cache the segmentation pipeline (once per warm replica).

    NOTE: Mask2Former's post-processing requires `scipy`; without it the failure is an
    opaque `requires_backends` ImportError at first inference, not at import.
    """
    import torch
    from transformers import pipeline

    if model_id not in _CACHE:
        device = 0 if torch.cuda.is_available() else -1
        _CACHE.clear()
        _CACHE[model_id] = pipeline("image-segmentation", model=model_id, device=device)
    return _CACHE[model_id]


def segment(image, model_id: str = MODEL_ID):
    """RGB PIL image -> list of {score, label, mask(np.bool), is_stuff, area, bbox}."""
    import numpy as np

    raw = load_pipeline(model_id)(image)
    out = []
    for s in raw:
        m = np.asarray(s["mask"]) > 127
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        out.append({
            "score": float(s.get("score") or 1.0),
            "label": s["label"],
            "mask": m,
            "is_stuff": is_stuff(s["label"]),
            "area": int(m.sum()),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        })
    out.sort(key=lambda s: -s["area"])
    return out


# ------------------------------------------------------------------
# Colour + rendering
# ------------------------------------------------------------------

def _distinct_color(i: int):
    """Golden-angle hue walk — maximally distinct colours without a fixed palette size."""
    import colorsys

    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.68, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def overlay(image, segments, alpha: float = 0.55, draw_boxes: bool = True):
    """Colour every segment, then box + label the things. Returns an RGB uint8 array."""
    import numpy as np
    from PIL import Image, ImageDraw

    base = np.asarray(image.convert("RGB")).astype("float32")
    colored = base.copy()
    colors = {}
    for i, s in enumerate(segments):
        c = _distinct_color(i)
        colors[id(s)] = c
        m = s["mask"]
        for k in range(3):
            colored[:, :, k][m] = colored[:, :, k][m] * (1 - alpha) + c[k] * alpha

    canvas = Image.fromarray(np.clip(colored, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(canvas)
    if draw_boxes:
        for s in segments:
            if s["is_stuff"]:
                continue  # boxes only make sense on countable things
            x0, y0, x1, y1 = s["bbox"]
            c = colors[id(s)]
            d.rectangle([x0, y0, x1, y1], outline=c, width=2)
            txt = f"{s['label']} {s['score']:.2f}"
            ty = max(0, y0 - 11)
            try:
                tw = d.textlength(txt)
            except Exception:  # noqa: BLE001
                tw = 6.0 * len(txt)
            d.rectangle([x0, ty, x0 + tw + 5, ty + 11], fill=(0, 0, 0))
            d.text((x0 + 3, ty), txt, fill=c)
    return np.asarray(canvas)


def inventory(segments):
    """Per-class counts, split into things and stuff, for the scene inventory."""
    from collections import Counter

    things = Counter(s["label"] for s in segments if not s["is_stuff"])
    stuff = Counter(s["label"] for s in segments if s["is_stuff"])
    return {
        "things": dict(things.most_common()),
        "stuff": dict(stuff.most_common()),
        "n_things": sum(things.values()),
        "n_stuff": len(stuff),
        "n_segments": len(segments),
    }


# ------------------------------------------------------------------
# Ground-truth panoptic decode + comparison
# ------------------------------------------------------------------

def decode_gt(label_png):
    """
    Decode a COCO panoptic PNG into a per-pixel segment-id map.

    COCO encodes the segment id per pixel as R + G*256 + B*256^2. Returns (id_map, colored)
    where `colored` gives each ground-truth segment its own distinct colour so it can be
    shown side by side with the prediction.
    """
    import numpy as np

    arr = np.asarray(label_png.convert("RGB")).astype("uint32")
    ids = arr[:, :, 0] + arr[:, :, 1] * 256 + arr[:, :, 2] * 256 * 256
    uniq = [u for u in np.unique(ids) if u != 0]
    colored = np.zeros(arr.shape, dtype="uint8")
    for i, u in enumerate(uniq):
        colored[ids == u] = _distinct_color(i)
    return ids, colored, len(uniq)


def coverage(segments, shape):
    """Fraction of pixels the prediction assigned to some segment — panoptic should be ~1."""
    import numpy as np

    covered = np.zeros(shape, dtype=bool)
    for s in segments:
        covered |= s["mask"]
    return float(covered.mean())
