"""
Open-vocabulary detection over synthetic driving frames.

Uses OWLv2, which takes free-text queries rather than a fixed class list. That matters
here: the useful AV question is not "label everything COCO knows about" but "does this
clip actually contain an ambulance", and the prompt list can be changed without retraining
anything.

Two things this measures that rendering alone cannot:

  * **Label verification** — a scenario filed under `emergency` should contain an emergency
    vehicle. Sometimes it does not, or the vehicle is out of frame for most of the clip.
  * **The sim-to-real gap** — OWLv2 was trained on real photographs. Its confidence on
    Omniverse renders is measurably lower than on real imagery, and reporting that
    distribution turns NVIDIA's qualitative warning into a number.
"""

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# AV-relevant prompts. Phrased as "a <thing>" because OWLv2 is trained on caption-like
# text and bare nouns score noticeably worse.
DEFAULT_PROMPTS = [
    "a police car",
    "an ambulance",
    "a fire truck",
    "a car",
    "a truck",
    "a bus",
    "a person",
    "a traffic light",
    "a traffic cone",
]

# Single score floor for drawn boxes.
#
# Measured over 8 camera-clips, raising the floor from 0.12 to 0.32 cuts detections
# 127 -> 58 and removes the false "car" that fires on the ego vehicle's own bonnet
# (~0.20). It also drops most traffic lights, which score 0.11-0.17 — a real cost, but
# a clean frame of confident boxes reads better than a busy one full of maybes.
# The emergency vehicle measures 0.471, so label verification is unaffected.
DETECTION_THRESHOLD = 0.32

# Prompts that indicate the emergency scenario family actually delivered on its label.
EMERGENCY_PROMPTS = {"a police car", "an ambulance", "a fire truck"}

# Box colours per prompt group.
PROMPT_COLORS = {
    "a police car": "#ef4444",
    "an ambulance": "#ef4444",
    "a fire truck": "#ef4444",
    "a car": "#38bdf8",
    "a truck": "#34d399",
    "a bus": "#a78bfa",
    "a person": "#f472b6",
    "a traffic light": "#22d3ee",
    "a traffic cone": "#fbbf24",
}
DEFAULT_COLOR = "#e2e8f0"

_CACHE: dict = {}


def load_model(model_id: str = MODEL_ID):
    """Load and cache the processor/model.

    Cached at module scope: under a ReusePolicy the container is reused across tasks, so
    the ~600 MB of weights deserialize once per replica rather than once per scenario.
    NOTE: the OWLv2 image processor requires `scipy`; without it the failure is an opaque
    `requires_backends` ImportError at first call, not at import.
    """
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    if model_id not in _CACHE:
        proc = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id).eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        _CACHE.clear()
        _CACHE[model_id] = (proc, model, device)
    return _CACHE[model_id]


def detect(images, prompts=None, threshold: float = 0.10, model_id: str = MODEL_ID):
    """
    Run open-vocabulary detection over a list of PIL images.

    Returns one list of {prompt, score, box} per image.
    """
    import torch

    prompts = prompts or DEFAULT_PROMPTS
    proc, model, device = load_model(model_id)

    out = []
    for img in images:
        inputs = proc(text=[prompts], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            pred = model(**inputs)
        sizes = torch.tensor([img.size[::-1]]).to(device)
        res = proc.post_process_grounded_object_detection(
            pred, threshold=threshold, target_sizes=sizes
        )[0]
        dets = [
            {"prompt": prompts[int(l)], "score": float(s),
             "box": [float(v) for v in b]}
            for s, l, b in zip(res["scores"].tolist(), res["labels"].tolist(),
                               res["boxes"].tolist())
        ]
        dets = [d for d in dets if d["score"] >= DETECTION_THRESHOLD]
        dets.sort(key=lambda d: -d["score"])
        out.append(dets)
    return out


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def draw_detections(img, dets, min_score: float = 0.0, base_alpha: int = 46,
                    max_alpha: int = 132):
    """
    Draw translucent filled boxes onto a copy of `img`.

    PIL will not alpha-blend a fill drawn straight onto an RGB image — the alpha channel is
    silently dropped and the box comes out opaque, hiding the very thing it marks. So the
    boxes are drawn onto a transparent RGBA overlay and composited in one pass.

    Fill opacity scales with confidence, so a 0.45 detection reads as more solid than a
    0.13 one. On synthetic footage where scores cluster low (0.18-0.34) that is the
    difference between "the model is sure" and "the model is guessing" being visible at a
    glance rather than requiring the label to be read.
    """
    from PIL import Image, ImageDraw

    canvas = img.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    for det in dets:
        if det["score"] < min_score:
            continue
        x0, y0, x1, y1 = det["box"]
        r, g, b = _rgb(PROMPT_COLORS.get(det["prompt"], DEFAULT_COLOR))
        # Confidence -> opacity. Clamped so even a weak box stays visible.
        t = max(0.0, min(1.0, det["score"] / 0.5))
        alpha = int(base_alpha + t * (max_alpha - base_alpha))
        d.rectangle([x0, y0, x1, y1], fill=(r, g, b, alpha), outline=(r, g, b, 255), width=2)

        label = f"{det['prompt'].removeprefix('a ').removeprefix('an ')} {det['score']:.2f}"
        ty = max(0, y0 - 11)
        # Dark plate behind the text so labels stay readable over bright sky or headlights.
        try:
            tw = d.textlength(label)
        except Exception:  # noqa: BLE001 — older Pillow without textlength
            tw = 6.0 * len(label)
        d.rectangle([x0, ty, x0 + tw + 5, ty + 11], fill=(0, 0, 0, 150))
        d.text((x0 + 3, ty), label, fill=(r, g, b, 255))

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def summarize(per_frame: list[list[dict]], prompts=None) -> dict:
    """Aggregate detections into hit rates and a confidence distribution."""
    prompts = prompts or DEFAULT_PROMPTS
    n = max(len(per_frame), 1)
    hits = {p: 0 for p in prompts}
    scores = {p: [] for p in prompts}
    for dets in per_frame:
        seen = set()
        for det in dets:
            scores[det["prompt"]].append(det["score"])
            seen.add(det["prompt"])
        for p in seen:
            hits[p] += 1
    all_scores = [s for v in scores.values() for s in v]
    return {
        "frames": len(per_frame),
        "hit_rate": {p: hits[p] / n for p in prompts},
        "counts": {p: len(scores[p]) for p in prompts},
        "mean_score": {p: (sum(v) / len(v) if v else 0.0) for p, v in scores.items()},
        "overall_mean_score": (sum(all_scores) / len(all_scores)) if all_scores else 0.0,
        "total_detections": len(all_scores),
        "emergency_hit": any(hits[p] > 0 for p in prompts if p in EMERGENCY_PROMPTS),
    }
