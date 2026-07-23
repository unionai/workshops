"""
Monocular depth estimation with Depth Anything V2, plus alignment and metrics.

Depth Anything V2 predicts **relative inverse depth** (disparity): a single forward pass
turns one RGB image into a per-pixel value where nearer surfaces are larger. It has no
absolute scale — it does not know whether a room is 3 m or 30 m across — so to compare
against a metric ground-truth sensor the prediction is aligned to it with a per-image
least-squares scale and shift. That is the standard protocol for scale-invariant depth,
and it is why the numbers here are honest rather than flattering.
"""

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_CACHE: dict = {}


def load_pipeline(model_id: str = MODEL_ID):
    """Load and cache the depth pipeline (weights ~100 MB, once per warm replica)."""
    import torch
    from transformers import pipeline

    if model_id not in _CACHE:
        device = 0 if torch.cuda.is_available() else -1
        _CACHE.clear()
        _CACHE[model_id] = pipeline("depth-estimation", model=model_id, device=device)
    return _CACHE[model_id]


def predict(image, model_id: str = MODEL_ID):
    """RGB PIL image -> predicted relative inverse-depth array (float32, H x W)."""
    import numpy as np

    out = load_pipeline(model_id)(image)
    return np.asarray(out["depth"], dtype="float32")


# ------------------------------------------------------------------
# Alignment and metrics
# ------------------------------------------------------------------

def align_to_metric(pred, gt, valid_min: float = 0.1, valid_max: float = 10.0):
    """
    Least-squares align the (unitless, inverse) prediction to metric ground truth.

    Fits `gt ~= a * pred + b` over valid pixels and returns the aligned prediction in
    metres. The fitted `a` is negative because the model outputs inverse depth — nearer is
    larger — while ground truth is forward depth in metres. Returns (aligned, mask, a, b).
    """
    import numpy as np

    mask = np.isfinite(gt) & (gt > valid_min) & (gt < valid_max)
    if mask.sum() < 100:
        return None, mask, 0.0, 0.0
    A = np.stack([pred[mask], np.ones(int(mask.sum()))], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gt[mask], rcond=None)
    aligned = a * pred + b
    # Depth cannot be negative; clamp so error and colouring stay sane.
    aligned = np.clip(aligned, valid_min, None)
    return aligned, mask, float(a), float(b)


def metrics(aligned, gt, mask):
    """Standard scale-invariant depth metrics over the valid region."""
    import numpy as np

    p = aligned[mask]
    g = gt[mask]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    return {
        "abs_rel": absrel,
        "rmse": rmse,
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
    }


# ------------------------------------------------------------------
# Colourisation
# ------------------------------------------------------------------

def _turbo(t):
    """
    Turbo colormap (Google, Anton Mikhailov) via the standard polynomial approximation.

    Turbo is used instead of jet because it is perceptually smoother and does not invent
    false banding — which on a depth map reads as fake surfaces.
    """
    import numpy as np

    t = np.clip(t, 0.0, 1.0)
    r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    rgb = np.stack([r, g, b], axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype("uint8")


def colorize(depth, mask=None, invert=False, lo_pct=2, hi_pct=98):
    """
    Depth array -> RGB uint8 via turbo, percentile-normalised.

    `invert=True` flips so that near reads warm and far reads cool regardless of whether
    the source is depth or inverse depth, keeping predicted and ground-truth panels
    visually comparable.
    """
    import numpy as np

    d = depth.astype("float32").copy()
    valid = mask if mask is not None else np.isfinite(d) & (d > 0)
    if valid.sum() < 10:
        return np.zeros(d.shape + (3,), dtype="uint8")
    lo, hi = np.percentile(d[valid], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    t = (d - lo) / (hi - lo)
    if invert:
        t = 1.0 - t
    rgb = _turbo(t)
    rgb[~valid] = (14, 16, 22)  # dark for invalid/no-return pixels
    return rgb


def error_map(aligned, gt, mask, cap: float = 1.0):
    """Per-pixel absolute error in metres -> RGB (dark = accurate, bright = wrong)."""
    import numpy as np

    err = np.zeros(gt.shape, dtype="float32")
    err[mask] = np.abs(aligned[mask] - gt[mask])
    t = np.clip(err / cap, 0, 1)
    # black -> red -> yellow "heat" ramp, intuitive for error
    r = np.clip(t * 2, 0, 1)
    g = np.clip(t * 2 - 1, 0, 1)
    rgb = (np.stack([r, g, np.zeros_like(t)], axis=-1) * 255).astype("uint8")
    rgb[~mask] = (14, 16, 22)
    return rgb


# ------------------------------------------------------------------
# The money shot: 3D parallax from a single photo
# ------------------------------------------------------------------

def parallax_frames(rgb, pred, n_frames: int = 12, max_shift: float = 14.0):
    """
    Fake a small camera sweep by shifting pixels horizontally in proportion to depth.

    Nearer pixels (larger inverse-depth) move more than far ones, which produces the
    parallax the eye reads as 3D. Gaps opened behind foreground objects are filled from
    the last valid column so the result stays clean. Returns a list of RGB uint8 frames
    that play back as a subtle wobble — the single most "wow" way to show that a flat
    photo now has geometry.
    """
    import numpy as np

    h, w = pred.shape
    # Normalise inverse-depth to 0..1; nearer -> 1 -> larger shift.
    d = pred.astype("float32")
    d = (d - d.min()) / (np.ptp(d) or 1.0)

    frames = []
    for k in range(n_frames):
        # smooth back-and-forth sweep
        phase = np.sin(2 * np.pi * k / n_frames)
        shift = (d * max_shift * phase).astype("int32")
        out = np.zeros_like(rgb)
        filled = np.zeros((h, w), dtype=bool)
        # Paint far pixels first, near last, so foreground overwrites — correct occlusion.
        order = np.argsort(d, axis=None)  # ascending: far -> near
        ys, xs = np.unravel_index(order, (h, w))
        nx = np.clip(xs + shift[ys, xs], 0, w - 1)
        out[ys, nx] = rgb[ys, xs]
        filled[ys, nx] = True
        # Fill disocclusion holes left-to-right from the nearest painted pixel.
        for row in range(h):
            last = None
            fr = filled[row]
            orow = out[row]
            for col in range(w):
                if fr[col]:
                    last = orow[col]
                elif last is not None:
                    orow[col] = last
        frames.append(out)
    return frames
