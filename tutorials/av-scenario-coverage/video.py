"""
Video decoding and surround-view compositing for the scenario coverage pipeline.

Frames are decoded with PyAV (statically-linked FFmpeg, so nothing extra is needed in the
image) and composited into a single surround-view image per timestep. Compositing on the
server rather than shipping seven synchronised players means playback cannot drift: the
cameras are literally the same image.
"""

import io

# The 7-camera rig, laid out roughly as mounted on the vehicle. `front_wide` is the hero
# view and gets the centre-top slot at double width.
RIG_LAYOUT = [
    # (name, row, col, colspan)
    ("front_tele", 0, 0, 1),
    ("front_wide", 0, 1, 2),
    ("left_fisheye", 1, 0, 1),
    ("right_fisheye", 1, 2, 1),
    ("rear_left", 2, 0, 1),
    ("rear_fisheye", 2, 1, 1),
    ("rear_right", 2, 2, 1),
]

# Shorter 4-camera rig used by most campaigns.
DEFAULT_RIG = ["front_tele", "front_wide", "rear_left", "rear_right"]
FULL_RIG = [name for name, _, _, _ in RIG_LAYOUT]

LABEL_COLOR = (226, 232, 240)
PANEL_BG = (8, 11, 18)


def decode_frames(path: str, n_frames: int, size: tuple[int, int]):
    """
    Decode `n_frames` evenly spaced frames from a video, resized to `size`.

    Decodes sequentially and keeps the wanted indices rather than seeking: these clips are
    short (~460 frames) and seeking an MPEG-4 stream to an exact frame is both slower and
    less reliable than a linear pass.
    """
    import av
    from PIL import Image

    container = av.open(path)
    stream = container.streams.video[0]
    total = stream.frames or 0

    if total <= 0:
        frames = [f.to_image() for f in container.decode(video=0)]
        total = len(frames)
        if not total:
            return []
        idxs = {round(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)}
        return [f.resize(size, Image.BILINEAR) for i, f in enumerate(frames) if i in idxs]

    wanted = sorted({round(i * (total - 1) / max(n_frames - 1, 1)) for i in range(n_frames)})
    want = set(wanted)
    out = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in want:
            out.append(frame.to_image().resize(size, Image.BILINEAR))
            if len(out) == len(wanted):
                break
    container.close()
    return out


def composite_surround(frames_by_cam: dict, tile: tuple[int, int] = (320, 180),
                       gap: int = 4) -> list[bytes]:
    """
    Build one composited surround image per timestep.

    Returns JPEG bytes per frame. JPEG rather than PNG because these are photographic:
    at 480x270 a PNG frame is ~180 KB against ~22 KB for JPEG q78, and a seven-camera
    sequence is embedded in the report as base64.
    """
    from PIL import Image, ImageDraw

    present = [(n, r, c, cs) for n, r, c, cs in RIG_LAYOUT if frames_by_cam.get(n)]
    if not present:
        return []

    n_steps = min(len(v) for v in frames_by_cam.values() if v)
    tw, th = tile
    rows = max(r for _, r, _, _ in present) + 1
    cols = max(c + cs for _, _, c, cs in present)
    label_h = 15
    W = cols * tw + (cols + 1) * gap
    H = rows * (th + label_h) + (rows + 1) * gap

    out = []
    for k in range(n_steps):
        canvas = Image.new("RGB", (W, H), PANEL_BG)
        d = ImageDraw.Draw(canvas)
        for name, r, c, cs in present:
            fr = frames_by_cam[name][k]
            w = cs * tw + (cs - 1) * gap
            x = gap + c * (tw + gap)
            y = gap + r * (th + label_h + gap)
            canvas.paste(fr.resize((w, th), Image.BILINEAR), (x, y + label_h))
            d.text((x + 2, y + 2), name.replace("_", " "), fill=LABEL_COLOR)
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=78, optimize=True)
        out.append(buf.getvalue())
    return out


def encode_jpeg(img, quality: int = 78) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
