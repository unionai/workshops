"""
Bird's-eye-view renderer for Cosmos Drive Dreams clips.

Everything is drawn from the released annotations: the HD map layers (lanes, lane lines,
crosswalks, road boundaries, poles, traffic lights and signs), the per-frame 3D object
boxes, and the ego pose. No model runs — see the README for why that framing is deliberate.

Coordinates are the sensor rig frame, in metres: +x forward, +y left, +z up. The renderer
projects to screen with +x up and +y left, which is the conventional BEV orientation.
"""

import json
import math
import os

# Map layers, drawn back-to-front. Colour, stroke width, and whether it closes into a loop.
MAP_LAYERS = [
    ("3d_road_boundaries", "#334155", 2.0, False),
    ("3d_lanes", "#1e293b", 6.0, False),
    ("3d_lanelines", "#475569", 1.2, False),
    ("3d_road_markings", "#64748b", 1.0, False),
    ("3d_crosswalks", "#f59e0b", 1.6, True),
    ("3d_wait_lines", "#94a3b8", 1.4, False),
]

# Point-like map furniture.
POINT_LAYERS = [
    ("3d_poles", "#64748b", 1.6),
    ("3d_traffic_lights", "#22d3ee", 3.0),
    ("3d_traffic_signs", "#a3e635", 2.6),
]

# Object classes. Colour-blind-safe hues, distinct in lightness as well as hue so the
# classes stay separable in greyscale.
OBJECT_COLORS = {
    "Automobile": "#38bdf8",
    "Person": "#f472b6",
    "Rider": "#fbbf24",
    "Bus": "#a78bfa",
    "Heavy_truck": "#34d399",
    "Trailer": "#2dd4bf",
    "Animal": "#fb7185",
    "Protruding_object": "#94a3b8",
}
DEFAULT_OBJECT_COLOR = "#e2e8f0"
EGO_COLOR = "#ef4444"


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

def parse_objects(frame_dir: str) -> list[dict]:
    """Per-frame object annotations -> [{track, type, corners, centre, moving}, ...]."""
    import glob

    frames = []
    for path in sorted(glob.glob(os.path.join(frame_dir, "*all_object_info.json"))):
        with open(path) as f:
            raw = json.load(f)
        objs = []
        for track_id, o in raw.items():
            m = o["object_to_world"]
            length, width, _h = o["object_lwh"]
            # Rotation is the upper-left 2x2 of the 4x4; translation is the last column.
            cos_r, sin_r = m[0][0], m[1][0]
            tx, ty = m[0][3], m[1][3]
            corners = []
            for lx, ly in ((length / 2, width / 2), (length / 2, -width / 2),
                           (-length / 2, -width / 2), (-length / 2, width / 2)):
                corners.append((tx + cos_r * lx - sin_r * ly,
                                ty + sin_r * lx + cos_r * ly))
            objs.append({
                "track": track_id,
                "type": o.get("object_type", "Unknown"),
                "corners": corners,
                "centre": (tx, ty),
                "moving": bool(o.get("object_is_moving", False)),
            })
        frames.append(objs)
    return frames


def _xy(verts) -> list[tuple[float, float]]:
    return [(v[0], v[1]) for v in verts if isinstance(v, (list, tuple)) and len(v) >= 2]


def _geometry_from_label(label: dict) -> list[list[tuple[float, float]]]:
    """
    Extract 2D geometry from one label, whatever shape encoding it uses.

    The map layers do NOT share a schema — there are four encodings in this dataset, and
    handling only one silently yields an almost-empty map rather than an error:

      polylines3d.polylines[].vertices  multi-polyline  (3d_lanes)
      polyline3d.vertices               single polyline (lanelines, road_boundaries,
                                                         wait_lines, poles)
      surface.vertices                  polygon         (crosswalks, road_markings)
      cuboid3d.vertices                 3D box          (traffic_lights, traffic_signs)
    """
    shape = label.get("labelData", {}).get("shape3d")
    if not shape:
        return []

    out = []
    for poly in shape.get("polylines3d", {}).get("polylines", []):
        pts = _xy(poly.get("vertices") or [])
        if len(pts) >= 2:
            out.append(pts)

    single = shape.get("polyline3d")
    if single:
        pts = _xy(single.get("vertices") or [])
        if len(pts) >= 2:
            out.append(pts)

    surface = shape.get("surface")
    if surface:
        pts = _xy(surface.get("vertices") or [])
        if len(pts) >= 3:
            out.append(pts)

    cuboid = shape.get("cuboid3d")
    if cuboid:
        pts = _xy(cuboid.get("vertices") or [])
        if pts:
            out.append(pts)

    return out


def parse_map_layer(path: str) -> list[list[tuple[float, float]]]:
    """A `3d_*` json -> list of 2D polylines/polygons in the rig frame."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    lines = []
    for label in data.get("labels", []):
        lines.extend(_geometry_from_label(label))
    return lines


def parse_point_layer(path: str) -> list[tuple[float, float]]:
    """Poles / lights / signs -> one representative 2D position per feature."""
    pts = []
    for line in parse_map_layer(path):
        xs = [p[0] for p in line]
        ys = [p[1] for p in line]
        pts.append((sum(xs) / len(xs), sum(ys) / len(ys)))
    return pts


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------

class BevView:
    """Maps rig-frame metres to image pixels. +x forward becomes up, +y left becomes left."""

    def __init__(self, width: int, height: int, fwd: float, side: float,
                 back: float = 20.0):
        self.w, self.h = width, height
        self.fwd, self.side, self.back = fwd, side, back
        self.sx = width / (2 * side)
        self.sy = height / (fwd + back)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        return (self.w / 2 - y * self.sx, self.h - (x + self.back) * self.sy)

    def visible(self, x: float, y: float, pad: float = 12.0) -> bool:
        return (-self.back - pad) <= x <= (self.fwd + pad) and abs(y) <= (self.side + pad)


def render_frame(view: BevView, map_lines, point_feats, objects, trails=None,
                 frame_idx: int = 0, n_frames: int = 0, label: str = "") -> bytes:
    """Render one BEV frame to PNG bytes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (view.w, view.h), "#080b12")
    d = ImageDraw.Draw(img, "RGBA")

    # range rings every 25 m, so distances are readable
    for r in range(25, int(view.fwd) + 1, 25):
        top = view(r, 0)[1]
        d.line([(0, top), (view.w, top)], fill=(30, 41, 59, 255), width=1)
        d.text((6, top + 2), f"{r} m", fill=(71, 85, 105, 255))

    for layer_lines, color, w, closed in map_lines:
        for line in layer_lines:
            pts = [view(x, y) for x, y in line if view.visible(x, y, pad=40)]
            if len(pts) >= 2:
                d.line(pts + ([pts[0]] if closed else []), fill=color, width=int(w))

    for pts, color, r in point_feats:
        for x, y in pts:
            if view.visible(x, y):
                px, py = view(x, y)
                d.ellipse([px - r, py - r, px + r, py + r], fill=color)

    # motion trails behind each object, oldest faintest
    if trails:
        for track, hist in trails.items():
            pts = [view(x, y) for x, y in hist if view.visible(x, y)]
            if len(pts) >= 2:
                d.line(pts, fill=(148, 163, 184, 110), width=1)

    for o in objects:
        if not view.visible(*o["centre"]):
            continue
        color = OBJECT_COLORS.get(o["type"], DEFAULT_OBJECT_COLOR)
        poly = [view(x, y) for x, y in o["corners"]]
        # Filled when moving, outline-only when static — motion state is in the data and
        # is the single most useful thing to see at a glance in a BEV.
        d.polygon(poly, fill=(color + "55") if o["moving"] else None, outline=color)
        # heading tick from centre to the front edge
        cx, cy = view(*o["centre"])
        fx = (poly[0][0] + poly[1][0]) / 2, (poly[0][1] + poly[1][1]) / 2
        d.line([(cx, cy), fx], fill=color, width=1)

    # ego vehicle at the origin
    ex, ey = view(0, 0)
    d.polygon([(ex, ey - 7), (ex - 4, ey + 5), (ex + 4, ey + 5)], fill=EGO_COLOR)

    if n_frames:
        d.text((8, 8), f"frame {frame_idx + 1}/{n_frames}", fill=(203, 213, 225, 255))
    if label:
        d.text((8, 22), label, fill=(100, 116, 139, 255))
    d.text((8, view.h - 14), f"{len(objects)} objects in view", fill=(100, 116, 139, 255))

    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_trails(frames: list[list[dict]], upto: int, length: int = 25) -> dict:
    """Recent centre positions per track, for motion trails."""
    trails: dict[str, list] = {}
    for i in range(max(0, upto - length), upto + 1):
        if i >= len(frames):
            break
        for o in frames[i]:
            trails.setdefault(o["track"], []).append(o["centre"])
    return trails
