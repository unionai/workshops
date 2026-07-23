"""
Web-Mercator basemap tiles, fetched once and embedded in the report.

Tiles are fetched server-side during the task and baked into the report as a base64 PNG,
so the report stays a single self-contained file — it renders months later with no tile
server, no network, and no API key.

The important detail is alignment: the map image and the point projection must use the
*same* Web-Mercator window, to the pixel. Fitting the projection to the data extent and
the basemap to a bounding box independently puts every route a few hundred metres off the
streets, which looks like a data bug and isn't one.

Attribution is required by both providers and is rendered onto the map.
"""

import io
import math
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TILE = 256

STYLES = {
    # Carto basemaps, built on OpenStreetMap data.
    "dark": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "light": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}
ATTRIBUTION = "© OpenStreetMap contributors © CARTO"

# Tile servers require a genuine identifying User-Agent; a default urllib one gets blocked.
HEADERS = {"User-Agent": "union-flyte-tutorial/1.0 (+https://www.union.ai) logistics-demo"}


def lonlat_to_world(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Lon/lat -> global pixel coordinates at `zoom` (origin top-left)."""
    n = TILE * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * n
    la = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.log(math.tan(la) + 1.0 / math.cos(la)) / math.pi) / 2.0 * n
    return x, y


def fetch_basemap(bbox, zoom: int = 13, style: str = "dark", timeout: int = 30):
    """
    Fetch and stitch tiles covering `bbox` = (west, south, east, north).

    Returns (PIL.Image cropped exactly to bbox, (px_w, px_h)). Returns (None, None) if the
    tiles cannot be fetched — the caller should fall back to a plain background rather
    than fail the task, since a basemap is decoration and the routes are the content.
    """
    from PIL import Image

    west, south, east, north = bbox
    x0, y0 = lonlat_to_world(west, north, zoom)   # top-left
    x1, y1 = lonlat_to_world(east, south, zoom)   # bottom-right

    tx0, ty0 = int(x0 // TILE), int(y0 // TILE)
    tx1, ty1 = int(x1 // TILE), int(y1 // TILE)
    n_tiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    if n_tiles > 64:
        raise ValueError(f"{n_tiles} tiles at zoom {zoom} — lower the zoom")

    template = STYLES.get(style, STYLES["dark"])
    coords = [(tx, ty) for ty in range(ty0, ty1 + 1) for tx in range(tx0, tx1 + 1)]

    def get(tile):
        tx, ty = tile
        sub = "abc"[(tx + ty) % 3]
        url = template.replace("{s}", sub).format(z=zoom, x=tx, y=ty)
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return tile, Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception:
            return tile, None

    with ThreadPoolExecutor(max_workers=8) as ex:
        fetched = dict(ex.map(get, coords))

    if not any(v is not None for v in fetched.values()):
        return None, None

    canvas = Image.new("RGB", ((tx1 - tx0 + 1) * TILE, (ty1 - ty0 + 1) * TILE), (12, 16, 24))
    for (tx, ty), img in fetched.items():
        if img is not None:
            canvas.paste(img, ((tx - tx0) * TILE, (ty - ty0) * TILE))

    # Crop to the exact bbox so the image and the projection share one window.
    left = x0 - tx0 * TILE
    top = y0 - ty0 * TILE
    right = x1 - tx0 * TILE
    bottom = y1 - ty0 * TILE
    cropped = canvas.crop((int(left), int(top), max(int(right), int(left) + 1),
                           max(int(bottom), int(top) + 1)))
    return cropped, cropped.size


def image_to_uri(img, quality: int = 82) -> str:
    """JPEG data URI — a stitched basemap is photographic and ~6x smaller than PNG."""
    import base64

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_bbox_to_aspect(bbox, width: int, height: int):
    """
    Expand a bbox so its Web-Mercator aspect ratio matches width/height.

    Without this the basemap is drawn with preserveAspectRatio="none" into a frame of a
    different shape, which stretches the geography — Manhattan comes out visibly too wide.
    Expanding (never cropping) keeps every point inside the frame.
    """
    west, south, east, north = bbox
    ys, yn = _merc_y(south), _merc_y(north)
    dx = (east - west) or 1e-6
    dy = (yn - ys) or 1e-6
    target = width / height
    if dx / dy < target:                      # too tall -> widen
        want = dy * target
        cx = (west + east) / 2
        west, east = cx - want / 2, cx + want / 2
    else:                                     # too wide -> heighten
        want = dx / target
        cy = (ys + yn) / 2
        ys, yn = cy - want / 2, cy + want / 2
        south, north = _inv_merc_y(ys), _inv_merc_y(yn)
    return (west, south, east, north)


def _merc_y(lat: float) -> float:
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


def _inv_merc_y(y: float) -> float:
    return math.degrees(2 * math.atan(math.exp(math.radians(y))) - math.pi / 2)


def pad_bbox(lats, lngs, margin: float = 0.12):
    """Bounding box around points with a margin, so nothing sits on the frame edge."""
    south, north = min(lats), max(lats)
    west, east = min(lngs), max(lngs)
    dy = (north - south) or 0.01
    dx = (east - west) or 0.01
    return (west - dx * margin, south - dy * margin,
            east + dx * margin, north + dy * margin)
