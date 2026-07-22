"""
Report visuals for the demand-forecast + routing pipeline.

Everything renders to inline SVG so a Flyte Report is self-contained — no tile server, no
CDN, nothing to 404 later. Coordinates are real (Manhattan), projected with a Web-Mercator
y so the map isn't visibly stretched.
"""

import math

# Sequential ramp for demand (low -> high). Perceptually ordered, readable on white.
DEMAND_RAMP = ["#eff6ff", "#bfdbfe", "#7dd3fc", "#38bdf8", "#0284c7", "#1e40af"]

# Categorical, colour-blind-safe vehicle colours. Distinct in hue AND lightness so routes
# stay separable in greyscale and for viewers with deuteranopia.
VEHICLE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#17becf", "#bcbd22"]

DEPOT_COLOR = "#111827"


REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #0284c7; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #1e40af; margin-top: 20px; }
  .report .card { background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #1e3a5f; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report .stat .delta { font-size: 0.8em; font-weight: 600; margin-top: 2px; }
  .report .up { color: #15803d; } .report .down { color: #b91c1c; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #1e3a5f; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dbeafe; }
  .report tr:nth-child(even) { background: #eff6ff; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-warning { background: #fef3c7; color: #92400e; }
  .report .badge-info { background: #dbeafe; color: #1e40af; }
  .report .chart-container { background: #fff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #eff6ff; border-left: 4px solid #0284c7; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 16px; }
  .report .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0; font-size: 0.85em; }
  .report .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .report .swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,.15); }
</style>
"""


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# Geographic projection
# ------------------------------------------------------------------

def _mercator_y(lat: float) -> float:
    """Web-Mercator y. Without this, a lat/lng plot of Manhattan looks noticeably squashed."""
    lat = max(min(lat, 85.0), -85.0)
    return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


class Projection:
    """Project lon/lat into SVG pixel space, preserving aspect ratio."""

    def __init__(self, lats, lngs, width: int, height: int, pad: int = 28):
        ys = [_mercator_y(v) for v in lats]
        self.x0, self.x1 = min(lngs), max(lngs)
        self.y0, self.y1 = min(ys), max(ys)
        dx = (self.x1 - self.x0) or 1e-6
        dy = (self.y1 - self.y0) or 1e-6
        # One scale for both axes keeps the geography honest.
        self.s = min((width - 2 * pad) / dx, (height - 2 * pad) / dy)
        self.ox = pad + ((width - 2 * pad) - dx * self.s) / 2
        self.oy = pad + ((height - 2 * pad) - dy * self.s) / 2
        self.height = height
        self.pad = pad

    def __call__(self, lat: float, lng: float) -> tuple[float, float]:
        x = self.ox + (lng - self.x0) * self.s
        y = self.oy + (self.y1 - _mercator_y(lat)) * self.s
        return x, y


def _ramp_color(t: float) -> str:
    """t in [0,1] -> colour from DEMAND_RAMP with linear interpolation between stops."""
    t = max(0.0, min(1.0, t))
    n = len(DEMAND_RAMP) - 1
    i = int(t * n)
    if i >= n:
        return DEMAND_RAMP[-1]
    f = t * n - i

    def hx(c):
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

    r1, g1, b1 = hx(DEMAND_RAMP[i])
    r2, g2, b2 = hx(DEMAND_RAMP[i + 1])
    return (f"rgb({int(r1 + (r2 - r1) * f)},"
            f"{int(g1 + (g2 - g1) * f)},{int(b1 + (b2 - b1) * f)})")


# ------------------------------------------------------------------
# Maps
# ------------------------------------------------------------------

def demand_map(zones: list[dict], title: str = "", width: int = 720, height: int = 560,
               scale_label: str = "Forecast demand") -> str:
    """
    Bubble map of per-zone demand. `zones` need `lat`, `lng`, `demand`.

    Area (not radius) is proportional to demand — scaling radius linearly would
    over-state large values by the square.
    """
    if not zones:
        return ""
    proj = Projection([z["lat"] for z in zones], [z["lng"] for z in zones], width, height)
    dmax = max(z["demand"] for z in zones) or 1.0
    dmin = min(z["demand"] for z in zones)
    rng = (dmax - dmin) or 1.0

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#1e3a5f">{title}</text>')

    for z in sorted(zones, key=lambda z: z["demand"]):
        x, y = proj(z["lat"], z["lng"])
        t = (z["demand"] - dmin) / rng
        r = 3 + math.sqrt(max(z["demand"], 0) / dmax) * 13
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{_ramp_color(t)}" '
                   f'fill-opacity="0.82" stroke="#1e3a5f" stroke-width="0.5"/>')

    # colour scale
    bw, bh = 150, 9
    bx, by = width - bw - 20, height - 34
    for i in range(bw):
        svg.append(f'<rect x="{bx+i}" y="{by}" width="1" height="{bh}" '
                   f'fill="{_ramp_color(i/bw)}"/>')
    svg.append(f'<text x="{bx}" y="{by-5}" font-size="10" fill="#475569">{scale_label}</text>')
    svg.append(f'<text x="{bx}" y="{by+bh+11}" font-size="9" fill="#64748b">{dmin:.0f}</text>')
    svg.append(f'<text x="{bx+bw}" y="{by+bh+11}" text-anchor="end" font-size="9" '
               f'fill="#64748b">{dmax:.0f}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def route_map(zones: list[dict], routes: list[dict], depot: dict, title: str = "",
              width: int = 720, height: int = 560, animate: bool = True) -> str:
    """
    Draw solved vehicle routes over the zone map.

    `routes` entries: {"vehicle": int, "path": [zone_index, ...], "distance_m": float}.
    Indices refer to `zones`. The depot is drawn separately as a square.

    When `animate` is on each polyline draws itself in with a stroke-dash animation —
    pure SVG/SMIL, no JS, and it makes a screen recording of the mosaic actually move.
    """
    if not zones:
        return ""
    lats = [z["lat"] for z in zones] + [depot["lat"]]
    lngs = [z["lng"] for z in zones] + [depot["lng"]]
    proj = Projection(lats, lngs, width, height)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#1e3a5f">{title}</text>')

    # unvisited zones first, faintly
    for z in zones:
        x, y = proj(z["lat"], z["lng"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#cbd5e1"/>')

    dx, dy = proj(depot["lat"], depot["lng"])

    for r in routes:
        color = VEHICLE_COLORS[r["vehicle"] % len(VEHICLE_COLORS)]
        pts = [(dx, dy)]
        for zi in r["path"]:
            z = zones[zi]
            pts.append(proj(z["lat"], z["lng"]))
        pts.append((dx, dy))  # return to depot
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

        # rough path length in px, for the dash animation
        plen = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) or 1.0
        anim = ""
        if animate:
            anim = (f'<animate attributeName="stroke-dashoffset" from="{plen:.0f}" to="0" '
                    f'dur="2.4s" begin="{0.25*r["vehicle"]:.2f}s" fill="freeze"/>')
        dash = f'stroke-dasharray="{plen:.0f}" stroke-dashoffset="{plen:.0f}"' if animate else ""
        svg.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
                   f'stroke-opacity="0.9" stroke-linejoin="round" stroke-linecap="round" '
                   f'{dash}>{anim}</polyline>')

        for zi in r["path"]:
            z = zones[zi]
            x, y = proj(z["lat"], z["lng"])
            rr = 3.4 + math.sqrt(max(z.get("demand", 0), 0)) * 0.5
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{min(rr,9):.1f}" fill="{color}" '
                       f'stroke="#fff" stroke-width="1"/>')

    svg.append(f'<rect x="{dx-6:.1f}" y="{dy-6:.1f}" width="12" height="12" rx="2" '
               f'fill="{DEPOT_COLOR}" stroke="#fff" stroke-width="1.5"/>')
    svg.append(f'<text x="{dx:.1f}" y="{dy-11:.1f}" text-anchor="middle" font-size="10" '
               f'font-weight="600" fill="{DEPOT_COLOR}">Depot</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def vehicle_legend(routes: list[dict]) -> str:
    items = []
    for r in routes:
        c = VEHICLE_COLORS[r["vehicle"] % len(VEHICLE_COLORS)]
        items.append(f'<span><i class="swatch" style="background:{c}"></i> '
                     f'Vehicle {r["vehicle"]} · {len(r["path"])} stops · '
                     f'{r["distance_m"]/1000:.1f} km</span>')
    items.append(f'<span><i class="swatch" style="background:{DEPOT_COLOR}"></i> Depot</span>')
    return f'<div class="legend">{"".join(items)}</div>'


# ------------------------------------------------------------------
# Forecast charts
# ------------------------------------------------------------------

def forecast_chart(history, actual, median, lo, hi, title: str = "",
                   width: int = 760, height: int = 280, hist_tail: int = 96) -> str:
    """
    Context + prediction interval + actual, the standard forecast view.

    Showing the 10-90% band matters: a point forecast hides whether the model is confident,
    and capacity planning is a decision about the upper tail, not the mean.
    """
    hist = list(history)[-hist_tail:]
    H = len(median)
    n = len(hist) + H
    ml, mr, mt, mb = 52, 16, 34, 34
    cw, ch = width - ml - mr, height - mt - mb

    # `actual or []` would raise on a numpy array ("truth value is ambiguous"), so test
    # for None explicitly — callers pass numpy arrays as often as lists.
    actual = [] if actual is None else list(actual)
    allv = list(hist) + list(lo) + list(hi) + actual
    if not allv:
        return ""
    v_min, v_max = min(allv), max(allv)
    if v_max <= v_min:
        v_max = v_min + 1
    pad = (v_max - v_min) * 0.1
    v_min, v_max = v_min - pad, v_max + pad

    def sx(i):
        return ml + (i / max(n - 1, 1)) * cw

    def sy(v):
        return mt + ch - ((v - v_min) / (v_max - v_min)) * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="20" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#1e3a5f">{title}</text>')

    for k in range(4):
        y = mt + (k / 3) * ch
        val = v_max - (k / 3) * (v_max - v_min)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ml-7}" y="{y+4:.1f}" text-anchor="end" font-size="9" '
                   f'fill="#64748b">{val:.0f}</text>')

    # forecast region shading + boundary
    fx = sx(len(hist) - 1)
    svg.append(f'<rect x="{fx:.1f}" y="{mt}" width="{ml+cw-fx:.1f}" height="{ch}" '
               f'fill="#eff6ff" opacity="0.7"/>')
    svg.append(f'<line x1="{fx:.1f}" y1="{mt}" x2="{fx:.1f}" y2="{mt+ch}" stroke="#94a3b8" '
               f'stroke-dasharray="3,3"/>')

    # 10-90% band
    band = ([f"{sx(len(hist)+i):.1f},{sy(v):.1f}" for i, v in enumerate(hi)]
            + [f"{sx(len(hist)+i):.1f},{sy(v):.1f}" for i, v in reversed(list(enumerate(lo)))])
    svg.append(f'<polygon points="{" ".join(band)}" fill="#0284c7" fill-opacity="0.18"/>')

    svg.append('<polyline points="' + " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(hist))
               + '" fill="none" stroke="#475569" stroke-width="1.6"/>')
    svg.append('<polyline points="' + " ".join(f"{sx(len(hist)+i):.1f},{sy(v):.1f}" for i, v in enumerate(median))
               + '" fill="none" stroke="#0284c7" stroke-width="2.2"/>')
    if len(actual):
        svg.append('<polyline points="' + " ".join(f"{sx(len(hist)+i):.1f},{sy(v):.1f}" for i, v in enumerate(actual))
                   + '" fill="none" stroke="#dc2626" stroke-width="1.8" stroke-dasharray="4,2"/>')

    leg = [("#475569", "history"), ("#0284c7", "forecast (median)"), ("#dc2626", "actual")]
    for i, (c, lab) in enumerate(leg):
        lx = ml + 6 + i * 130
        svg.append(f'<line x1="{lx}" y1="{mt+ch+22}" x2="{lx+16}" y2="{mt+ch+22}" '
                   f'stroke="{c}" stroke-width="2.2"/>')
        svg.append(f'<text x="{lx+21}" y="{mt+ch+26}" font-size="10" fill="#475569">{lab}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def make_bar_chart(labels, values, colors=None, title: str = "", width: int = 700,
                   height: int = 290, value_format: str = ".2f",
                   y_label: str = "", lower_is_better: bool = False) -> str:
    if not labels:
        return ""
    colors = colors or ["#0284c7"] * len(labels)
    ml, mr, mt, mb = 58, 20, 42, 54
    cw, ch = width - ml - mr, height - mt - mb
    top = max(max(values), 1e-9) * 1.18

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#1e3a5f">{title}</text>')
    for k in range(5):
        y = mt + (k / 4) * ch
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="#64748b">{top-(k/4)*top:.1f}</text>')

    slot = cw / len(labels)
    bw = min(slot * 0.58, 80)
    best = min(values) if lower_is_better else max(values)
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = (val / top) * ch if top else 0
        x = ml + i * slot + (slot - bw) / 2
        y = mt + ch - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                   f'fill="{colors[i%len(colors)]}" rx="3"/>')
        star = " ★" if val == best else ""
        svg.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11" '
                   f'font-weight="600" fill="#1a1a2e">{val:{value_format}}{star}</text>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{mt+ch+17}" text-anchor="middle" font-size="10" '
                   f'fill="#374151">{lab}</text>')
    if y_label:
        svg.append(f'<text x="14" y="{mt+ch/2:.1f}" text-anchor="middle" font-size="10" '
                   f'fill="#64748b" transform="rotate(-90,14,{mt+ch/2:.1f})">{y_label}</text>')
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#cbd5e1"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def heatmap_hour_dow(matrix, title: str = "", width: int = 760, height: int = 260) -> str:
    """7x24 demand heatmap — day-of-week by hour. Makes the commute peaks obvious."""
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ml, mr, mt, mb = 44, 16, 34, 26
    cw, ch = width - ml - mr, height - mt - mb
    flat = [v for row in matrix for v in row]
    if not flat:
        return ""
    lo, hi = min(flat), max(flat)
    rng = (hi - lo) or 1
    cellw, cellh = cw / 24, ch / 7

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="21" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#1e3a5f">{title}</text>')
    for r in range(7):
        svg.append(f'<text x="{ml-7}" y="{mt+r*cellh+cellh/2+3:.1f}" text-anchor="end" '
                   f'font-size="9" fill="#475569">{days[r]}</text>')
        for c in range(24):
            v = matrix[r][c]
            svg.append(f'<rect x="{ml+c*cellw:.1f}" y="{mt+r*cellh:.1f}" width="{cellw+0.5:.1f}" '
                       f'height="{cellh+0.5:.1f}" fill="{_ramp_color((v-lo)/rng)}"/>')
    for c in range(0, 24, 3):
        svg.append(f'<text x="{ml+c*cellw+cellw/2:.1f}" y="{mt+ch+14}" text-anchor="middle" '
                   f'font-size="9" fill="#64748b">{c:02d}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def progress_html(steps: list[str], current: int, note: str = "") -> str:
    dots = ""
    for i, s in enumerate(steps):
        if i + 1 < current:
            icon = '<span style="color:#0284c7;">&#10003;</span>'
        elif i + 1 == current:
            icon = '<span style="color:#0284c7;">&#9679;</span>'
        else:
            icon = '<span style="color:#cbd5e1;">&#9675;</span>'
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))
