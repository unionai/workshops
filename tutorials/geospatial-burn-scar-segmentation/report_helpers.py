"""
Report visuals for the burn-scar pipeline.

Everything here renders to inline SVG or base64 PNG data URIs so a Flyte Report is fully
self-contained — no external assets, no CDN, nothing to 404 when someone opens the report
six months later.

Band order throughout is HLS S30 as shipped by the dataset:
    0 B02 blue | 1 B03 green | 2 B04 red | 3 B8A NIR | 4 B11 SWIR1 | 5 B12 SWIR2
"""

import base64
import io

# Band indices
BLUE, GREEN, RED, NIR, SWIR1, SWIR2 = range(6)

BURN_COLOR = (255, 61, 61)      # predicted / ground-truth burn overlay
TRUTH_COLOR = (56, 189, 248)    # ground truth when shown against a prediction
NODATA_COLOR = (120, 120, 120)


REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #1e3a5f; border-bottom: 2px solid #ea580c; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #9a3412; margin-top: 20px; }
  .report .card { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #9a3412; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #7c2d12; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #fed7aa; }
  .report tr:nth-child(even) { background: #fff7ed; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d1fae5; color: #065f46; }
  .report .badge-warning { background: #fef3c7; color: #92400e; }
  .report .badge-danger { background: #fee2e2; color: #991b1b; }
  .report .chart-container { background: #fff; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #fff7ed; border-left: 4px solid #ea580c; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .scene-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin: 12px 0; }
  .report .scene { background: #fff; border: 1px solid #fed7aa; border-radius: 8px; padding: 10px; }
  .report .scene img { width: 100%; border-radius: 4px; display: block; image-rendering: pixelated; }
  .report .scene .cap { font-size: 0.8em; color: #6c757d; margin-top: 6px; text-align: center; }
  .report .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0; font-size: 0.85em; }
  .report .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .report .swatch { width: 14px; height: 14px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,.15); }
</style>
"""


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


# ------------------------------------------------------------------
# Raster rendering
# ------------------------------------------------------------------

def _stretch(band, lo_pct: float = 2.0, hi_pct: float = 98.0, rng=None):
    """
    Percentile contrast stretch to 0-255. Satellite bands occupy a narrow reflectance
    range, so raw min/max scaling renders almost black.

    Pass an explicit `rng` of (lo, hi) to use a fixed scaling instead of a per-array
    percentile. That matters when rendering tiles that will be stitched together: stretching
    each tile independently gives every tile its own mapping, and the seams show as visible
    blocking across the mosaic.
    """
    import numpy as np

    b = band.astype("float32")
    if rng is not None:
        lo, hi = float(rng[0]), float(rng[1])
        if hi <= lo:
            return np.zeros(b.shape, dtype="uint8")
        return (np.clip((b - lo) / (hi - lo), 0, 1) * 255).astype("uint8")

    valid = b[np.isfinite(b)]
    if valid.size == 0:
        return np.zeros(b.shape, dtype="uint8")
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(b.shape, dtype="uint8")
    out = (b - lo) / (hi - lo)
    return (np.clip(out, 0, 1) * 255).astype("uint8")


def to_png_uri(rgb) -> str:
    """uint8 (H, W, 3) array -> inline base64 PNG data URI."""
    from PIL import Image

    img = Image.fromarray(rgb)
    return image_to_uri(img)


def image_to_uri(img) -> str:
    """PIL Image -> inline base64 PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def uri_to_image(uri: str):
    """Inverse of `image_to_uri` — used to restitch tile thumbnails into a mosaic."""
    from PIL import Image

    return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))


def composite(scene, bands=(SWIR2, NIR, RED), ranges=None):
    """
    Render a 3-band composite as a uint8 RGB array.

    The default (SWIR2, NIR, Red) is the standard burn-severity composite: healthy
    vegetation is bright green, water is near black, and recently burned ground reads as
    vivid magenta-crimson — charred surfaces are highly reflective in SWIR (the red
    channel) and dark in NIR (the green channel). This is the single reason the tutorial
    uses the 6-band HLS product rather than an RGB-only derivative: in true colour the
    same scar is an unremarkable brown smudge.

    Pass `ranges` (a list of (lo, hi) per source band index) to apply one fixed scaling
    across many tiles. Without it each tile is stretched on its own statistics, and the
    stitched mosaic shows the tile grid as visible seams.
    """
    import numpy as np

    return np.stack(
        [_stretch(scene[b], rng=(ranges[b] if ranges else None)) for b in bands],
        axis=-1,
    )


def true_color(scene):
    return composite(scene, bands=(RED, GREEN, BLUE))


def overlay(rgb, mask, color=BURN_COLOR, alpha: float = 0.45):
    """Blend a binary mask over an RGB array."""
    import numpy as np

    out = rgb.astype("float32").copy()
    m = mask.astype(bool)
    if m.any():
        for c in range(3):
            out[:, :, c] = np.where(m, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return np.clip(out, 0, 255).astype("uint8")


def outline(rgb, mask, color=TRUTH_COLOR, width: int = 2):
    """Draw the boundary of a mask, so ground truth can sit over a prediction fill
    without hiding it."""
    import numpy as np

    m = mask.astype(bool)
    if not m.any():
        return rgb
    edge = np.zeros_like(m)
    for shift in range(1, width + 1):
        edge |= m ^ np.roll(m, shift, axis=0)
        edge |= m ^ np.roll(m, shift, axis=1)
    edge &= m
    out = rgb.astype("float32").copy()
    for c in range(3):
        out[:, :, c] = np.where(edge, color[c], out[:, :, c])
    return np.clip(out, 0, 255).astype("uint8")


def scene_uri(scene, mask=None, bands=(SWIR2, NIR, RED), color=BURN_COLOR, alpha=0.45) -> str:
    rgb = composite(scene, bands=bands)
    if mask is not None:
        rgb = overlay(rgb, mask, color=color, alpha=alpha)
    return to_png_uri(rgb)


# How each surface actually renders in the SWIR2/NIR/Red composite. Swatches are sampled
# from real Dixie Fire pixels rather than guessed, so the legend matches what is on screen.
COMPOSITE_CLASSES = [
    ("#c73e7f", "Burn scar", "high SWIR2, low NIR"),
    ("#2f9e44", "Healthy vegetation", "high NIR"),
    ("#e3cec6", "Bare rock / soil", "high across SWIR"),
    ("#243b53", "Terrain shadow", "NIR down ~60%, visible unchanged"),
    ("#0b1a2b", "Water", "NIR collapses, NDWI > 0.8"),
]


def burn_legend_html(overlay: bool = True, composite: bool = True) -> str:
    """
    Colour key for the reports.

    Two separate things need explaining and they are easy to conflate: what the *imagery*
    colours mean, and what the *model overlay* colours mean. Terrain shadow in particular
    reads as a suspicious dark patch to anyone scanning a burn map, so it gets its own
    entry rather than being lumped in with water.
    """
    out = []
    if overlay:
        out.append(
            '<div class="legend" style="margin-bottom:4px;">'
            '<b style="color:#7c2d12;font-size:.9em;">Model overlay:</b>'
            f'<span><i class="swatch" style="background:rgb{BURN_COLOR}"></i> Predicted burn</span>'
            f'<span><i class="swatch" style="background:rgb{TRUTH_COLOR}"></i> Ground truth outline</span>'
            '</div>'
        )
    if composite:
        items = "".join(
            f'<span><i class="swatch" style="background:{c}"></i> {name} '
            f'<span style="color:#9ca3af;">({why})</span></span>'
            for c, name, why in COMPOSITE_CLASSES
        )
        out.append(
            '<div class="legend">'
            '<b style="color:#7c2d12;font-size:.9em;">SWIR2/NIR/Red imagery:</b>'
            f'{items}</div>'
        )
    return "".join(out)


# ------------------------------------------------------------------
# Before / after wipe — the headline visual
# ------------------------------------------------------------------

def wipe_html(before_uri: str, after_uri: str, label_before: str, label_after: str,
              slug: str, height: int = 460) -> str:
    """
    A draggable before/after wipe between two images.

    Self-contained: one range input driving a clip-path, no libraries. Reads well as a
    still and even better in a screen recording, which is the point — dragging the handle
    across a burn scar is the money shot of this pipeline.
    """
    img_css = ("position:absolute;inset:0;width:100%;height:100%;object-fit:contain;"
               "image-rendering:pixelated;")
    stage_css = "position:absolute;inset:0;transform-origin:0 0;will-change:transform;"
    return f"""
    <div class="chart-container">
      <div id="vp-{slug}" style="position:relative;width:100%;max-width:640px;margin:0 auto;
           height:{height}px;overflow:hidden;border-radius:8px;user-select:none;
           cursor:grab;background:#0b0f17;">
        <div id="sa-{slug}" style="{stage_css}"><img src="{before_uri}" style="{img_css}"></div>
        <!-- Clipping happens on this wrapper, which is NOT transformed, so the seam stays
             in viewport space and keeps matching the divider at any zoom or pan. -->
        <div id="clip-{slug}" style="position:absolute;inset:0;clip-path:inset(0 0 0 50%);">
          <div id="sb-{slug}" style="{stage_css}"><img src="{after_uri}" style="{img_css}"></div>
        </div>
        <div id="bar-{slug}" style="position:absolute;top:0;bottom:0;left:50%;width:2px;
             background:#fff;box-shadow:0 0 6px rgba(0,0,0,.6);pointer-events:none;"></div>
        <div style="position:absolute;top:8px;left:10px;background:rgba(0,0,0,.65);color:#fff;
             padding:3px 9px;border-radius:4px;font-size:.78em;pointer-events:none;">{label_before}</div>
        <div style="position:absolute;top:8px;right:10px;background:rgba(0,0,0,.65);color:#fff;
             padding:3px 9px;border-radius:4px;font-size:.78em;pointer-events:none;">{label_after}</div>
        <div id="z-{slug}" style="position:absolute;bottom:8px;right:10px;background:rgba(0,0,0,.6);
             color:#cbd5e1;padding:2px 8px;border-radius:4px;font-size:.72em;pointer-events:none;">1.0&times;</div>
      </div>
      <input id="slider-{slug}" type="range" min="0" max="100" value="50"
             style="width:100%;max-width:640px;display:block;margin:12px auto 0;">
      <div style="text-align:center;color:#6c757d;font-size:.78em;margin-top:6px;">
        drag the slider to wipe &nbsp;·&nbsp; scroll to zoom &nbsp;·&nbsp; drag the image to pan
        &nbsp;·&nbsp; double-click to reset
      </div>
      <script>
      (function() {{
        var vp=document.getElementById('vp-{slug}'), s=document.getElementById('slider-{slug}');
        var sa=document.getElementById('sa-{slug}'), sb=document.getElementById('sb-{slug}');
        var clip=document.getElementById('clip-{slug}'), bar=document.getElementById('bar-{slug}');
        var zl=document.getElementById('z-{slug}');
        var z=1, tx=0, ty=0, drag=false, lx=0, ly=0;
        function apply() {{
          var t='translate('+tx+'px,'+ty+'px) scale('+z+')';
          sa.style.transform=t; sb.style.transform=t;
          zl.textContent=z.toFixed(1)+'\\u00d7';
        }}
        function wipe() {{
          clip.style.clipPath='inset(0 0 0 '+s.value+'%)';
          bar.style.left=s.value+'%';
        }}
        s.addEventListener('input', wipe);
        vp.addEventListener('wheel', function(e) {{
          e.preventDefault();
          var r=vp.getBoundingClientRect(), mx=e.clientX-r.left, my=e.clientY-r.top;
          var nz=Math.min(12, Math.max(1, z*(e.deltaY<0?1.15:1/1.15)));
          // keep the point under the cursor fixed while scaling
          tx=mx-(mx-tx)*(nz/z); ty=my-(my-ty)*(nz/z); z=nz;
          if(z===1) {{ tx=0; ty=0; }}
          apply();
        }}, {{passive:false}});
        vp.addEventListener('mousedown', function(e) {{
          drag=true; lx=e.clientX; ly=e.clientY; vp.style.cursor='grabbing';
        }});
        window.addEventListener('mouseup', function() {{ drag=false; vp.style.cursor='grab'; }});
        window.addEventListener('mousemove', function(e) {{
          if(!drag) return;
          tx+=e.clientX-lx; ty+=e.clientY-ly; lx=e.clientX; ly=e.clientY; apply();
        }});
        vp.addEventListener('dblclick', function() {{ z=1; tx=0; ty=0; apply(); }});
        wipe(); apply();
      }})();
      </script>
    </div>
    """


# ------------------------------------------------------------------
# Mosaic / tile fan-out
# ------------------------------------------------------------------

def mosaic_grid_html(tiles: list[dict], cols: int, cell: int | None = None,
                     max_width: int = 720) -> str:
    """
    The tile fan-out visual: one cell per tile, coloured by state.

    `tiles` entries need `row`, `col`, `state` ('pending' | 'done'), and for finished
    tiles a `uri` and `burn_frac`. Re-rendering this as tiles land is what makes a mosaic
    run watchable rather than a progress bar.
    """
    done = [t for t in tiles if t.get("state") == "done"]
    rows = (max((t["row"] for t in tiles), default=0) + 1) if tiles else 0

    # Size cells to the grid so the whole mosaic stays on screen. A 12x12 run at a fixed
    # 96px would be 1152px wide and taller than the viewport, which matters because the
    # report re-renders on every completed tile: anything below the fold means the reader
    # gets bounced back to the top each refresh instead of watching the mosaic fill.
    if cell is None:
        cell = max(28, min(96, max_width // max(cols, 1)))

    cells = []
    grid = {(t["row"], t["col"]): t for t in tiles}
    for r in range(rows):
        for c in range(cols):
            t = grid.get((r, c))
            if t is None:
                cells.append(f'<div style="width:{cell}px;height:{cell}px;"></div>')
            elif t.get("state") == "done" and t.get("uri"):
                frac = t.get("burn_frac", 0.0)
                ring = "#ef4444" if frac > 0.02 else "rgba(255,255,255,.12)"
                cells.append(
                    f'<div style="width:{cell}px;height:{cell}px;position:relative;">'
                    f'<img src="{t["uri"]}" style="width:100%;height:100%;display:block;'
                    f'image-rendering:pixelated;outline:2px solid {ring};outline-offset:-2px;">'
                    f'</div>'
                )
            else:
                cells.append(
                    f'<div style="width:{cell}px;height:{cell}px;background:#1c1917;'
                    f'outline:1px solid #292524;outline-offset:-1px;"></div>'
                )

    return f"""
    <div class="chart-container" style="background:#0c0a09;border-color:#292524;">
      <div style="display:grid;grid-template-columns:repeat({cols}, {cell}px);gap:2px;
           justify-content:center;">
        {''.join(cells)}
      </div>
      <div style="text-align:center;color:#a8a29e;font-size:.82em;margin-top:10px;">
        {len(done)} / {len(tiles)} tiles segmented
        &nbsp;·&nbsp; <span style="color:#ef4444;">red outline</span> = burn detected
      </div>
    </div>
    """


# ------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------

def make_line_chart(series: dict, title: str = "", width: int = 760, height: int = 300,
                    x_label: str = "Epoch", y_label: str = "") -> str:
    """Multi-series SVG line chart. `series` maps name -> (color, [values])."""
    if not series:
        return ""

    ml, mr, mt, mb = 60, 130, 40, 45
    cw, ch = width - ml - mr, height - mt - mb

    all_vals = [v for _, vals in series.values() for v in vals]
    if not all_vals:
        return ""
    v_min, v_max = min(all_vals), max(all_vals)
    if v_max <= v_min:
        v_max = v_min + 1.0
    pad = (v_max - v_min) * 0.08
    v_min, v_max = v_min - pad, v_max + pad
    n = max(len(vals) for _, vals in series.values())

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
        svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#7c2d12">{title}</text>')

    for k in range(5):
        y = mt + (k / 4) * ch
        val = v_max - (k / 4) * (v_max - v_min)
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f3f4f6"/>')
        svg.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="#6b7280">{val:.3g}</text>')

    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#d1d5db"/>')
    for i in range(0, n, max(1, n // 8)):
        svg.append(f'<text x="{sx(i):.1f}" y="{mt+ch+16}" text-anchor="middle" font-size="10" '
                   f'fill="#6b7280">{i+1}</text>')
    svg.append(f'<text x="{ml+cw/2:.1f}" y="{height-6}" text-anchor="middle" font-size="11" '
               f'fill="#6b7280">{x_label}</text>')
    if y_label:
        svg.append(f'<text x="14" y="{mt+ch/2:.1f}" text-anchor="middle" font-size="11" '
                   f'fill="#6b7280" transform="rotate(-90, 14, {mt+ch/2:.1f})">{y_label}</text>')

    for idx, (name, (color, vals)) in enumerate(series.items()):
        if not vals:
            continue
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
                   f'stroke-linejoin="round"/>')
        svg.append(f'<circle cx="{sx(len(vals)-1):.1f}" cy="{sy(vals[-1]):.1f}" r="3.5" fill="{color}"/>')
        ly = mt + 8 + idx * 20
        svg.append(f'<rect x="{ml+cw+14}" y="{ly-8}" width="11" height="11" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{ml+cw+31}" y="{ly+1}" font-size="11" fill="#374151">'
                   f'{name} ({vals[-1]:.3f})</text>')

    svg.append("</svg>")
    return "\n".join(svg)


def make_bar_chart(labels: list[str], values: list[float], colors: list[str] | None = None,
                   title: str = "", width: int = 700, height: int = 300,
                   value_format: str = ".3f", y_max: float | None = None) -> str:
    if not labels:
        return ""
    colors = colors or ["#ea580c"] * len(labels)
    ml, mr, mt, mb = 55, 20, 44, 56
    cw, ch = width - ml - mr, height - mt - mb
    top = y_max if y_max is not None else max(max(values), 1e-9) * 1.15

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#7c2d12">{title}</text>')

    for k in range(5):
        y = mt + (k / 4) * ch
        val = top - (k / 4) * top
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f3f4f6"/>')
        svg.append(f'<text x="{ml-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="#6b7280">{val:.2f}</text>')

    slot = cw / len(labels)
    bw = min(slot * 0.6, 76)
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = (val / top) * ch if top else 0
        x = ml + i * slot + (slot - bw) / 2
        y = mt + ch - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                   f'fill="{colors[i % len(colors)]}" rx="3"/>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11" '
                   f'font-weight="600" fill="#1a1a2e">{val:{value_format}}</text>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{mt+ch+17}" text-anchor="middle" font-size="10" '
                   f'fill="#374151">{lab}</text>')

    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#d1d5db"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def make_histogram(counts: list[int], edges: list[float], title: str = "",
                   width: int = 700, height: int = 240, color: str = "#ea580c") -> str:
    """Histogram for band reflectance distributions."""
    if not counts:
        return ""
    ml, mr, mt, mb = 50, 16, 38, 40
    cw, ch = width - ml - mr, height - mt - mb
    top = max(counts) or 1

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#7c2d12">{title}</text>')

    bw = cw / len(counts)
    for i, c in enumerate(counts):
        h = (c / top) * ch
        svg.append(f'<rect x="{ml+i*bw:.2f}" y="{mt+ch-h:.2f}" width="{max(bw-0.5,0.5):.2f}" '
                   f'height="{h:.2f}" fill="{color}" opacity="0.85"/>')

    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#d1d5db"/>')
    for k in range(5):
        x = ml + (k / 4) * cw
        val = edges[0] + (k / 4) * (edges[-1] - edges[0])
        svg.append(f'<text x="{x:.1f}" y="{mt+ch+16}" text-anchor="middle" font-size="10" '
                   f'fill="#6b7280">{val:.0f}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def progress_html(steps: list[str], current: int, note: str = "") -> str:
    """Pipeline step indicator, matching the house pattern."""
    dots = ""
    for i, s in enumerate(steps):
        if i + 1 < current:
            icon = '<span style="color:#ea580c;">&#10003;</span>'
        elif i + 1 == current:
            icon = '<span style="color:#ea580c;">&#9679;</span>'
        else:
            icon = '<span style="color:#d6d3d1;">&#9675;</span>'
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))
