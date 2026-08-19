"""Report visuals for the AV replay pipeline. Self-contained SVG + base64 PNG."""

import base64

from bev import DEFAULT_OBJECT_COLOR, OBJECT_COLORS

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0f172a; border-bottom: 2px solid #38bdf8; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0369a1; margin-top: 20px; }
  .report .card { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #bae6fd; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f172a; }
  .report .stat .label { font-size: 0.85em; color: #64748b; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f172a; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #e0f2fe; }
  .report tr:nth-child(even) { background: #f0f9ff; }
  .report .chart-container { background: #fff; border: 1px solid #bae6fd; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #f0f9ff; border-left: 4px solid #38bdf8; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 10px 0; font-size: 0.85em; }
  .report .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .report .swatch { width: 13px; height: 13px; border-radius: 3px; display: inline-block; border: 1px solid rgba(0,0,0,.15); }
  .report .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:.8em; font-weight:600; background:#e0f2fe; color:#075985; }
</style>
"""


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def png_uri(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def object_legend_html() -> str:
    items = "".join(
        f'<span><i class="swatch" style="background:{c}"></i> {name.replace("_", " ")}</span>'
        for name, c in OBJECT_COLORS.items()
    )
    return (
        '<div class="legend"><b style="font-size:.9em;color:#0369a1;">Objects:</b>'
        f'{items}'
        f'<span><i class="swatch" style="background:{DEFAULT_OBJECT_COLOR}"></i> other</span>'
        '<span><i class="swatch" style="background:#ef4444"></i> ego vehicle</span>'
        '</div>'
        '<div class="legend"><b style="font-size:.9em;color:#0369a1;">Map:</b>'
        '<span><i class="swatch" style="background:#1e293b"></i> lanes</span>'
        '<span><i class="swatch" style="background:#f59e0b"></i> crosswalks</span>'
        '<span><i class="swatch" style="background:#334155"></i> road boundaries</span>'
        '<span><i class="swatch" style="background:#22d3ee"></i> traffic lights</span>'
        '<span><i class="swatch" style="background:#a3e635"></i> traffic signs</span>'
        '</div>'
        '<div style="font-size:.82em;color:#64748b;margin-top:2px;">'
        'Filled box = moving · outline only = stationary · tick = heading · '
        'faint tail = recent track history</div>'
    )


def playback_html(frame_uris: list[str], slug: str, fps: int = 10,
                  width: int = 760, caption: str = "") -> str:
    """
    Frame-by-frame replay with play/pause and a scrubber.

    Frames are pre-rendered PNGs embedded as data URIs and swapped by JS. That keeps the
    report a single self-contained file — no video encoder in the image, no external
    player, and it still scrubs frame-accurately, which a GIF cannot do.
    """
    if not frame_uris:
        return ""
    arr = "[" + ",".join(f'"{u}"' for u in frame_uris) + "]"
    n = len(frame_uris)
    return f"""
    <div class="chart-container" style="background:#080b12;border-color:#1e293b;">
      <img id="img-{slug}" src="{frame_uris[0]}" style="width:100%;max-width:{width}px;
           display:block;margin:0 auto;border-radius:6px;image-rendering:auto;">
      <div style="display:flex;align-items:center;gap:10px;max-width:{width}px;margin:10px auto 0;">
        <button id="play-{slug}" style="background:#0ea5e9;color:#fff;border:0;border-radius:5px;
                padding:5px 14px;cursor:pointer;font-size:.9em;min-width:64px;">Pause</button>
        <input id="scrub-{slug}" type="range" min="0" max="{n - 1}" value="0" style="flex:1;">
        <span id="lbl-{slug}" style="color:#94a3b8;font-size:.8em;min-width:64px;
              text-align:right;font-variant-numeric:tabular-nums;">1 / {n}</span>
      </div>
      {f'<div style="text-align:center;color:#64748b;font-size:.8em;margin-top:8px;">{caption}</div>' if caption else ''}
      <script>
      (function() {{
        var F={arr}, i=0, playing=true, timer=null;
        var img=document.getElementById('img-{slug}'), btn=document.getElementById('play-{slug}');
        var sc=document.getElementById('scrub-{slug}'), lb=document.getElementById('lbl-{slug}');
        // Decode every frame up front so playback doesn't stutter on first pass.
        F.forEach(function(u) {{ var p=new Image(); p.src=u; }});
        function show(k) {{
          i=(k+F.length)%F.length; img.src=F[i]; sc.value=i; lb.textContent=(i+1)+' / '+F.length;
        }}
        function tick() {{ if(playing) show(i+1); }}
        function start() {{ if(timer) clearInterval(timer); timer=setInterval(tick, {int(1000 / max(fps, 1))}); }}
        btn.addEventListener('click', function() {{
          playing=!playing; btn.textContent=playing?'Pause':'Play';
        }});
        sc.addEventListener('input', function() {{
          playing=false; btn.textContent='Play'; show(parseInt(sc.value,10));
        }});
        show(0); start();
      }})();
      </script>
    </div>
    """


def make_bar_chart(labels, values, colors=None, title="", width=700, height=280,
                   value_format=",.0f", horizontal=False):
    if not labels:
        return ""
    colors = colors or ["#38bdf8"] * len(labels)
    top = max(max(values), 1) * 1.15
    if horizontal:
        ml, mr, mt, mb = 130, 60, 36, 20
        cw, ch = width - ml - mr, height - mt - mb
        slot = ch / len(labels)
        bh = min(slot * 0.62, 30)
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
               f'style="width:100%;max-width:{width}px;height:auto;">',
               f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
        if title:
            svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="14" '
                       f'font-weight="600" fill="#0f172a">{title}</text>')
        for i, (lab, val) in enumerate(zip(labels, values)):
            y = mt + i * slot + (slot - bh) / 2
            w = (val / top) * cw
            svg.append(f'<rect x="{ml}" y="{y:.1f}" width="{max(w,1):.1f}" height="{bh:.1f}" '
                       f'fill="{colors[i%len(colors)]}" rx="3"/>')
            svg.append(f'<text x="{ml-8}" y="{y+bh/2+4:.1f}" text-anchor="end" font-size="11" '
                       f'fill="#334155">{lab}</text>')
            svg.append(f'<text x="{ml+w+6:.1f}" y="{y+bh/2+4:.1f}" font-size="11" '
                       f'font-weight="600" fill="#0f172a">{val:{value_format}}</text>')
        svg.append("</svg>")
        return "\n".join(svg)

    ml, mr, mt, mb = 56, 20, 40, 50
    cw, ch = width - ml - mr, height - mt - mb
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'style="width:100%;max-width:{width}px;height:auto;">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
    if title:
        svg.append(f'<text x="{width/2}" y="23" text-anchor="middle" font-size="14" '
                   f'font-weight="600" fill="#0f172a">{title}</text>')
    slot = cw / len(labels)
    bw = min(slot * 0.6, 70)
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = (val / top) * ch
        x = ml + i * slot + (slot - bw) / 2
        y = mt + ch - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                   f'fill="{colors[i%len(colors)]}" rx="3"/>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11" '
                   f'font-weight="600" fill="#0f172a">{val:{value_format}}</text>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{mt+ch+16}" text-anchor="middle" font-size="10" '
                   f'fill="#334155">{lab}</text>')
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#cbd5e1"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def make_line_chart(series: dict, title="", width=760, height=260, x_label="Frame",
                    y_label=""):
    if not series:
        return ""
    ml, mr, mt, mb = 56, 120, 38, 44
    cw, ch = width - ml - mr, height - mt - mb
    allv = [v for _, vals in series.values() for v in vals]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    if hi <= lo:
        hi = lo + 1
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad
    n = max(len(v) for _, v in series.values())

    def sx(i):
        return ml + (i / max(n - 1, 1)) * cw

    def sy(v):
        return mt + ch - ((v - lo) / (hi - lo)) * ch

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'style="width:100%;max-width:{width}px;height:auto;">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#0f172a">{title}</text>')
    for k in range(5):
        y = mt + (k / 4) * ch
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f1f5f9"/>')
        svg.append(f'<text x="{ml-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="#64748b">{hi-(k/4)*(hi-lo):.0f}</text>')
    for idx, (name, (color, vals)) in enumerate(series.items()):
        pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(vals))
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        ly = mt + 10 + idx * 18
        svg.append(f'<rect x="{ml+cw+14}" y="{ly-8}" width="10" height="10" rx="2" fill="{color}"/>')
        svg.append(f'<text x="{ml+cw+29}" y="{ly+1}" font-size="11" fill="#334155">{name}</text>')
    svg.append(f'<text x="{ml+cw/2:.0f}" y="{height-8}" text-anchor="middle" font-size="11" '
               f'fill="#64748b">{x_label}</text>')
    if y_label:
        svg.append(f'<text x="14" y="{mt+ch/2:.0f}" text-anchor="middle" font-size="10" '
                   f'fill="#64748b" transform="rotate(-90,14,{mt+ch/2:.0f})">{y_label}</text>')
    svg.append("</svg>")
    return "\n".join(svg)


def progress_html(steps: list[str], current: int, note: str = "") -> str:
    dots = ""
    for i, s in enumerate(steps):
        if i + 1 < current:
            icon = '<span style="color:#0ea5e9;">&#10003;</span>'
        elif i + 1 == current:
            icon = '<span style="color:#0ea5e9;">&#9679;</span>'
        else:
            icon = '<span style="color:#cbd5e1;">&#9675;</span>'
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))
