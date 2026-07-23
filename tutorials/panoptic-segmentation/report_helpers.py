"""Report visuals: three-panel comparison, scene inventory, coverage."""

import base64
import io

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0f172a; border-bottom: 2px solid #7c3aed; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #6d28d9; margin-top: 20px; }
  .report .card { background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #ddd6fe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f172a; }
  .report .stat .label { font-size: 0.85em; color: #64748b; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #4c1d95; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #ede9fe; }
  .report tr:nth-child(even) { background: #f5f3ff; }
  .report .chart-container { background: #fff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #f5f3ff; border-left: 4px solid #7c3aed; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .triptych { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
  .report .triptych figure { margin: 0; }
  .report .triptych img { width: 100%; border-radius: 6px; display: block; }
  .report .triptych figcaption { font-size: 0.8em; color: #64748b; text-align: center; margin-top: 4px; }
  .report .chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .report .chip { font-size: 0.82em; padding: 3px 9px; border-radius: 12px; background: #ede9fe; color: #5b21b6; font-weight: 600; }
  .report .chip.stuff { background: #e2e8f0; color: #334155; }
</style>
"""


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def jpeg_uri(arr_or_img, quality: int = 88) -> str:
    from PIL import Image

    img = arr_or_img if hasattr(arr_or_img, "save") else Image.fromarray(arr_or_img)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def triptych(rgb_uri, pred_uri, gt_uri) -> str:
    return f"""
    <div class="chart-container">
      <div class="triptych">
        <figure><img src="{rgb_uri}"><figcaption>Input image</figcaption></figure>
        <figure><img src="{pred_uri}"><figcaption>Predicted panoptic — Mask2Former</figcaption></figure>
        <figure><img src="{gt_uri}"><figcaption>Ground-truth panoptic (COCO)</figcaption></figure>
      </div>
    </div>
    """


def inventory_chips(inv) -> str:
    things = "".join(f'<span class="chip">{k}{"" if v == 1 else f" ×{v}"}</span>'
                     for k, v in inv["things"].items())
    stuff = "".join(f'<span class="chip stuff">{k}</span>' for k in inv["stuff"])
    return (f'<div style="font-size:.85em;color:#6d28d9;font-weight:600;">Objects '
            f'({inv["n_things"]})</div><div class="chips">{things or "&mdash;"}</div>'
            f'<div style="font-size:.85em;color:#334155;font-weight:600;margin-top:6px;">'
            f'Regions ({inv["n_stuff"]})</div><div class="chips">{stuff or "&mdash;"}</div>')


def make_bar_chart(labels, values, colors=None, title="", width=760, height=300,
                   value_format=",.0f", horizontal=True):
    if not labels:
        return ""
    colors = colors or ["#7c3aed"] * len(labels)
    top = max(max(values), 1) * 1.15
    ml, mr, mt, mb = 150, 60, 34, 18
    cw, ch = width - ml - mr, height - mt - mb
    slot = ch / len(labels)
    bh = min(slot * 0.62, 26)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'style="width:100%;max-width:{width}px;height:auto;">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
    if title:
        svg.append(f'<text x="{width/2}" y="21" text-anchor="middle" font-size="13" '
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


def progress_html(steps, current, note=""):
    dots = ""
    for i, s in enumerate(steps):
        icon = ('<span style="color:#7c3aed;">&#10003;</span>' if i + 1 < current else
                '<span style="color:#7c3aed;">&#9679;</span>' if i + 1 == current else
                '<span style="color:#d6d3d1;">&#9675;</span>')
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))


def frame_player(frame_uris, slug, fps=12, width=900, caption=""):
    """Autoplay loop of pre-composited overlay frames — the segmented video."""
    if not frame_uris:
        return ""
    arr = "[" + ",".join(f'"{u}"' for u in frame_uris) + "]"
    n = len(frame_uris)
    return f"""
    <div class="chart-container" style="background:#0b1016;border-color:#1e293b;">
      <img id="fp-{slug}" src="{frame_uris[0]}" style="width:100%;max-width:{width}px;
           display:block;margin:0 auto;border-radius:6px;">
      <div style="display:flex;align-items:center;gap:10px;max-width:{width}px;margin:10px auto 0;">
        <button id="pb-{slug}" style="background:#7c3aed;color:#fff;border:0;border-radius:5px;
                padding:5px 14px;cursor:pointer;font-size:.9em;min-width:64px;">Pause</button>
        <input id="sc-{slug}" type="range" min="0" max="{n-1}" value="0" style="flex:1;">
        <span id="lb-{slug}" style="color:#94a3b8;font-size:.8em;min-width:62px;text-align:right;
              font-variant-numeric:tabular-nums;">1 / {n}</span>
      </div>
      {f'<div style="text-align:center;color:#64748b;font-size:.8em;margin-top:8px;">{caption}</div>' if caption else ''}
      <script>
      (function() {{
        var F={arr}, i=0, playing=true, t=null;
        var im=document.getElementById('fp-{slug}'), b=document.getElementById('pb-{slug}');
        var sc=document.getElementById('sc-{slug}'), lb=document.getElementById('lb-{slug}');
        F.forEach(function(u){{var p=new Image();p.src=u;}});
        function show(k){{i=(k+F.length)%F.length;im.src=F[i];sc.value=i;lb.textContent=(i+1)+' / '+F.length;}}
        b.addEventListener('click',function(){{playing=!playing;b.textContent=playing?'Pause':'Play';}});
        sc.addEventListener('input',function(){{playing=false;b.textContent='Play';show(parseInt(sc.value,10));}});
        t=setInterval(function(){{if(playing)show(i+1);}},{int(1000/max(fps,1))});
        show(0);
      }})();
      </script>
    </div>
    """
