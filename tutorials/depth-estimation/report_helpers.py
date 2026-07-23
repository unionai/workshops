"""Report visuals: image panels, metric bars, and the parallax player."""

import base64
import io

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0f172a; border-bottom: 2px solid #f97316; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #c2410c; margin-top: 20px; }
  .report .card { background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f172a; }
  .report .stat .label { font-size: 0.85em; color: #78716c; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #7c2d12; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #fed7aa; }
  .report tr:nth-child(even) { background: #fff7ed; }
  .report .chart-container { background: #fff; border: 1px solid #fed7aa; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #fff7ed; border-left: 4px solid #f97316; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .panels { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .report .panels figure { margin: 0; }
  .report .panels img { width: 100%; border-radius: 6px; display: block; }
  .report .panels figcaption { font-size: 0.8em; color: #78716c; text-align: center; margin-top: 4px; }
  .report .legend { display: flex; flex-wrap: wrap; gap: 14px; margin: 8px 0; font-size: 0.85em; align-items: center; }
  .report .ramp { height: 12px; width: 180px; border-radius: 3px;
    background: linear-gradient(90deg,#3010a0,#2b7ef8,#28d9a0,#c8f030,#f8a020,#c81000); }
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


def turbo_legend() -> str:
    return (
        '<div class="legend"><b style="color:#c2410c;">Depth (turbo):</b>'
        '<span>near</span><span class="ramp"></span><span>far</span>'
        '&nbsp;&nbsp;<b style="color:#c2410c;">Error:</b>'
        '<span style="color:#78716c;">dark = accurate, bright = wrong</span></div>'
    )


def panels_html(rgb_uri, pred_uri, gt_uri, err_uri) -> str:
    return f"""
    <div class="chart-container">
      <div class="panels">
        <figure><img src="{rgb_uri}"><figcaption>Input RGB (single photo)</figcaption></figure>
        <figure><img src="{pred_uri}"><figcaption>Predicted depth — Depth Anything V2</figcaption></figure>
        <figure><img src="{gt_uri}"><figcaption>Ground truth — Kinect depth sensor</figcaption></figure>
        <figure><img src="{err_uri}"><figcaption>Absolute error after scale/shift align</figcaption></figure>
      </div>
    </div>
    """


def parallax_player(frame_uris, slug, fps: int = 12, width: int = 640, caption: str = "") -> str:
    """Loop the depth-driven parallax frames — the '3D from one photo' visual."""
    if not frame_uris:
        return ""
    arr = "[" + ",".join(f'"{u}"' for u in frame_uris) + "]"
    return f"""
    <div class="chart-container" style="text-align:center;background:#0b1016;border-color:#1e293b;">
      <img id="px-{slug}" src="{frame_uris[0]}" style="width:100%;max-width:{width}px;
           border-radius:6px;display:inline-block;">
      {f'<div style="color:#94a3b8;font-size:.82em;margin-top:8px;">{caption}</div>' if caption else ''}
      <script>
      (function(){{
        var F={arr}, i=0;
        var im=document.getElementById('px-{slug}');
        F.forEach(function(u){{var p=new Image();p.src=u;}});
        setInterval(function(){{ i=(i+1)%F.length; im.src=F[i]; }}, {int(1000/max(fps,1))});
      }})();
      </script>
    </div>
    """


def make_bar_chart(labels, values, colors=None, title="", width=700, height=260,
                   value_format=".3f", y_max=1.0, higher_better=True):
    if not labels:
        return ""
    colors = colors or ["#f97316"] * len(labels)
    ml, mr, mt, mb = 52, 18, 40, 46
    cw, ch = width - ml - mr, height - mt - mb
    top = y_max if y_max else max(max(values), 1e-9) * 1.15
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'style="width:100%;max-width:{width}px;height:auto;">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#0f172a">{title}</text>')
    for k in range(5):
        y = mt + (k / 4) * ch
        svg.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+cw}" y2="{y:.1f}" stroke="#f5f5f4"/>')
        svg.append(f'<text x="{ml-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" '
                   f'fill="#78716c">{top-(k/4)*top:.2f}</text>')
    slot = cw / len(labels)
    bw = min(slot * 0.6, 70)
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = (val / top) * ch if top else 0
        x = ml + i * slot + (slot - bw) / 2
        y = mt + ch - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                   f'fill="{colors[i%len(colors)]}" rx="3"/>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="11" '
                   f'font-weight="600" fill="#0f172a">{val:{value_format}}</text>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{mt+ch+16}" text-anchor="middle" font-size="10" '
                   f'fill="#57534e">{lab}</text>')
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#d6d3d1"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def progress_html(steps, current, note=""):
    dots = ""
    for i, s in enumerate(steps):
        icon = ('<span style="color:#f97316;">&#10003;</span>' if i + 1 < current else
                '<span style="color:#f97316;">&#9679;</span>' if i + 1 == current else
                '<span style="color:#d6d3d1;">&#9675;</span>')
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))
