"""Report visuals: coverage matrix, surround-view player, distribution charts."""

import base64

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #0f172a; border-bottom: 2px solid #8b5cf6; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #6d28d9; margin-top: 20px; }
  .report .card { background: #f5f3ff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #ddd6fe; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f172a; }
  .report .stat .label { font-size: 0.85em; color: #64748b; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #2e1065; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #ede9fe; }
  .report tr:nth-child(even) { background: #f5f3ff; }
  .report .chart-container { background: #fff; border: 1px solid #ddd6fe; border-radius: 8px; padding: 16px; margin: 16px 0; }
  .report .note { background: #f5f3ff; border-left: 4px solid #8b5cf6; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .warn { background: #fffbeb; border-left: 4px solid #f59e0b; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:.78em; font-weight:600; background:#ede9fe; color:#5b21b6; }
  .report .gap { background:#fee2e2 !important; color:#991b1b; font-weight:600; }
</style>
"""

# Sequential ramp for the coverage matrix: pale = thin coverage, deep = well covered.
RAMP = ["#f5f3ff", "#ddd6fe", "#c4b5fd", "#a78bfa", "#8b5cf6", "#6d28d9"]


def wrap_report(html: str) -> str:
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def jpeg_uri(data: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(data).decode()


def _ramp(t: float) -> str:
    t = max(0.0, min(1.0, t))
    n = len(RAMP) - 1
    i = min(int(t * n), n - 1)
    f = t * n - i

    def hx(c):
        return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)

    r1, g1, b1 = hx(RAMP[i])
    r2, g2, b2 = hx(RAMP[i + 1])
    return f"rgb({int(r1+(r2-r1)*f)},{int(g1+(g2-g1)*f)},{int(b1+(b2-b1)*f)})"


def coverage_matrix_html(matrix: dict, rows: list[str], cols: list[str],
                         title: str = "", row_label: str = "", col_label: str = "") -> str:
    """
    Scenario coverage as a table of counts, with empty cells called out.

    The point of this view is the **zeros**. A coverage matrix that only shows what you
    have is a vanity chart; the useful information is which combinations of scenario and
    condition are missing, because those are the ones a fleet will meet and the model will
    not have seen.
    """
    if not rows or not cols:
        return ""
    vals = [matrix.get((r, c), 0) for r in rows for c in cols]
    hi = max(vals) or 1

    head = "".join(f"<th style='text-align:center;'>{c}</th>" for c in cols)
    body = ""
    for r in rows:
        cells = ""
        for c in cols:
            v = matrix.get((r, c), 0)
            if v == 0:
                cells += "<td class='gap' style='text-align:center;'>0</td>"
            else:
                bg = _ramp(v / hi)
                fg = "#fff" if v / hi > 0.55 else "#1e1b4b"
                cells += (f"<td style='text-align:center;background:{bg};color:{fg};"
                          f"font-weight:600;'>{v}</td>")
        body += f"<tr><td style='font-weight:600;'>{r}</td>{cells}</tr>"

    gaps = sum(1 for r in rows for c in cols if matrix.get((r, c), 0) == 0)
    total_cells = len(rows) * len(cols)
    return f"""
    <div class="chart-container">
      {f'<div style="font-weight:600;color:#0f172a;margin-bottom:8px;">{title}</div>' if title else ''}
      <table>
        <tr><th>{row_label} \\ {col_label}</th>{head}</tr>
        {body}
      </table>
      <div style="font-size:.82em;color:#64748b;">
        <span style="color:#991b1b;font-weight:600;">{gaps}</span> of {total_cells}
        combinations have no coverage.
      </div>
    </div>
    """


def surround_player_html(frame_uris: list[str], slug: str, fps: int = 8,
                         width: int = 900, caption: str = "") -> str:
    """
    Playback of pre-composited surround frames.

    The seven camera views are baked into one image per timestep on the server, so the
    views cannot drift out of sync in the browser — there is only ever one image to swap.
    """
    if not frame_uris:
        return ""
    arr = "[" + ",".join(f'"{u}"' for u in frame_uris) + "]"
    n = len(frame_uris)
    return f"""
    <div class="chart-container" style="background:#080b12;border-color:#1e293b;">
      <img id="im-{slug}" src="{frame_uris[0]}" style="width:100%;max-width:{width}px;
           display:block;margin:0 auto;border-radius:6px;">
      <div style="display:flex;align-items:center;gap:10px;max-width:{width}px;margin:10px auto 0;">
        <button id="pb-{slug}" style="background:#8b5cf6;color:#fff;border:0;border-radius:5px;
                padding:5px 14px;cursor:pointer;font-size:.9em;min-width:64px;">Pause</button>
        <input id="sc-{slug}" type="range" min="0" max="{n-1}" value="0" style="flex:1;">
        <span id="lb-{slug}" style="color:#94a3b8;font-size:.8em;min-width:62px;text-align:right;
              font-variant-numeric:tabular-nums;">1 / {n}</span>
      </div>
      {f'<div style="text-align:center;color:#64748b;font-size:.8em;margin-top:8px;">{caption}</div>' if caption else ''}
      <script>
      (function() {{
        var F={arr}, i=0, playing=true, t=null;
        var im=document.getElementById('im-{slug}'), b=document.getElementById('pb-{slug}');
        var sc=document.getElementById('sc-{slug}'), lb=document.getElementById('lb-{slug}');
        F.forEach(function(u){{ var p=new Image(); p.src=u; }});
        function show(k){{ i=(k+F.length)%F.length; im.src=F[i]; sc.value=i; lb.textContent=(i+1)+' / '+F.length; }}
        b.addEventListener('click', function(){{ playing=!playing; b.textContent=playing?'Pause':'Play'; }});
        sc.addEventListener('input', function(){{ playing=false; b.textContent='Play'; show(parseInt(sc.value,10)); }});
        t=setInterval(function(){{ if(playing) show(i+1); }}, {int(1000/max(fps,1))});
        show(0);
      }})();
      </script>
    </div>
    """


def make_bar_chart(labels, values, colors=None, title="", width=700, height=280,
                   horizontal=True, value_format=",.0f"):
    if not labels:
        return ""
    colors = colors or ["#8b5cf6"] * len(labels)
    top = max(max(values), 1) * 1.15
    if horizontal:
        ml, mr, mt, mb = 150, 70, 34, 18
        cw, ch = width - ml - mr, height - mt - mb
        slot = ch / len(labels)
        bh = min(slot * 0.62, 28)
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

    ml, mr, mt, mb = 52, 18, 38, 52
    cw, ch = width - ml - mr, height - mt - mb
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'style="width:100%;max-width:{width}px;height:auto;">',
           f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>']
    if title:
        svg.append(f'<text x="{width/2}" y="22" text-anchor="middle" font-size="13" '
                   f'font-weight="600" fill="#0f172a">{title}</text>')
    slot = cw / len(labels)
    bw = min(slot * 0.6, 64)
    for i, (lab, val) in enumerate(zip(labels, values)):
        h = (val / top) * ch
        x = ml + i * slot + (slot - bw) / 2
        y = mt + ch - h
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{max(h,0):.1f}" '
                   f'fill="{colors[i%len(colors)]}" rx="3"/>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="10" '
                   f'font-weight="600" fill="#0f172a">{val:{value_format}}</text>')
        svg.append(f'<text x="{x+bw/2:.1f}" y="{mt+ch+16}" text-anchor="middle" font-size="10" '
                   f'fill="#334155">{lab}</text>')
    svg.append(f'<line x1="{ml}" y1="{mt+ch}" x2="{ml+cw}" y2="{mt+ch}" stroke="#cbd5e1"/>')
    svg.append("</svg>")
    return "\n".join(svg)


def progress_html(steps: list[str], current: int, note: str = "") -> str:
    dots = ""
    for i, s in enumerate(steps):
        if i + 1 < current:
            icon = '<span style="color:#8b5cf6;">&#10003;</span>'
        elif i + 1 == current:
            icon = '<span style="color:#8b5cf6;">&#9679;</span>'
        else:
            icon = '<span style="color:#cbd5e1;">&#9675;</span>'
        dots += f"<span style='margin:0 8px;white-space:nowrap;'>{icon} {s}</span>"
    return (f'<div class="card" style="text-align:center;">{dots}</div>'
            + (f"<p>{note}</p>" if note else ""))
