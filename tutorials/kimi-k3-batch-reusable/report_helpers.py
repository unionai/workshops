"""Flyte report helpers — CSS, SVG charts, and HTML wrappers."""

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f3460; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  .report tr:nth-child(even) { background: #f8f9fa; }
  .report .highlight { color: #0f3460; font-weight: 700; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d4edda; color: #155724; }
  .report .badge-info { background: #d1ecf1; color: #0c5460; }
  .report .badge-danger { background: #f8d7da; color: #721c24; }
  .report .chart-container { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0; }
</style>
"""


def wrap_report(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def make_line_chart(
    data: list[dict],
    x_key: str,
    y_keys: list[str],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    y_max_cap: float | None = None,
    x_range_override: tuple[float, float] | None = None,
    y_display_names: dict[str, str] | None = None,
) -> str:
    """Generate an SVG line chart from a list of dicts."""
    default_colors = ["#5a7db5", "#0f3460", "#06d6a0", "#ffc107", "#6c757d"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    x_vals = [d[x_key] for d in data] if data else []
    if x_range_override:
        x_min, x_max = x_range_override
    elif x_vals:
        x_min, x_max = min(x_vals), max(x_vals)
    else:
        x_min, x_max = 0, 1
    x_range = x_max - x_min or 1

    all_y = []
    for key in y_keys:
        all_y.extend(d[key] for d in data if key in d)
    y_min = min(all_y) if all_y else 0
    y_max = max(all_y) if all_y else 1
    y_pad = (y_max - y_min) * 0.1 or 0.1
    y_min_plot = max(0, y_min - y_pad)
    y_max_plot = y_max + y_pad
    if y_max_cap is not None:
        y_max_plot = min(y_max_plot, y_max_cap)
    y_range = y_max_plot - y_min_plot or 1

    def sx(v):
        return ml + (v - x_min) / x_range * cw

    def sy(v):
        return mt + ch - (v - y_min_plot) / y_range * ch

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Grid lines
    for i in range(6):
        y_tick = y_min_plot + y_range * i / 5
        py = sy(y_tick)
        lines.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        whole = all(float(v).is_integer() for v in all_y) and y_range >= 5
        lines.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.0f}</text>' if whole else
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.3f}</text>'
        )

    # Axes
    lines.append(
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ch}" '
        f'stroke="#adb5bd" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{ml}" y1="{mt + ch}" x2="{ml + cw}" y2="{mt + ch}" '
        f'stroke="#adb5bd" stroke-width="1.5"/>'
    )

    # X-axis ticks
    if x_vals:
        n_x_ticks = min(len(data), 10)
        step = max(1, len(data) // n_x_ticks)
        for i in range(0, len(data), step):
            px = sx(x_vals[i])
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_vals[i]:.1f}</text>'
            )
    else:
        for i in range(6):
            x_tick = x_min + x_range * i / 5
            px = sx(x_tick)
            lines.append(
                f'<text x="{px:.1f}" y="{mt + ch + 20}" text-anchor="middle" '
                f'font-size="11" fill="#6c757d">{x_tick:.1f}</text>'
            )

    # Plot series
    if not data:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="13" fill="#adb5bd" font-style="italic">Waiting for data...</text>'
        )
    for si, key in enumerate(y_keys):
        color = colors[si % len(colors)]
        points = [(sx(d[x_key]), sy(d[key])) for d in data if key in d]
        if not points:
            continue
        if len(points) >= 2:
            path_d = f"M {points[0][0]:.1f},{points[0][1]:.1f}"
            for px, py in points[1:]:
                path_d += f" L {px:.1f},{py:.1f}"
            dash = ' stroke-dasharray="6,3"' if si % 2 == 1 else ""
            lines.append(
                f'<path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="2" stroke-linejoin="round"{dash}/>'
            )
        if len(points) <= 30:
            for px, py in points:
                lines.append(
                    f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{color}"/>'
                )

    if title:
        lines.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )
    if x_label:
        lines.append(
            f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d">{x_label}</text>'
        )
    if y_label:
        lines.append(
            f'<text x="14" y="{mt + ch / 2}" text-anchor="middle" '
            f'font-size="12" fill="#6c757d" '
            f'transform="rotate(-90, 14, {mt + ch / 2})">{y_label}</text>'
        )

    # Legend
    names = y_display_names or {}
    if len(y_keys) > 1:
        lx = ml + 10
        for si, key in enumerate(y_keys):
            color = colors[si % len(colors)]
            ly = mt + 14 + si * 18
            lines.append(
                f'<rect x="{lx}" y="{ly - 6}" width="12" height="12" '
                f'rx="2" fill="{color}"/>'
            )
            label = names.get(key, key)
            lines.append(
                f'<text x="{lx + 16}" y="{ly + 4}" font-size="11" '
                f'fill="#1a1a2e">{label}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines)


def make_bar_chart(
    labels: list[str],
    series: dict[str, list[float]],
    title: str = "",
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 300,
    y_max_cap: float | None = None,
    value_suffix: str = "%",
    value_fmt: str = "{:.1f}",
) -> str:
    """Generate an SVG grouped bar chart. Every bar carries its value (`value_fmt` + `value_suffix`)."""
    if not labels:
        return ""

    default_colors = ["#adb5bd", "#0f3460", "#06d6a0", "#5a7db5"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 60
    cw = width - ml - mr
    ch = height - mt - mb

    all_vals = [v for vals in series.values() for v in vals]
    y_max = max(all_vals) if all_vals else 1
    y_max_plot = y_max * 1.15 or 1
    if y_max_cap is not None:
        y_max_plot = min(y_max_plot, y_max_cap) or y_max_cap

    n_groups = len(labels)
    n_series = len(series)
    group_width = cw / n_groups
    bar_width = group_width * 0.7 / max(n_series, 1)
    gap = group_width * 0.15

    def sy(v):
        return mt + ch - (v / y_max_plot) * ch

    lines_svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        lines_svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        lines_svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.1f}</text>'
        )

    for gi, label in enumerate(labels):
        gx = ml + gi * group_width + gap
        for si, (name, vals) in enumerate(series.items()):
            color = colors[si % len(colors)]
            bx = gx + si * bar_width
            val = vals[gi]
            by = sy(val)
            bh = mt + ch - by
            lines_svg.append(
                f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width - 1:.1f}" '
                f'height="{bh:.1f}" fill="{color}" rx="2"/>'
            )
            lines_svg.append(
                f'<text x="{bx + bar_width / 2:.1f}" y="{by - 4:.1f}" '
                f'text-anchor="middle" font-size="10" fill="#1a1a2e">'
                f'{value_fmt.format(val)}{value_suffix}</text>'
            )
        lines_svg.append(
            f'<text x="{gx + n_series * bar_width / 2:.1f}" y="{mt + ch + 18}" '
            f'text-anchor="middle" font-size="11" fill="#6c757d">{label}</text>'
        )

    if title:
        lines_svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    lx = ml + cw - len(series) * 100
    for si, name in enumerate(series):
        color = colors[si % len(colors)]
        lines_svg.append(
            f'<rect x="{lx + si * 100}" y="{mt + ch + 35}" width="12" '
            f'height="12" rx="2" fill="{color}"/>'
        )
        lines_svg.append(
            f'<text x="{lx + si * 100 + 16}" y="{mt + ch + 46}" font-size="11" '
            f'fill="#1a1a2e">{name}</text>'
        )

    lines_svg.append("</svg>")
    return "\n".join(lines_svg)


def pipeline_step_indicator(current: int, steps: list[str]) -> str:
    """Build a visual step indicator (checkmark/dot/circle)."""
    html = '<div style="display:flex;gap:24px;margin:16px 0;">'
    for i, label in enumerate(steps):
        if i < current:
            icon = '<span style="color:#155724;font-size:1.2em;">&#10003;</span>'
            color = "#155724"
        elif i == current:
            icon = '<span style="color:#0f3460;font-size:1.2em;">&#9679;</span>'
            color = "#0f3460"
        else:
            icon = '<span style="color:#adb5bd;font-size:1.2em;">&#9675;</span>'
            color = "#adb5bd"
        html += f'<div style="text-align:center;"><div>{icon}</div><div style="font-size:0.85em;color:{color};">{label}</div></div>'
    html += '</div>'
    return html


def make_timeline_chart(spans: list[dict], title: str = "", width: int = 900, row_height: int = 22, color: str = "#5a7db5") -> str:
    """A Gantt-style timeline: one bar per span ({label, start, end} in seconds from the same origin), value labels on each."""
    if not spans:
        return ""
    ml, mr, mt, mb = 110, 20, 36, 30
    height = mt + mb + row_height * len(spans)
    cw = width - ml - mr
    t_max = max(s["end"] for s in spans) or 1

    def sx(t):
        return ml + t / t_max * cw

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    for i in range(6):
        t = t_max * i / 5
        px = sx(t)
        out.append(f'<line x1="{px:.1f}" y1="{mt}" x2="{px:.1f}" y2="{mt + row_height * len(spans)}" stroke="#e9ecef" stroke-width="1"/>')
        out.append(f'<text x="{px:.1f}" y="{mt + row_height * len(spans) + 16}" text-anchor="middle" font-size="11" fill="#6c757d">{t:.0f}s</text>')
    for i, sp in enumerate(spans):
        y = mt + i * row_height + 3
        x0, x1 = sx(sp["start"]), sx(sp["end"])
        out.append(f'<text x="{ml - 8}" y="{y + row_height / 2 + 1:.1f}" text-anchor="end" font-size="11" fill="#1a1a2e">{sp["label"]}</text>')
        out.append(f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 2):.1f}" height="{row_height - 6}" fill="{sp.get("color", color)}" rx="3"/>')
        label = sp.get("text", f'{sp["end"] - sp["start"]:.0f}s')
        tx = x1 + 6 if x1 + 60 < ml + cw else x0 - 6
        anchor = "start" if x1 + 60 < ml + cw else "end"
        fill = "#1a1a2e" if anchor == "start" else "#fff"
        out.append(f'<text x="{tx:.1f}" y="{y + row_height / 2 + 1:.1f}" text-anchor="{anchor}" font-size="10" fill="{fill}">{label}</text>')
    if title:
        out.append(f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')
    out.append("</svg>")
    return "\n".join(out)


def make_histogram(values: list[float], bins: int = 20, title: str = "", x_label: str = "", width: int = 700, height: int = 260, color: str = "#5a7db5") -> str:
    """An SVG histogram with the count on each bar that has one."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    hi = hi if hi > lo else lo + 1
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    ml, mr, mt, mb = 50, 20, 36, 44
    cw, ch = width - ml - mr, height - mt - mb
    c_max = max(counts) or 1
    bw = cw / bins
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    for i in range(5):
        y = mt + ch - ch * i / 4
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + cw}" y2="{y:.1f}" stroke="#e9ecef"/>')
        out.append(f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#6c757d">{c_max * i / 4:.0f}</text>')
    for i, c in enumerate(counts):
        h = ch * c / c_max
        x = ml + i * bw
        out.append(f'<rect x="{x + 1:.1f}" y="{mt + ch - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" fill="{color}" rx="2"/>')
        if c:
            out.append(f'<text x="{x + bw / 2:.1f}" y="{mt + ch - h - 3:.1f}" text-anchor="middle" font-size="9" fill="#1a1a2e">{c}</text>')
    for i in range(0, bins + 1, max(1, bins // 5)):
        out.append(f'<text x="{ml + i * bw:.1f}" y="{mt + ch + 16}" text-anchor="middle" font-size="11" fill="#6c757d">{lo + i * step:.1f}</text>')
    if x_label:
        out.append(f'<text x="{ml + cw / 2}" y="{height - 6}" text-anchor="middle" font-size="12" fill="#6c757d">{x_label}</text>')
    if title:
        out.append(f'<text x="{width / 2}" y="20" text-anchor="middle" font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')
    out.append("</svg>")
    return "\n".join(out)


def stat(value, label: str) -> str:
    return f"<div class='stat'><div class='value'>{value}</div><div class='label'>{label}</div></div>"
