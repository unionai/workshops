"""Lightweight HTML + inline-SVG report helpers.

Shared styling and charts for the Flyte task reports in this tutorial. We render
charts as inline SVG (no matplotlib) so reports stay small and dependency-free —
the same approach used by the other CV tutorials in this repo.
"""

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report p { line-height: 1.5; }
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
  .report .win { color: #06d6a0; font-weight: 700; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
  .report .thumbs { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0; }
  .report .thumbs img { height: 96px; border-radius: 6px; border: 1px solid #dee2e6; }
  .report .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; margin: 16px 0; }
  .report .gallery figure { margin: 0; }
  .report .gallery img { width: 100%; border-radius: 6px; border: 1px solid #dee2e6; display: block; }
  .report .gallery figcaption { font-size: 0.85em; color: #6c757d; text-align: center; margin-top: 6px; }
  .report .chart-container { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 16px 0; }
</style>
"""


def wrap_report(html: str) -> str:
    """Wrap HTML content in the shared report shell."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def stat(value: str, label: str) -> str:
    """A single stat tile for a `.stat-grid`."""
    return f'<div class="stat"><div class="value">{value}</div><div class="label">{label}</div></div>'


def bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    unit: str = "",
    decimals: int = 0,
    colors: list[str] | None = None,
    width: int = 700,
    height: int = 320,
) -> str:
    """Single-series vertical bar chart as inline SVG.

    Args:
        labels: One category label per bar.
        values: One value per bar (same length as labels).
        title: Chart title.
        unit: Suffix appended to the value label on top of each bar (e.g. " img/s").
        decimals: Decimal places for value + axis labels (use 2–3 for 0–1 metrics
            like AP, 0 for large counts like img/s).
        colors: Per-bar fill colors. Cycles if shorter than the number of bars.
        width, height: SVG canvas size.

    Returns:
        An SVG string sized to fit its container.
    """
    if not labels:
        return ""

    default_colors = ["#adb5bd", "#06d6a0", "#0f3460", "#5a7db5"]
    colors = colors or default_colors

    ml, mr, mt, mb = 60, 20, 40, 50
    cw = width - ml - mr
    ch = height - mt - mb

    y_max = max(values) if values else 1
    y_max_plot = (y_max * 1.15) or 1

    n = len(labels)
    slot = cw / n
    bar_width = slot * 0.55

    def sy(v: float) -> float:
        return mt + ch - (v / y_max_plot) * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]

    # Horizontal grid lines + y ticks.
    for i in range(6):
        y_tick = y_max_plot * i / 5
        py = sy(y_tick)
        svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{y_tick:.{decimals}f}</text>'
        )

    # Bars.
    for i, (label, val) in enumerate(zip(labels, values)):
        bx = ml + i * slot + (slot - bar_width) / 2
        by = sy(val)
        bh = mt + ch - by
        color = colors[i % len(colors)]
        svg.append(
            f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_width:.1f}" '
            f'height="{bh:.1f}" fill="{color}" rx="3"/>'
        )
        svg.append(
            f'<text x="{bx + bar_width / 2:.1f}" y="{by - 6:.1f}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="#1a1a2e">{val:,.{decimals}f}{unit}</text>'
        )
        svg.append(
            f'<text x="{bx + bar_width / 2:.1f}" y="{mt + ch + 18}" text-anchor="middle" '
            f'font-size="11" fill="#6c757d">{label}</text>'
        )

    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def line_chart(
    values: list[float],
    title: str = "",
    x_label: str = "step",
    y_label: str = "",
    color: str = "#0f3460",
    width: int = 700,
    height: int = 300,
) -> str:
    """Single-series line chart (e.g. a loss curve) as inline SVG. `values` is
    plotted against its own index."""
    if not values:
        return ""

    ml, mr, mt, mb = 60, 20, 40, 46
    cw = width - ml - mr
    ch = height - mt - mb

    y_max = max(values)
    y_min = min(values)
    span = (y_max - y_min) or 1.0
    y_hi = y_max + span * 0.1
    y_lo = y_min - span * 0.1
    n = len(values)

    def sx(i: int) -> float:
        return ml + (i / max(n - 1, 1)) * cw

    def sy(v: float) -> float:
        return mt + ch - ((v - y_lo) / (y_hi - y_lo)) * ch

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="width:100%;max-width:{width}px;height:auto;">',
        f'<rect width="{width}" height="{height}" fill="#fff" rx="6"/>',
    ]
    for i in range(6):
        v = y_lo + (y_hi - y_lo) * i / 5
        py = sy(v)
        svg.append(
            f'<line x1="{ml}" y1="{py:.1f}" x2="{ml + cw}" y2="{py:.1f}" '
            f'stroke="#e9ecef" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{ml - 8}" y="{py + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="#6c757d">{v:.2f}</text>'
        )

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
    svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')

    if title:
        svg.append(
            f'<text x="{width / 2}" y="22" text-anchor="middle" '
            f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>'
        )
    svg.append(
        f'<text x="{ml + cw / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="11" fill="#6c757d">{x_label}</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)
