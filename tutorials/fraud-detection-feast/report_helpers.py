"""Report helpers — styled HTML components for Flyte task reports."""

REPORT_CSS = """
<style>
  .report { font-family: system-ui, -apple-system, sans-serif; max-width: 960px; margin: 0 auto; color: #1a1a2e; }
  .report h2 { color: #16213e; border-bottom: 2px solid #0f3460; padding-bottom: 8px; margin-top: 24px; }
  .report h3 { color: #0f3460; margin-top: 20px; }
  .report .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .report .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin: 12px 0; }
  .report .stat { background: #fff; border: 1px solid #e9ecef; border-radius: 6px; padding: 12px; text-align: center; }
  .report .stat .value { font-size: 1.5em; font-weight: 700; color: #0f3460; }
  .report .stat .label { font-size: 0.85em; color: #6c757d; margin-top: 4px; }
  .report table { border-collapse: collapse; width: 100%; margin: 12px 0; }
  .report th { background: #0f3460; color: #fff; padding: 10px 14px; text-align: left; font-weight: 600; }
  .report td { padding: 8px 14px; border-bottom: 1px solid #dee2e6; }
  .report tr:nth-child(even) { background: #f8f9fa; }
  .report .highlight { color: #0f3460; font-weight: 700; }
  .report .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 600; }
  .report .badge-success { background: #d4edda; color: #155724; }
  .report .badge-info { background: #d1ecf1; color: #0c5460; }
  .report .badge-warning { background: #fff3cd; color: #856404; }
  .report .badge-danger { background: #f8d7da; color: #721c24; }
  .report .note { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 14px; border-radius: 4px; margin: 12px 0; font-size: 0.9em; }
</style>
"""


def wrap(html: str) -> str:
    """Wrap HTML content with report styling."""
    return f'{REPORT_CSS}<div class="report">{html}</div>'


def stat_grid(stats: list[tuple[str, str]]) -> str:
    """Render a grid of KPI stat cards. Each stat is (value, label)."""
    cards = ""
    for value, label in stats:
        cards += f'<div class="stat"><div class="value">{value}</div><div class="label">{label}</div></div>'
    return f'<div class="stat-grid">{cards}</div>'


def confusion_matrix_html(cm, labels: list[str] = ("Legit", "Fraud")) -> str:
    """Render a styled confusion matrix table."""
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total * 100
    precision = tp / max(tp + fp, 1) * 100
    recall = tp / max(tp + fn, 1) * 100

    return (
        '<h3>Confusion Matrix</h3>'
        '<table>'
        f'<tr><th></th><th>Predicted {labels[0]}</th><th>Predicted {labels[1]}</th><th></th></tr>'
        f'<tr><td><b>Actual {labels[0]}</b></td>'
        f'<td style="background:#d4edda;color:#155724;font-weight:600;text-align:center;">{tn:,}</td>'
        f'<td style="background:#f8d7da;color:#721c24;text-align:center;">{fp:,}</td>'
        f'<td style="font-size:0.85em;color:#6c757d;">FP rate: {fp / max(tn + fp, 1) * 100:.2f}%</td></tr>'
        f'<tr><td><b>Actual {labels[1]}</b></td>'
        f'<td style="background:#f8d7da;color:#721c24;text-align:center;">{fn:,}</td>'
        f'<td style="background:#d4edda;color:#155724;font-weight:600;text-align:center;">{tp:,}</td>'
        f'<td style="font-size:0.85em;color:#6c757d;">Recall: {recall:.1f}%</td></tr>'
        f'<tr><td></td><td colspan="2" style="font-size:0.85em;color:#6c757d;text-align:center;">'
        f'Accuracy: {accuracy:.1f}% | Precision: {precision:.1f}%</td><td></td></tr>'
        '</table>'
    )


def horizontal_bar_chart(
    labels: list[str],
    values: list[float],
    title: str = "",
    color: str = "#0f3460",
    width: int = 700,
    value_fmt: str = ".4f",
) -> str:
    """Generate an SVG horizontal bar chart."""
    n = len(labels)
    row_height = max(22, min(32, 300 // max(n, 1)))
    ml = 160  # left margin for labels
    mr = 60
    mt = 30
    cw = width - ml - mr
    actual_height = mt + n * row_height + 20
    bar_h = row_height - 6

    v_max = max(values) if values else 1

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {actual_height}" '
           f'style="width:100%;max-width:{width}px;">']

    if title:
        svg.append(f'<text x="{width // 2}" y="18" text-anchor="middle" '
                   f'font-size="14" font-weight="600" fill="#1a1a2e">{title}</text>')

    for i, (label, val) in enumerate(zip(labels, values)):
        y = mt + i * row_height
        bw = (val / v_max) * cw if v_max else 0
        # Truncate long labels
        display_label = label if len(label) <= 20 else label[:18] + "..."
        svg.append(f'<text x="{ml - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
                   f'font-size="11" fill="#1a1a2e">{display_label}</text>')
        svg.append(f'<rect x="{ml}" y="{y:.1f}" width="{bw:.1f}" height="{bar_h:.1f}" '
                   f'fill="{color}" rx="3" opacity="0.85"/>')
        svg.append(f'<text x="{ml + bw + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}" '
                   f'font-size="11" fill="#0f3460" font-weight="600">{val:{value_fmt}}</text>')

    svg.append('</svg>')
    return '\n'.join(svg)


def pipeline_step_indicator(current: int, steps: list[str]) -> str:
    """Build a visual step indicator showing progress through pipeline stages."""
    html = '<div style="display:flex;gap:24px;margin:16px 0;justify-content:center;">'
    for i, label in enumerate(steps):
        if i < current:
            icon = '<span style="color:#155724;font-size:1.3em;">&#10003;</span>'
            color = "#155724"
            bg = "#d4edda"
        elif i == current:
            icon = '<span style="color:#0f3460;font-size:1.3em;">&#9679;</span>'
            color = "#0f3460"
            bg = "#d1ecf1"
        else:
            icon = '<span style="color:#adb5bd;font-size:1.3em;">&#9675;</span>'
            color = "#adb5bd"
            bg = "#f8f9fa"
        html += (
            f'<div style="text-align:center;background:{bg};border-radius:8px;padding:8px 16px;">'
            f'<div>{icon}</div>'
            f'<div style="font-size:0.85em;color:{color};font-weight:600;">{label}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def class_distribution_bar(n_legit: int, n_fraud: int) -> str:
    """Render a stacked bar showing class imbalance."""
    total = n_legit + n_fraud
    fraud_pct = n_fraud / total * 100
    legit_pct = n_legit / total * 100
    return (
        '<div class="card">'
        '<div style="font-weight:600;margin-bottom:8px;">Class Distribution</div>'
        f'<div style="display:flex;border-radius:6px;overflow:hidden;height:28px;">'
        f'<div style="background:#0f3460;width:{legit_pct:.1f}%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:0.8em;font-weight:600;">'
        f'Legit {legit_pct:.1f}%</div>'
        f'<div style="background:#e74c3c;width:{fraud_pct:.1f}%;display:flex;align-items:center;'
        f'justify-content:center;color:#fff;font-size:0.8em;font-weight:600;">'
        f'Fraud {fraud_pct:.1f}%</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.8em;color:#6c757d;margin-top:4px;">'
        f'<span>{n_legit:,} transactions</span><span>{n_fraud:,} transactions</span></div>'
        '</div>'
    )
