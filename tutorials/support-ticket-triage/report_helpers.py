"""HTML report builder for ticket triage results."""

CATEGORY_COLORS = {
    "outage": "#e74c3c",
    "security": "#e74c3c",
    "bug": "#e67e22",
    "billing": "#f39c12",
    "feature_request": "#3498db",
    "general": "#95a5a6",
}


def build_html(ranked: list[dict]) -> str:
    """Build an HTML triage report from ranked ticket dicts."""
    rows = ""
    for i, t in enumerate(ranked, 1):
        color = CATEGORY_COLORS.get(t.get("category", ""), "#95a5a6")
        bar_width = int(t["priority"] * 100)
        rows += f"""
        <tr>
            <td style="text-align:center;font-weight:bold;">{i}</td>
            <td><span style="background:{color};color:#fff;padding:2px 8px;
                border-radius:4px;font-size:0.85em;">{t.get('category','?')}</span></td>
            <td style="max-width:300px;">{t['ticket']}</td>
            <td>{t.get('summary','')}</td>
            <td>
                <div style="background:#eee;border-radius:4px;height:18px;width:100px;">
                    <div style="background:{color};height:18px;width:{bar_width}px;
                        border-radius:4px;text-align:center;color:#fff;font-size:0.75em;
                        line-height:18px;">{t['priority']:.2f}</div>
                </div>
            </td>
            <td style="font-size:0.9em;">{t.get('suggested_action','')}</td>
        </tr>"""

    # Category summary pills
    cat_counts: dict[str, int] = {}
    for t in ranked:
        cat = t.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    cat_pills = " ".join(
        f'<span style="background:{CATEGORY_COLORS.get(c, "#95a5a6")};color:#fff;'
        f'padding:4px 12px;border-radius:12px;margin:2px;font-size:0.9em;">'
        f'{c}: {n}</span>'
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])
    )

    return f"""
    <div style="font-family:system-ui,sans-serif;max-width:1100px;margin:auto;padding:20px;">
        <h2>Support Ticket Triage Report</h2>
        <p><strong>{len(ranked)}</strong> tickets classified &middot;
           <strong>{sum(1 for t in ranked if t['priority'] >= 0.5)}</strong> high priority</p>

        <div style="margin:12px 0;">{cat_pills}</div>

        <table style="width:100%;border-collapse:collapse;margin-top:16px;">
            <thead>
                <tr style="background:#f8f9fa;border-bottom:2px solid #dee2e6;">
                    <th style="padding:8px;">#</th>
                    <th style="padding:8px;">Category</th>
                    <th style="padding:8px;">Ticket</th>
                    <th style="padding:8px;">Summary</th>
                    <th style="padding:8px;">Priority</th>
                    <th style="padding:8px;">Action</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """
