"""HTML for the Flyte reports.

Presentation only — no retrieval logic lives here. It is shared so that a chunk
looks the same in step 1 (where it is the whole answer), step 2 (where it is a
citation) and step 3 (where it is also a dot on a chart). The rank colors are
the link: chunk #1 is red in the cards and red in the projection.
"""

from __future__ import annotations

import html

# Rank colors, warmest first. Step 3 imports these so the scatter plot and the
# chunk cards agree; Plotly renders SVG and cannot read our CSS, so the palette
# has to live in Python.
RANK_COLORS = [
    "#ef4444",  # 1
    "#f97316",  # 2
    "#eab308",  # 3
    "#22c55e",  # 4
    "#06b6d4",  # 5
    "#3b82f6",  # 6
    "#8b5cf6",  # 7
    "#d946ef",  # 8
    "#ec4899",  # 9
    "#10b981",  # 10
]

QUERY_COLOR = "#ffd700"
MUTED_COLOR = "#cfd5db"

CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
       margin: 0; padding: 24px; line-height: 1.55; }
h1 { font-size: 1.35rem; margin: 0 0 4px; }
h2 { font-size: 1.05rem; margin: 28px 0 10px; }
.sub { color: #6b7280; margin: 0 0 22px; font-size: .9rem; }
.answer { padding: 16px 18px; border-left: 3px solid #6366f1;
          background: rgba(99,102,241,.08); border-radius: 4px; margin-bottom: 26px;
          white-space: pre-wrap; }
.note { padding: 12px 14px; border-radius: 6px; font-size: .85rem;
        background: rgba(127,127,127,.10); margin-bottom: 22px; }
.chunk { border: 1px solid rgba(128,128,128,.28); border-left-width: 4px;
         border-radius: 8px; padding: 12px 14px; margin-bottom: 12px; }
.chunk-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
              font-size: .78rem; margin-bottom: 8px; }
.rank { font-weight: 700; }
.source { font-family: ui-monospace, Menlo, monospace; color: #6b7280;
          overflow-wrap: anywhere; }
.sim { margin-left: auto; font-variant-numeric: tabular-nums; color: #6b7280; }
.bar { height: 4px; border-radius: 2px; background: rgba(128,128,128,.2);
       margin-bottom: 10px; overflow: hidden; }
.bar > span { display: block; height: 100%; }
.text { font-size: .87rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.empty { color: #6b7280; font-style: italic; }
.kv { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
.kv > div { flex: 1 1 150px; padding: 10px 12px; border-radius: 8px;
            border: 1px solid rgba(128,128,128,.25); }
.kv .label { font-size: .7rem; text-transform: uppercase; letter-spacing: .04em;
             color: #6b7280; }
.kv .value { font-size: 1.15rem; font-weight: 600; overflow-wrap: anywhere; }
"""


def page(title: str, subtitle: str, body: str) -> str:
    return (
        f"<style>{CSS}</style>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='sub'>{html.escape(subtitle)}</p>"
        f"{body}"
    )


def stats(**pairs: object) -> str:
    """A row of labelled numbers."""
    cells = "".join(
        f"<div><div class='label'>{html.escape(k.replace('_', ' '))}</div>"
        f"<div class='value'>{html.escape(str(v))}</div></div>"
        for k, v in pairs.items()
    )
    return f"<div class='kv'>{cells}</div>"


def note(text: str) -> str:
    return f"<div class='note'>{text}</div>"


def chunk_cards(hits, max_chars: int = 600) -> str:
    """Render retrieved chunks, colored by rank.

    The similarity bar is worth watching more than the text: a top hit at 0.35
    means nothing in the corpus is close to the question, and any answer built
    on it is the model improvising.
    """
    if not hits:
        return "<p class='empty'>Nothing retrieved.</p>"

    cards = []
    for hit in hits:
        color = RANK_COLORS[(hit.rank - 1) % len(RANK_COLORS)]
        text = hit.text if len(hit.text) <= max_chars else hit.text[:max_chars] + "…"
        # Similarity is roughly 0-1 in practice; clamp so the bar stays sane.
        width = max(0.0, min(1.0, hit.similarity)) * 100
        cards.append(
            f"<div class='chunk' style='border-left-color:{color}'>"
            f"<div class='chunk-head'>"
            f"<span class='rank' style='color:{color}'>#{hit.rank}</span>"
            f"<span class='source'>{html.escape(hit.source)}</span>"
            f"<span class='sim'>similarity {hit.similarity:.3f}</span>"
            f"</div>"
            f"<div class='bar'><span style='width:{width:.1f}%;background:{color}'></span></div>"
            f"<div class='text'>{html.escape(text)}</div>"
            f"</div>"
        )
    return "".join(cards)
