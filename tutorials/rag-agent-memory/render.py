"""HTML for the Flyte reports.

Presentation only — no retrieval logic lives here. It is shared so that a chunk
looks the same in step 1 (where it is the whole answer), step 2 (where it is a
citation) and step 3 (where it is also a dot on a chart). The rank ramp is the
link: chunk #1 wears the darkest blue on its card and the darkest blue on the
chart, and carries the same number in both places.
"""

from __future__ import annotations

import html

# Rank is an *ordered magnitude*, so it gets a sequential single-hue ramp —
# darkest is rank 1, the best match. The previous version here was a red→orange→
# yellow→green ramp, which was a mistake: red reads as "bad" and green as "good",
# so it said the exact opposite of what it meant.
#
# These five steps are validated as an ordinal ramp (monotone lightness, adjacent
# gaps >= 0.06, light end clearing 2:1 on the surface). Adjacent steps are
# deliberately close — that is what makes it a ramp — so rank is NEVER carried by
# color alone: the number is drawn on the marker, repeated in the legend with its
# similarity, and repeated again on the chunk cards.
#
# Plotly renders SVG and cannot read our CSS, so the palette lives in Python and
# the report commits to a light surface.
RANK_COLORS = [
    "#0d366b",  # 1 — best match, darkest
    "#184f95",  # 2
    "#256abf",  # 3
    "#3987e5",  # 4
    "#86b6ef",  # 5 and beyond
]

# Ink for the number drawn inside each marker, paired to its step.
RANK_TEXT = ["#ffffff", "#ffffff", "#ffffff", "#ffffff", "#0b0b0b"]

# The query. Orange is the one candidate that clears 3:1 on the light surface
# (the gold this used to be was 1.37:1) and stays separable from both the blue
# ramp and the gray corpus under protan/deutan/tritan simulation.
QUERY_COLOR = "#eb6834"

# The corpus. Recessive through size and opacity rather than through a color so
# pale it disappears — hundreds of these read as a cloud.
MUTED_COLOR = "#8a8a85"


def rank_color(rank: int) -> str:
    """Color for a 1-based rank, clamped past the end of the ramp."""
    return RANK_COLORS[min(rank, len(RANK_COLORS)) - 1]


def rank_text_color(rank: int) -> str:
    return RANK_TEXT[min(rank, len(RANK_TEXT)) - 1]

# The report commits to a light surface. Plotly draws SVG and cannot read CSS, so
# the chart's palette is fixed in Python; letting the page flip to dark would leave
# those marks validated against a surface that is no longer there. A fixed surface
# also means the chart looks the same for everyone when it is projected.
CSS = """
:root { color-scheme: light; }
body { font-family: ui-sans-serif, system-ui, -apple-system, sans-serif;
       margin: 0; padding: 24px; line-height: 1.55;
       background: #fcfcfb; color: #0b0b0b; }
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
.rank { font-weight: 700; color: #0b0b0b; }
.swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block;
          box-shadow: 0 0 0 1px rgba(0,0,0,.18); }
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
        color = rank_color(hit.rank)
        text = hit.text if len(hit.text) <= max_chars else hit.text[:max_chars] + "…"
        # Similarity is roughly 0-1 in practice; clamp so the bar stays sane.
        width = max(0.0, min(1.0, hit.similarity)) * 100
        cards.append(
            # The swatch carries identity; the number stays in ink. Putting the
            # rank in the series color would leave the pale end of the ramp at
            # 2:1 against the surface, which is not readable as text.
            f"<div class='chunk' style='border-left-color:{color}'>"
            f"<div class='chunk-head'>"
            f"<span class='swatch' style='background:{color}'></span>"
            f"<span class='rank'>#{hit.rank}</span>"
            f"<span class='source'>{html.escape(hit.source)}</span>"
            f"<span class='sim'>similarity {hit.similarity:.3f}</span>"
            f"</div>"
            f"<div class='bar'><span style='width:{width:.1f}%;background:{color}'></span></div>"
            f"<div class='text'>{html.escape(text)}</div>"
            f"</div>"
        )
    return "".join(cards)
