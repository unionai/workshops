"""Two ways to cut documents into chunks, so you can compare them.

Chunking is the most consequential decision in a RAG pipeline and the least
discussed. One vector has to stand for one chunk, so where you cut decides what
each vector *means*. Treat it as a hyperparameter, not plumbing.

    character   the naive baseline: split every ~1200 characters, preferring
                paragraph breaks. Fast, general, structure-blind.
    structural  (default) split on markdown headings, keep fenced code blocks
                whole, and stamp each chunk with the heading path it came from.

Measured on this tutorial's own corpus, `character` produces chunks where 23%
contain an unclosed code fence and 69% begin mid-sentence — one starts literally
`w.py pipeline \\`, which is the tail of a shell command whose beginning lives in
a different vector. That chunk cannot answer anything.
"""

from __future__ import annotations

import re

# A fenced code block: ``` or ~~~ with optional language, through the closing fence.
_FENCE = re.compile(r"^(```|~~~)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


# ── The naive baseline ────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    """Recursive character splitter: paragraphs → lines → sentences → words → chars.

    Character-based, not token-based, so we don't drag in a tokenizer just to
    split. It tries the biggest natural break first and only falls back to
    cutting mid-word when a single run of text has no break in it at all.

    It knows nothing about markdown, which is exactly the problem `split_markdown`
    exists to fix.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in ("\n\n", "\n", ". ", " ", ""):
        if sep == "":
            step = max(1, chunk_size - overlap)
            return [text[i:i + chunk_size] for i in range(0, len(text), step)]
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        out: list[str] = []
        buf = ""
        for part in parts:
            piece = part + sep
            if len(piece) > chunk_size:
                if buf.strip():
                    out.append(buf.strip())
                    buf = ""
                out.extend(split_text(part, chunk_size, overlap))
                continue
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    out.append(buf.strip())
                buf = (buf[-overlap:] if overlap and buf else "") + piece
        if buf.strip():
            out.append(buf.strip())
        return [c for c in out if c.strip()]
    return [text]


# ── Structure-aware ───────────────────────────────────────────────────────────

def _units(text: str) -> list[tuple[list[str], str]]:
    """Split markdown into (heading_path, block) pairs.

    A "unit" is one paragraph or one **whole fenced code block**. Code blocks are
    kept atomic because half a shell command retrieves as well as no shell
    command. The heading path travels with each unit so a chunk can say where in
    the document it came from.
    """
    units: list[tuple[list[str], str]] = []
    stack: list[str] = []
    buf: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        block = "\n".join(buf).strip()
        if block:
            units.append((list(stack), block))
        buf.clear()

    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            if not in_fence:
                flush()                       # the code block starts its own unit
                in_fence, fence_marker = True, fence.group(1)
                buf.append(line)
            elif line.startswith(fence_marker):
                buf.append(line)
                in_fence = False
                flush()                       # ...and ends it, kept whole
            else:
                buf.append(line)
            continue

        if in_fence:                          # headings inside code are not headings
            buf.append(line)
            continue

        heading = _HEADING.match(line)
        if heading:
            flush()
            depth = len(heading.group(1))
            del stack[depth - 1:]             # pop to this level, then push
            stack.append(heading.group(2).strip())
            continue

        if not line.strip():
            flush()                           # blank line ends a paragraph
        else:
            buf.append(line)

    flush()
    return units


def split_markdown(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 150,
    breadcrumbs: bool = True,
) -> list[str]:
    """Chunk on document structure instead of character count.

    Three things this does that character splitting cannot:

    1. **Never cuts a fenced code block.** A chunk holding half a command is
       worse than useless — it retrieves and then misleads.
    2. **Starts chunks at real boundaries**, so a chunk begins with a heading or
       a sentence rather than mid-word.
    3. **Stamps the heading path onto the chunk.** `bge-small` embeds the text it
       is given and nothing else, so a paragraph about "reward functions" that
       never says "GRPO" is invisible to a GRPO query. The breadcrumb puts that
       context *into the vector* — a cheap stand-in for the "contextual
       retrieval" trick of having a model write a summary sentence per chunk.

    Oversized units (a very long code block, a wall-of-text paragraph) fall back
    to `split_text`, because something has to give and a hard cut is the honest
    last resort.
    """
    out: list[str] = []
    current: list[str] = []
    current_path: list[str] = []
    size = 0

    def emit() -> None:
        nonlocal current, size
        if not current:
            return
        body = "\n\n".join(current).strip()
        if body:
            crumb = " > ".join(current_path)
            out.append(f"[{crumb}]\n\n{body}" if breadcrumbs and crumb else body)
        current, size = [], 0

    for path, block in _units(text):
        # A new section starts a new chunk: mixing two sections into one vector
        # is what makes an embedding mean "several things vaguely".
        if path != current_path:
            emit()
            current_path = path

        if len(block) > chunk_size:
            emit()
            saved = current_path
            for piece in split_text(block, chunk_size, overlap):
                current, current_path, size = [piece], saved, len(piece)
                emit()
            current_path = saved
            continue

        if size + len(block) > chunk_size:
            emit()
        current.append(block)
        size += len(block) + 2

    emit()
    return [c for c in out if c.strip()]


def split_document(text: str, chunk_size: int = 1200, overlap: int = 150,
                   strategy: str = "structural") -> list[str]:
    """Dispatch on strategy so step 0 can offer both from one flag."""
    if strategy == "structural":
        return split_markdown(text, chunk_size, overlap)
    if strategy == "character":
        return split_text(text, chunk_size, overlap)
    raise ValueError(f"Unknown --chunking {strategy!r}. Expected 'structural' or 'character'.")
