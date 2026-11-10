"""The batch job: clinical notes to a fixed JSON schema, scored field by field against the dataset's summary.

This is the only file that knows what a row is. `engine.py` asks it for the rows, the messages for a row, how to
parse a reply, and how to score it; the workflow asks it for the field names to chart and a preview cell. Swap this
module (keep the five functions) and the same pipeline runs a different batch job: a different extraction schema,
classification with a label set, translation with a reference, anything with a gold answer a function can check.

Dataset: AGBonnet/augmented-clinical-notes (MIT): 30k clinical notes, each paired with a 40-field structured summary.
The target here is a five-field projection of that summary (visit motivation, age, sex, symptoms, treatments). The
summaries were themselves produced by a model and then checked, so treat the gold as a strong reference, not truth:
a disagreement on `symptoms` is as often a granularity call as an error, which is why the list fields are compared as
normalised sets and why the report shows the rows side by side.
"""

from __future__ import annotations

import html
import json
import re

NAME = "clinical-notes"
DATASET = "AGBonnet/augmented-clinical-notes"
MAX_TOKENS = 320  # the longest gold JSON is ~120 tokens; headroom for a model that explains itself first

FIELDS = ("visit_motivation", "age", "sex", "symptoms", "treatments")
LIST_FIELDS = ("symptoms", "treatments")

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant. Read the clinical note and return a JSON object with exactly "
    "these keys: visit_motivation (one short phrase: why the patient presented), age (a number only), "
    "sex (\"male\" or \"female\"), symptoms (list of the presenting symptoms as short phrases, no duplicates), "
    "treatments (list of treatments given, as short phrases). Use \"None\" for a string that is not stated and [] for "
    "an empty list. Reply with the JSON object only."
)


# ------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------


def _clean(v) -> str:
    v = "" if v is None else str(v).strip()
    return "None" if v.lower() in ("", "none", "null", "n/a", "not stated", "unknown") else v


def _project(summary: dict) -> dict | None:
    """The dataset's 40-field summary -> the five fields."""
    try:
        info = summary.get("patient information") or {}
        symptoms = [_clean(s.get("name of symptom")) for s in summary.get("symptoms") or [] if isinstance(s, dict)]
        treatments = [_clean(t.get("name")) for t in summary.get("treatments") or [] if isinstance(t, dict)]
        return {
            "visit_motivation": _clean(summary.get("visit motivation")),
            "age": _clean(info.get("age")),
            "sex": _clean(info.get("sex")),
            "symptoms": [s for s in symptoms if s != "None"],
            "treatments": [t for t in treatments if t != "None"],
        }
    except AttributeError:
        return None


def load(n: int, seed: int = 42) -> list[dict]:
    """`n` usable rows: {"id", "input": the note, "gold": the target JSON as a string}."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train").shuffle(seed=seed)
    rows: list[dict] = []
    for i, row in enumerate(ds):
        note = (row.get("note") or "").strip()
        if len(note.split()) < 40 or len(note) > 6000:
            continue
        try:
            summary = json.loads(row["summary"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        target = _project(summary) if isinstance(summary, dict) else None
        if target is None:
            continue
        rows.append({"id": str(row.get("idx", i)), "input": note, "gold": json.dumps(target, ensure_ascii=False)})
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise ValueError(f"Only {len(rows)} usable rows in {DATASET}, need {n}")
    return rows


def messages(row: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Clinical note:\n{row['input']}"}]


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------


def parse(text: str) -> dict | None:
    """The first JSON object in the reply, or None."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


_STOP = {"the", "a", "an", "of", "and", "with", "for", "in", "on", "to", "due", "after", "patient", "presented", "presenting", "complaint", "complaints"}


def _norm(v) -> str:
    s = "none" if v is None else str(v).strip().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return "none" if s in ("", "null", "n a", "not stated", "unknown") else s


def _tokens(v) -> set[str]:
    return {t for t in _norm(v).split() if t not in _STOP and t != "none"}


def _f1(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    hit = len(a & b)
    if not hit:
        return 0.0
    p, r = hit / len(b), hit / len(a)
    return 2 * p * r / (p + r)


def _age(v) -> str:
    m = re.search(r"\d+(\.\d+)?", str(v or ""))
    return m.group(0) if m else _norm(v)


def _sex(v) -> str:
    s = _norm(v)
    return "male" if s in ("m", "male", "man", "boy") else "female" if s in ("f", "female", "woman", "girl") else s


def _list_f1(gold: list, pred) -> float:
    """Set F1 over list items, where an item matches if its tokens overlap the gold item's by at least half."""
    g = [_tokens(x) for x in (gold or []) if _norm(x) != "none"]
    p = [_tokens(x) for x in pred if _norm(x) != "none"] if isinstance(pred, list) else ([_tokens(pred)] if _norm(pred) != "none" else [])
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    matched_g, matched_p = set(), set()
    for i, gi in enumerate(g):
        for j, pj in enumerate(p):
            if j in matched_p or not gi or not pj:
                continue
            if _f1(gi, pj) >= 0.5:
                matched_g.add(i)
                matched_p.add(j)
                break
    prec, rec = len(matched_p) / len(p), len(matched_g) / len(g)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def score(pred: dict | None, gold_json: str) -> dict:
    """Per-field scores in [0, 1]: age and sex exact after normalisation, visit motivation token-F1, lists set-F1 with fuzzy items.

    `fields` holds a boolean per field (a score of at least 0.5 counts as a hit); `field_accuracy` is the mean score;
    `correct` means every field hit. The gold summaries were written by a model and checked, so this is agreement with a
    strong reference at a stated granularity, not truth: the side-by-side rows in the report are the other half.
    """
    gold = json.loads(gold_json)
    if pred is None:
        return {"valid": False, "correct": False, "field_accuracy": 0.0, "fields": {f: False for f in FIELDS}, "scores": {f: 0.0 for f in FIELDS}}
    scores = {
        "visit_motivation": _f1(_tokens(gold.get("visit_motivation")), _tokens(pred.get("visit_motivation"))),
        "age": 1.0 if _age(gold.get("age")) == _age(pred.get("age")) else 0.0,
        "sex": 1.0 if _sex(gold.get("sex")) == _sex(pred.get("sex")) else 0.0,
        "symptoms": _list_f1(gold.get("symptoms"), pred.get("symptoms")),
        "treatments": _list_f1(gold.get("treatments"), pred.get("treatments")),
    }
    hits = {f: scores[f] >= 0.5 for f in FIELDS}
    return {"valid": True, "correct": all(hits.values()), "field_accuracy": sum(scores.values()) / len(FIELDS), "fields": hits, "scores": scores}


def categories() -> list[str]:
    """What the per-category bar chart is over: the fields."""
    return list(FIELDS)


def preview(row: dict, result: dict) -> str:
    """One row for the side-by-side table: the input, the gold, the prediction."""
    pred = result.get("pred")
    return (
        f"<td><details><summary>{html.escape(row['input'][:140])}…</summary><pre style='white-space:pre-wrap;font-size:0.8em'>{html.escape(row['input'])}</pre></details></td>"
        f"<td><pre style='white-space:pre-wrap;margin:0;font-size:0.8em'>{html.escape(json.dumps(json.loads(row['gold']), indent=1, ensure_ascii=False))}</pre></td>"
        f"<td><pre style='white-space:pre-wrap;margin:0;font-size:0.8em'>{html.escape(json.dumps(pred, indent=1, ensure_ascii=False) if pred is not None else (result.get('text') or '(empty)')[:400])}</pre></td>"
    )
