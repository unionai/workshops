"""Shared state for the toy Acme ERP: one SQLite file used by the REST API and the desktop client.

Supplier prices are a pure function of the date, so the portal, the scripted flow and the
verification step agree on what "today's price changes" are without sharing state.
"""

from __future__ import annotations

import os
import random
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = Path(os.environ.get("ACME_DB", "/srv/acme/acme.db"))

# (sku, description, category, list price). Building-products flavoured, to match a distributor.
CATALOG = [
    ("SHG-30-CHR", "Architectural shingle, Charcoal, 33.3 sqft bundle", "Roofing", 42.50),
    ("SHG-30-WWD", "Architectural shingle, Weathered Wood, bundle", "Roofing", 42.50),
    ("SHG-30-SLT", "Architectural shingle, Slate, bundle", "Roofing", 43.10),
    ("SHG-3T-BLK", "3-tab shingle, Black, bundle", "Roofing", 31.75),
    ("UND-SYN-10", "Synthetic underlayment, 10 sq roll", "Roofing", 118.00),
    ("UND-ICE-2", "Ice & water shield, 2 sq roll", "Roofing", 96.40),
    ("RDG-CAP-CHR", "Hip & ridge cap, Charcoal, bundle", "Roofing", 64.90),
    ("DRP-EDG-WHT", "Drip edge, aluminium, White, 10 ft", "Roofing", 8.95),
    ("NAI-COIL-125", "Coil roofing nails 1-1/4in, 7200 ct", "Fasteners", 54.20),
    ("NAI-CAP-1", "Plastic cap nails 1in, 2000 ct", "Fasteners", 23.80),
    ("SCR-DECK-3", "Deck screws 3in, 5 lb box", "Fasteners", 36.15),
    ("GUT-K5-WHT", "K-style gutter 5in, White, 10 ft", "Gutters", 12.60),
    ("GUT-K6-BRN", "K-style gutter 6in, Brown, 10 ft", "Gutters", 17.95),
    ("DSP-2X3-WHT", "Downspout 2x3in, White, 10 ft", "Gutters", 10.40),
    ("SID-VNL-D4", "Vinyl siding D4, Clay, square", "Siding", 129.00),
    ("SID-FC-LAP", "Fiber cement lap siding 8.25in x 12 ft", "Siding", 14.85),
    ("TRM-PVC-1X4", "PVC trim board 1x4 x 18 ft", "Siding", 27.30),
    ("WRB-HW-9", "House wrap 9 x 150 ft", "Siding", 158.00),
    ("INS-R13-15", "Fiberglass batt R-13, 15in, 40 sqft", "Insulation", 18.70),
    ("INS-R30-24", "Fiberglass batt R-30, 24in, 58 sqft", "Insulation", 48.25),
    ("WIN-DH-3046", "Vinyl double-hung window 30x46", "Windows", 214.00),
    ("WIN-SL-6040", "Vinyl slider window 60x40", "Windows", 289.00),
    ("DOR-STL-3068", "Steel entry door 36x80, primed", "Doors", 349.00),
    ("FLS-STP-4", "Step flashing 4x4x8, 100 pc", "Roofing", 41.60),
    ("VNT-RDG-4", "Ridge vent, 4 ft section", "Roofing", 15.90),
]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init(reset: bool = False) -> None:
    with connect() as conn:
        if reset:
            conn.executescript("DROP TABLE IF EXISTS skus; DROP TABLE IF EXISTS audit;")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS skus (
                sku TEXT PRIMARY KEY, description TEXT, category TEXT,
                price REAL, active INTEGER DEFAULT 1, updated_at TEXT, updated_by TEXT
            );
            CREATE TABLE IF NOT EXISTS audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, sku TEXT,
                old_price REAL, new_price REAL, reason TEXT, source TEXT
            );
            """
        )
        if conn.execute("SELECT COUNT(*) FROM skus").fetchone()[0] == 0:
            now = _now()
            conn.executemany(
                "INSERT INTO skus VALUES (?, ?, ?, ?, 1, ?, 'seed')",
                [(s, d, c, p, now) for s, d, c, p in CATALOG],
            )


def list_skus() -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM skus WHERE active = 1 ORDER BY sku")]


def get_sku(sku: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM skus WHERE sku = ?", (sku.strip().upper(),)).fetchone()
        return dict(row) if row else None


def set_price(sku: str, new_price: float, reason: str, source: str) -> dict:
    with connect() as conn:
        row = conn.execute("SELECT price FROM skus WHERE sku = ?", (sku,)).fetchone()
        if row is None:
            raise KeyError(sku)
        now = _now()
        conn.execute(
            "UPDATE skus SET price = ?, updated_at = ?, updated_by = ? WHERE sku = ?",
            (round(new_price, 2), now, source, sku),
        )
        conn.execute(
            "INSERT INTO audit (ts, sku, old_price, new_price, reason, source) VALUES (?, ?, ?, ?, ?, ?)",
            (now, sku, row["price"], round(new_price, 2), reason, source),
        )
    return {"sku": sku, "old_price": row["price"], "new_price": round(new_price, 2)}


def audit(since: str | None = None) -> list[dict]:
    with connect() as conn:
        q, args = "SELECT * FROM audit", ()
        if since:
            q, args = q + " WHERE ts >= ?", (since,)
        return [dict(r) for r in conn.execute(q + " ORDER BY id", args)]


def supplier_prices(on: date) -> list[dict]:
    """Today's supplier price file: every SKU, with a dated subset moved by 2-14%."""
    rng = random.Random(on.isoformat())
    movers = set(rng.sample([c[0] for c in CATALOG], k=8))
    rows = []
    for sku, desc, _cat, list_price in CATALOG:
        price = list_price
        if sku in movers:
            pct = rng.choice([-1, 1]) * rng.uniform(0.02, 0.14)
            price = list_price * (1 + pct)
        rows.append({"sku": sku, "description": desc, "supplier_price": round(price, 2), "effective_date": on.isoformat()})
    return rows


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    import sys

    init(reset="--reset" in sys.argv)
    print(f"{DB_PATH}: {len(list_skus())} skus")
