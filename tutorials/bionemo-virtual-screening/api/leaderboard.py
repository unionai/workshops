"""
The booth leaderboard.

A visitor's run should still be on the board when they walk past again an hour later,
and ideally on day two as well. That means it has to outlive both the request and the
process, so it lives in SQLite rather than in memory.

Durability caveat, stated plainly: the database file sits on the app container's
filesystem, so it survives restarts of the server process but not a pod replacement.
For a two-day booth with a pinned single replica that is the right trade for the
complexity it saves. ``LEADERBOARD_PATH`` can point at a mounted volume if you want
it to genuinely survive anything.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

LEADERBOARD_PATH = Path(os.environ.get("LEADERBOARD_PATH", "/tmp/bionemo-leaderboard.sqlite"))

# Where the board is mirrored so it outlives the container.
#
# The local SQLite file lives on the app pod's filesystem, which is gone the moment the
# pod is replaced — and a redeploy replaces it. For a two-day booth that would mean
# losing every visitor's entry on any config change. So the file is copied to blob
# storage after each write and restored at startup.
#
# The key is deliberately fixed rather than derived from the deployment, because the
# whole point is to survive a new revision. Set BIONEMO_LEADERBOARD_URI to override.
LEADERBOARD_URI = os.environ.get("BIONEMO_LEADERBOARD_URI") or None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    run_name      TEXT    NOT NULL,
    run_url       TEXT    NOT NULL,
    nickname      TEXT    NOT NULL,
    target_key    TEXT    NOT NULL,
    target_label  TEXT    NOT NULL,
    seed_smiles   TEXT    NOT NULL,
    smiles        TEXT    NOT NULL,
    pose_conf     REAL,
    qed           REAL,
    mol_weight    REAL,
    logp          REAL,
    payload       TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_entries_target_conf
    ON entries (target_key, pose_conf DESC);

CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_run_smiles
    ON entries (run_name, smiles);
"""

_lock = threading.Lock()


@contextmanager
def _conn():
    """
    Serialized connection to the leaderboard.

    SQLite plus a process-wide lock is plenty for booth traffic — a handful of writes
    a minute — and removes any question of concurrent-write corruption.
    """
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        conn = sqlite3.connect(LEADERBOARD_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def resolve_uri(raw_data_path: str | None) -> str | None:
    """
    Work out a stable blob URI for the board.

    `raw_data_path` is the app's writable prefix, but it is scoped to the deployment,
    so writing the board there would lose it on the next revision — exactly the case
    this exists to survive. Instead take the bucket from it and use a fixed key
    alongside, which stays put across redeploys.
    """
    global LEADERBOARD_URI
    if LEADERBOARD_URI:
        return LEADERBOARD_URI
    if not raw_data_path or "://" not in raw_data_path:
        return None
    scheme, rest = raw_data_path.split("://", 1)
    bucket = rest.split("/", 1)[0]
    LEADERBOARD_URI = f"{scheme}://{bucket}/bionemo-playground/leaderboard.sqlite"
    return LEADERBOARD_URI


async def restore() -> bool:
    """
    Pull the board back from blob storage, if one was ever saved.

    Returns True when a board was restored. Never raises: starting a booth demo with an
    empty leaderboard is a bad day, but refusing to start at all is a worse one.
    """
    if not LEADERBOARD_URI:
        return False
    try:
        import flyte.storage as storage

        if not await storage.exists(LEADERBOARD_URI):
            logger.info("No saved leaderboard at %s — starting fresh.", LEADERBOARD_URI)
            return False
        LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
        await storage.get(LEADERBOARD_URI, str(LEADERBOARD_PATH))
        logger.info("Restored leaderboard from %s", LEADERBOARD_URI)
        return True
    except Exception:
        logger.exception("Could not restore leaderboard from %s", LEADERBOARD_URI)
        return False


async def persist() -> None:
    """
    Mirror the board to blob storage.

    Whole-file upload after each write. Crude, but the file is tens of kilobytes and a
    booth produces a few writes a minute, so the simplicity is worth more than the
    bytes. Failures are logged and swallowed — a demo that keeps working with a stale
    backup beats one that 500s because S3 was briefly unhappy.
    """
    if not LEADERBOARD_URI or not LEADERBOARD_PATH.exists():
        return
    try:
        import flyte.storage as storage

        await storage.put(str(LEADERBOARD_PATH), LEADERBOARD_URI)
    except Exception:
        logger.exception("Could not persist leaderboard to %s", LEADERBOARD_URI)


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


@dataclass
class Entry:
    rank: int
    nickname: str
    target_key: str
    target_label: str
    smiles: str
    pose_conf: float | None
    qed: float | None
    mol_weight: float | None
    logp: float | None
    run_url: str
    run_name: str
    created_at: str
    best_pose: str | None = field(default=None, repr=False)


def record_run(
    *,
    run_name: str,
    run_url: str,
    nickname: str,
    target_key: str,
    target_label: str,
    seed_smiles: str,
    candidates: list[dict],
    top_n: int = 3,
) -> int:
    """
    Add a run's best candidates to the board.

    Only the top ``top_n`` docked candidates are recorded. Putting all ten on the board
    would let one visitor flood it, and nobody reads past the first screen anyway.

    Returns the number of rows inserted. Re-recording the same run is a no-op thanks to
    the unique index, so a retried poll can't duplicate entries.
    """
    docked = [c for c in candidates if c.get("docking_status") == "success" and c.get("pose_confidence") is not None]
    docked.sort(key=lambda c: c["pose_confidence"], reverse=True)
    best = docked[:top_n]

    if not best:
        return 0

    rows = [
        (
            run_name,
            run_url,
            nickname,
            target_key,
            target_label,
            seed_smiles,
            c["smiles"],
            c.get("pose_confidence"),
            c.get("qed"),
            c.get("molecular_weight"),
            c.get("logp"),
            json.dumps(c),
        )
        for c in best
    ]

    with _conn() as conn:
        cur = conn.executemany(
            """
            INSERT OR IGNORE INTO entries (
                run_name, run_url, nickname, target_key, target_label, seed_smiles,
                smiles, pose_conf, qed, mol_weight, logp, payload
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return cur.rowcount or 0


def top_entries(target_key: str | None = None, limit: int = 20) -> list[Entry]:
    """Read the board, best pose confidence first."""
    sql = """
        SELECT * FROM entries
        {where}
        ORDER BY pose_conf DESC, qed DESC
        LIMIT ?
    """.format(where="WHERE target_key = ?" if target_key else "")

    params: tuple = (target_key, limit) if target_key else (limit,)

    with _conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    entries: list[Entry] = []
    for i, row in enumerate(rows, start=1):
        payload = json.loads(row["payload"])
        entries.append(
            Entry(
                rank=i,
                nickname=row["nickname"],
                target_key=row["target_key"],
                target_label=row["target_label"],
                smiles=row["smiles"],
                pose_conf=row["pose_conf"],
                qed=row["qed"],
                mol_weight=row["mol_weight"],
                logp=row["logp"],
                run_url=row["run_url"],
                run_name=row["run_name"],
                created_at=row["created_at"],
                best_pose=payload.get("best_pose"),
            )
        )
    return entries


def stats() -> dict:
    """Headline numbers for the booth screen."""
    with _conn() as conn:
        row = conn.execute(
            """
            SELECT COUNT(DISTINCT run_name) AS runs,
                   COUNT(*)                 AS molecules,
                   COUNT(DISTINCT nickname) AS players,
                   MAX(pose_conf)           AS best
            FROM entries
            """
        ).fetchone()

    return {
        "runs": row["runs"] or 0,
        "molecules": row["molecules"] or 0,
        "players": row["players"] or 0,
        "best_pose_confidence": row["best"],
    }
