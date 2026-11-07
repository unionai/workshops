"""
A fake backend for developing the UI without a cluster.

Serves the same routes as api/app.py with synthetic data and a simulated run that
takes a few seconds, so the whole UI — progress stages, candidate grid, pose viewer,
leaderboard — can be exercised on a laptop with no Union connection and no GPUs.

    python scripts/mock_api.py
    cd ui && npm run dev          # proxies /api to localhost:8000

The molecules and the structure are real (a tiny PDB and valid SMILES); the scores
are random. Nothing here is imported by the deployed app.
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.targets import SEEDS, load_targets  # noqa: E402

app = FastAPI(title="BioNeMo Playground API (mock)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

TARGETS = load_targets()

# A real but minimal PDB fragment: enough for 3Dmol to draw something.
MOCK_PDB = """\
ATOM      1  N   MET A   1      -8.901   4.127  -0.555  1.00  0.00           N
ATOM      2  CA  MET A   1      -8.608   3.135  -1.618  1.00  0.00           C
ATOM      3  C   MET A   1      -7.117   2.964  -1.897  1.00  0.00           C
ATOM      4  O   MET A   1      -6.634   1.849  -1.758  1.00  0.00           O
ATOM      5  CB  MET A   1      -9.437   3.396  -2.889  1.00  0.00           C
ATOM      6  N   GLY A   2      -6.379   4.031  -2.228  1.00  0.00           N
ATOM      7  CA  GLY A   2      -4.923   3.983  -2.452  1.00  0.00           C
ATOM      8  C   GLY A   2      -4.136   4.146  -1.158  1.00  0.00           C
ATOM      9  O   GLY A   2      -4.667   4.607  -0.147  1.00  0.00           O
ATOM     10  N   ALA A   3      -2.872   3.759  -1.200  1.00  0.00           N
ATOM     11  CA  ALA A   3      -1.999   3.863  -0.032  1.00  0.00           C
ATOM     12  C   ALA A   3      -0.624   3.290  -0.349  1.00  0.00           C
ATOM     13  O   ALA A   3      -0.455   2.249  -0.985  1.00  0.00           O
ATOM     14  CB  ALA A   3      -1.859   5.317   0.415  1.00  0.00           C
END
"""

MOCK_POSE = """\
  Mrv  mock

  3  2  0  0  0  0            999 V2000
   -1.2000    0.3000    0.0000 C   0  0
    0.0000   -0.4000    0.0000 C   0  0
    1.2000    0.3000    0.0000 O   0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
"""

# Valid SMILES so the 2D depiction is real.
MOCK_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1",
    "COc1cc2c(cc1OC)C(=O)C(C)N2",
    "c1ccc2[nH]ccc2c1",
    "CCN(CC)CCNC(=O)c1ccc(N)cc1",
    "OC(=O)c1ccccc1O",
    "CN1CCC[C@H]1c1cccnc1",
    "Cc1ccc(cc1)S(=O)(=O)N",
    "CC1=CC(=O)C=CC1=O",
    "NC(=O)c1ccncc1",
]

_RUNS: dict[str, dict] = {}
_BOARD: list[dict] = []


class ScreenRequest(BaseModel):
    target_key: str
    seed_smiles: str
    nickname: str = "anonymous"
    num_molecules: int = 10
    temperature: float = 1.0
    noise: float = 0.2
    step_size: int = 4
    scoring: str = "QED"
    num_poses: int = 10
    fold_live: bool = False


@app.get("/health")
async def health():
    return {"status": "ok", "mock": True}


@app.get("/api/catalogue")
async def catalogue():
    return {
        "targets": [
            {
                "key": t.key,
                "label": t.label,
                "blurb": t.blurb,
                "uniprot": t.uniprot,
                "length": t.length,
                "source": t.source,
                # Pretend everything but the last target is precomputed, so the UI's
                # "must fold live" path is exercised too.
                "has_structure": t.key != "trypsin",
            }
            for t in TARGETS.values()
        ],
        "seeds": [{"key": s.key, "label": s.label, "smiles": s.smiles, "blurb": s.blurb} for s in SEEDS],
        "defaults": {
            "num_molecules": 10,
            "temperature": 1.0,
            "noise": 0.2,
            "step_size": 4,
            "scoring": "QED",
            "num_poses": 10,
        },
    }


@app.get("/api/structure/{target_key}")
async def structure(target_key: str):
    return {"target_key": target_key, "pdb": MOCK_PDB}


@app.post("/api/screen")
async def launch(req: ScreenRequest):
    target = TARGETS.get(req.target_key)
    if target is None:
        raise HTTPException(404, "unknown target")

    run_name = f"mock-{int(time.time() * 1000) % 1_000_000}"
    molecules = random.sample(MOCK_SMILES, min(req.num_molecules, len(MOCK_SMILES)))

    candidates = []
    for smiles in molecules:
        docked = random.random() > 0.15
        candidates.append(
            {
                "smiles": smiles,
                "genmol_score": round(random.uniform(0.3, 0.95), 4),
                "pose_confidence": round(random.uniform(-0.5, 0.95), 4) if docked else None,
                "docking_status": "success" if docked else "failed",
                "pose_count": req.num_poses if docked else 0,
                "molecular_weight": round(random.uniform(150, 450), 2),
                "logp": round(random.uniform(-1, 5), 2),
                "qed": round(random.uniform(0.2, 0.95), 4),
                "num_heavy_atoms": random.randint(12, 34),
                "num_rotatable_bonds": random.randint(0, 8),
                "ring_count": random.randint(1, 4),
            }
        )

    candidates.sort(
        key=lambda c: (c["pose_confidence"] is not None, c["pose_confidence"] or -99),
        reverse=True,
    )

    stages = {"generate": round(random.uniform(1.5, 4.0), 2), "dock": round(random.uniform(14, 32), 2)}
    if req.fold_live:
        stages = {"fold": round(random.uniform(120, 260), 2), **stages}

    _RUNS[run_name] = {
        "nickname": req.nickname,
        "target_key": target.key,
        "target_label": target.label,
        "seed_smiles": req.seed_smiles,
        "candidates": candidates,
        "timings": stages,
        "folded_in_run": req.fold_live,
        "started": time.time(),
        # Simulated wall clock, compressed so UI iteration isn't tedious.
        "duration": 7.0 if req.fold_live else 4.0,
    }

    return {
        "run_name": run_name,
        "run_url": f"https://your-org.hosted.unionai.cloud/v2/runs/{run_name}",
        "target_key": target.key,
        "target_label": target.label,
        "mode": "full_pipeline" if req.fold_live else "screen",
    }


@app.get("/api/run/{run_name}")
async def run_status(run_name: str):
    state = _RUNS.get(run_name)
    if state is None:
        raise HTTPException(404, "unknown run")

    elapsed = time.time() - state["started"]
    url = f"https://your-org.hosted.unionai.cloud/v2/runs/{run_name}"

    if elapsed < state["duration"]:
        # Reveal stage timings progressively so the progress component animates.
        keys = list(state["timings"])
        revealed = int(len(keys) * (elapsed / state["duration"]))
        return {
            "run_name": run_name,
            "phase": "RUNNING",
            "url": url,
            "done": False,
            "timings": {k: state["timings"][k] for k in keys[:revealed]},
        }

    if not any(e["run_name"] == run_name for e in _BOARD):
        for cand in [c for c in state["candidates"] if c["pose_confidence"] is not None][:3]:
            _BOARD.append(
                {
                    "nickname": state["nickname"],
                    "target_key": state["target_key"],
                    "target_label": state["target_label"],
                    "smiles": cand["smiles"],
                    "pose_conf": cand["pose_confidence"],
                    "qed": cand["qed"],
                    "mol_weight": cand["molecular_weight"],
                    "logp": cand["logp"],
                    "run_url": url,
                    "run_name": run_name,
                    "created_at": "now",
                    "best_pose": None,
                }
            )

    return {
        "run_name": run_name,
        "phase": "SUCCEEDED",
        "url": url,
        "done": True,
        "target_key": state["target_key"],
        "target_label": state["target_label"],
        "seed_smiles": state["seed_smiles"],
        "timings": state["timings"],
        "folded_in_run": state["folded_in_run"],
        "candidates": state["candidates"],
    }


@app.get("/api/run/{run_name}/pose/{index}")
async def run_pose(run_name: str, index: int):
    state = _RUNS.get(run_name)
    if state is None or not 0 <= index < len(state["candidates"]):
        raise HTTPException(404, "no such pose")
    cand = state["candidates"][index]
    await asyncio.sleep(0.2)
    docked = cand["pose_confidence"] is not None
    n = cand["pose_count"] or 0
    # Mirror the real API: every pose, best first, with matching confidences, so the
    # pose-ranking list in the UI can be exercised without a cluster.
    poses = [MOCK_POSE] * n if docked else []
    confidences = [round(cand["pose_confidence"] - i * random.uniform(0.2, 0.6), 4) for i in range(n)] if docked else []
    return {
        "smiles": cand["smiles"],
        "pose": poses[0] if poses else None,
        "pose_confidence": cand["pose_confidence"],
        "poses": poses,
        "confidences": confidences,
        "protein_pdb": MOCK_PDB,
    }


@app.get("/api/leaderboard")
async def leaderboard(target_key: str | None = None, limit: int = 20):
    rows = [e for e in _BOARD if target_key is None or e["target_key"] == target_key]
    rows.sort(key=lambda e: e["pose_conf"], reverse=True)
    rows = rows[:limit]
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    confs = [e["pose_conf"] for e in _BOARD]
    return {
        "entries": rows,
        "stats": {
            "runs": len({e["run_name"] for e in _BOARD}),
            "molecules": len(_BOARD),
            "players": len({e["nickname"] for e in _BOARD}),
            "best_pose_confidence": max(confs) if confs else None,
        },
    }


@app.get("/api/nims")
async def nims():
    return {
        "nims": [
            {"name": "nim-genmol", "status": "ready", "detail": "HTTP 200"},
            {"name": "nim-diffdock", "status": "ready", "detail": "HTTP 200"},
            {"name": "nim-msa-search", "status": "unreachable", "detail": "scaled to zero"},
            {"name": "nim-openfold2", "status": "unreachable", "detail": "scaled to zero"},
        ]
    }


if __name__ == "__main__":
    print("Mock API on http://localhost:8000 — run `cd ui && npm run dev` alongside it.")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
