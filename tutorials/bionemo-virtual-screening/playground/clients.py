"""
Typed clients for the four BioNeMo NIMs.

Request paths and payloads are copied from the upstream blueprint notebook so that
results can be diffed against NVIDIA's reference run. The only additions are
timeouts, readiness polling, and structured return types.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from playground import config


class NimError(RuntimeError):
    """A NIM returned something we can't use."""


async def wait_until_ready(endpoint: str, *, timeout: float = 600.0, interval: float = 3.0) -> None:
    """
    Poll a NIM's readiness probe until it reports ready.

    A NIM that is still loading weights returns a non-200 from /v1/health/ready rather
    than queueing the request, so without this the first call after a cold start fails
    instead of waiting.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    url = f"{endpoint.rstrip('/')}/v1/health/ready"
    last: str = "no attempt made"

    async with httpx.AsyncClient(timeout=30.0) as client:
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
                last = f"HTTP {resp.status_code}"
            except httpx.RequestError as exc:
                last = f"{type(exc).__name__}: {exc}"
            await asyncio.sleep(interval)

    raise NimError(f"NIM at {endpoint} not ready after {timeout:.0f}s (last: {last})")


async def _post(endpoint: str, path: str, payload: dict, timeout: float) -> dict:
    url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code != 200:
            raise NimError(f"{url} returned HTTP {resp.status_code}: {resp.text[:500]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise NimError(f"{url} returned non-JSON: {resp.text[:200]!r}") from exc


# ---------------------------------------------------------------------------
# MSA-Search (MMseqs2)
# ---------------------------------------------------------------------------


async def msa_search(endpoint: str, sequence: str) -> dict:
    """
    Run multiple sequence alignment for a protein sequence.

    Returns the raw ``alignments`` mapping, which is passed straight to OpenFold2.
    This is the slow step: minutes, and it wants ~1.3 TB of databases on disk.
    """
    payload = {
        "sequence": sequence,
        "e_value": config.MSA_E_VALUE,
        "iterations": config.MSA_ITERATIONS,
        "search_type": config.MSA_SEARCH_TYPE,
        "output_alignment_formats": ["fasta", "a3m"],
        "databases": config.MSA_DATABASES,
    }
    result = await _post(endpoint, "biology/colabfold/msa-search/predict", payload, config.TIMEOUT_MSA)
    if "alignments" not in result:
        raise NimError(f"MSA response has no 'alignments' key; got {sorted(result)}")
    return result["alignments"]


# ---------------------------------------------------------------------------
# OpenFold2
# ---------------------------------------------------------------------------


async def openfold2_predict(endpoint: str, sequence: str, alignments: dict) -> str:
    """Fold a sequence given its alignments. Returns the top-ranked structure as PDB text."""
    payload = {
        "sequence": sequence,
        "use_templates": False,
        "relaxed_prediction": False,
        "alignments": alignments,
    }
    result = await _post(
        endpoint,
        "biology/openfold/openfold2/predict-structure-from-msa-and-template",
        payload,
        config.TIMEOUT_OPENFOLD2,
    )

    ranked = result.get("structures_in_ranked_order")
    if not ranked:
        raise NimError(f"OpenFold2 returned no structures; keys were {sorted(result)}")
    return ranked[0]["structure"]


# ---------------------------------------------------------------------------
# GenMol
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratedMolecule:
    smiles: str
    score: float


async def genmol_generate(
    endpoint: str,
    seed_smiles: str,
    *,
    num_molecules: int = config.DEFAULT_NUM_MOLECULES,
    temperature: float = config.DEFAULT_TEMPERATURE,
    noise: float = config.DEFAULT_NOISE,
    step_size: int = config.DEFAULT_STEP_SIZE,
    scoring: str = config.DEFAULT_SCORING,
) -> list[GeneratedMolecule]:
    """
    Generate molecules from a SAFE-format seed.

    ``seed_smiles`` carries a ``[*{m-n}]`` mask telling GenMol how many atoms it may
    add. ``scoring`` selects the property being optimized, e.g. "QED" for
    drug-likeness.
    """
    payload = {
        "smiles": seed_smiles,
        "num_molecules": num_molecules,
        "temperature": temperature,
        "noise": noise,
        "step_size": step_size,
        "scoring": scoring,
    }
    result = await _post(endpoint, "generate", payload, config.TIMEOUT_GENMOL)

    molecules = result.get("molecules")
    if not molecules:
        raise NimError(f"GenMol returned no molecules; keys were {sorted(result)}")

    return [GeneratedMolecule(smiles=m["smiles"], score=float(m["score"])) for m in molecules]


# ---------------------------------------------------------------------------
# DiffDock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockedLigand:
    """One ligand's docking result, with poses ordered by descending confidence."""

    smiles: str
    status: str
    # Pose geometries as SDF / MOL-block text, best first.
    poses: list[str]
    # DiffDock's confidence-model score per pose, aligned with ``poses``.
    #
    # IMPORTANT, and worth saying out loud at the booth: this scores how likely the
    # predicted *geometry* is to be correct. It is not a binding affinity and not a
    # measure of how tightly the ligand binds. DiffDock does not predict affinity at
    # all. Any UI that ranks on this must say "pose confidence", never "best binder".
    confidences: list[float]

    @property
    def ok(self) -> bool:
        return self.status == "success" and bool(self.poses)

    @property
    def best_confidence(self) -> float | None:
        return self.confidences[0] if self.confidences else None

    @property
    def best_pose(self) -> str | None:
        return self.poses[0] if self.poses else None


async def diffdock_generate(
    endpoint: str,
    protein_pdb: str,
    ligand_smiles: list[str],
    *,
    num_poses: int = config.DEFAULT_NUM_POSES,
    time_divisions: int = config.DEFAULT_TIME_DIVISIONS,
    num_steps: int = config.DEFAULT_NUM_STEPS,
) -> list[DockedLigand]:
    """
    Blind-dock a batch of ligands against one receptor.

    DiffDock v2 takes the whole batch in a single request, which is substantially
    faster than one request per ligand and is the reason the booth loop fits in
    tens of seconds.
    """
    payload = {
        "protein": protein_pdb,
        "ligand": "\n".join(ligand_smiles),
        "ligand_file_type": "txt",
        "num_poses": num_poses,
        "time_divisions": time_divisions,
        "num_steps": num_steps,
    }
    result = await _post(endpoint, "molecular-docking/diffdock/generate", payload, config.TIMEOUT_DIFFDOCK)

    returned = [s for s in result.get("ligand", "").split("\n") if s.strip()]

    # The confidence field was renamed between DiffDock NIM versions: the 2.x API
    # documents `position_confidence`, while older references use `pose_confidence`.
    # Accept either so a NIM image bump doesn't silently flatten the leaderboard.
    scores = result.get("position_confidence")
    if scores is None:
        scores = result.get("pose_confidence", [])

    statuses = result.get("status", [])
    positions = result.get("ligand_positions", [])

    # DiffDock changes the *shape* of its response depending on how many ligands you
    # sent, which is the kind of thing that works in testing and breaks in production.
    #
    #   one ligand : status is a str, ligand_positions is a flat list of poses,
    #                position_confidence is a flat list of floats
    #   many       : status is a list, ligand_positions is a list per ligand,
    #                position_confidence is a list per ligand
    #
    # Normalise the single-ligand case up into the batch shape so the loop below has
    # one thing to handle. Detected on `status` being a str, which is unambiguous.
    if isinstance(statuses, str):
        statuses = [statuses]
        positions = [positions]
        scores = [scores]

    docked: list[DockedLigand] = []
    for idx, smiles in enumerate(returned):
        status = statuses[idx] if idx < len(statuses) else "unknown"

        raw_poses = positions[idx] if idx < len(positions) else []
        poses = [p for p in (raw_poses if isinstance(raw_poses, list) else [raw_poses]) if p]

        raw_scores = scores[idx] if idx < len(scores) else []
        if not isinstance(raw_scores, list):
            raw_scores = [raw_scores]
        # -1000.0 is DiffDock's sentinel for "this did not dock". Keeping it would put
        # a meaningless -1000 at the bottom of the leaderboard as though it were a
        # measurement; drop it so the candidate reads as undocked instead.
        confidences = [float(c) for c in raw_scores if c is not None and float(c) > -999.0]

        docked.append(
            DockedLigand(
                smiles=smiles,
                status=status,
                poses=poses,
                confidences=confidences,
            )
        )

    return docked
