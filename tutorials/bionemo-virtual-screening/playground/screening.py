"""
The generative virtual screening pipeline, as Flyte 2 tasks.

Two entry points:

``screen``
    The booth loop. Generate molecules with GenMol, batch-dock them with DiffDock
    against a *precomputed* structure, score, and return a ranked table. Both NIMs
    are already warm, so this is tens of seconds.

``full_pipeline``
    The complete blueprint DAG: MSA-Search -> OpenFold2 -> GenMol -> DiffDock.
    Folding is the slow part (minutes), so this is the "yes, we do the whole thing"
    run you start and come back to, not the one someone waits for.

``fold_target``
    Just the folding half, cached. Used by ``scripts/precompute_structures.py`` to
    fill data/structures/ before the conference.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import flyte
import flyte.report

from playground import clients, config
from playground.nims import diffdock_env, genmol_env, msa_env, nim_url, openfold2_env

# ---------------------------------------------------------------------------
# Images
#
# The pipeline tasks are thin HTTP clients plus RDKit for descriptors. None of the
# model weights live here; all the heavy lifting happens in the NIM apps.
# ---------------------------------------------------------------------------

# Local modules under playground/ travel with the run's code bundle, so the image
# only needs third-party dependencies.
pipeline_image = flyte.Image.from_debian_base(python_version=(3, 12), name="gvs-pipeline").with_pip_packages(
    "httpx", "rdkit"
)

# Booth-time environment. Depends on the two warm NIMs so that deploying the
# pipeline deploys its dependencies too, and so `.endpoint` resolves.
screen_env = flyte.TaskEnvironment(
    name="gvs-screen",
    image=pipeline_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[genmol_env, diffdock_env],
)

# Folding environment, which additionally needs the two scale-to-zero NIMs.
fold_env = flyte.TaskEnvironment(
    name="gvs-fold",
    image=pipeline_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    depends_on=[msa_env, openfold2_env],
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One generated molecule with everything we know about it."""

    smiles: str
    # GenMol's optimization score for the requested property (QED by default).
    genmol_score: float
    # DiffDock confidence for the best pose. Pose-correctness, NOT binding affinity.
    pose_confidence: float | None = None
    docking_status: str = "not docked"
    # Best pose geometry, SDF/MOL-block text, for the 3D viewer.
    best_pose: str | None = field(default=None, repr=False)
    pose_count: int = 0
    # Every pose DiffDock returned, best first, with the matching confidences. Kept so
    # the UI can offer the pose-by-pose ranking the NVIDIA blueprint page shows. Bulky,
    # so the API strips these from poll payloads and serves them on demand.
    poses: list[str] = field(default_factory=list, repr=False)
    confidences: list[float] = field(default_factory=list, repr=False)

    # RDKit descriptors, computed locally. Cheap, and they make the leaderboard
    # mean something chemically rather than being a single opaque number.
    molecular_weight: float | None = None
    logp: float | None = None
    qed: float | None = None
    num_heavy_atoms: int | None = None
    num_rotatable_bonds: int | None = None
    ring_count: int | None = None

    @property
    def docked(self) -> bool:
        return self.docking_status == "success" and self.pose_confidence is not None


@dataclass
class ScreenResult:
    """Everything the UI needs to render one screening run."""

    target_key: str
    target_label: str
    seed_smiles: str
    protein_pdb: str = field(repr=False, default="")
    candidates: list[Candidate] = field(default_factory=list)
    # Wall-clock seconds per stage, so the booth UI can show where time went.
    timings: dict[str, float] = field(default_factory=dict)
    folded_in_run: bool = False


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------


def _describe(smiles: str) -> dict[str, float | int | None]:
    """
    Compute RDKit descriptors for a SMILES string.

    Returns all-None on an unparseable molecule rather than raising: GenMol can emit
    a string RDKit won't sanitize, and losing one row of the leaderboard is much
    better than losing the run.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import QED, Crippen, Descriptors

    RDLogger.DisableLog("rdApp.*")

    empty: dict[str, float | int | None] = {
        "molecular_weight": None,
        "logp": None,
        "qed": None,
        "num_heavy_atoms": None,
        "num_rotatable_bonds": None,
        "ring_count": None,
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty

    try:
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "qed": round(QED.qed(mol), 4),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "ring_count": mol.GetRingInfo().NumRings(),
        }
    except Exception:  # noqa: BLE001 - a bad molecule must not fail the run
        return empty


# ---------------------------------------------------------------------------
# Folding
# ---------------------------------------------------------------------------


@fold_env.task(cache="auto", retries=2)
async def fold_target(sequence: str) -> str:
    """
    Fold a protein sequence: MSA-Search, then OpenFold2. Returns PDB text.

    Cached on the sequence, so refolding the same target is free. This is what makes
    precomputing the demo structures a one-time cost.
    """
    await clients.wait_until_ready(nim_url(msa_env), timeout=config.TIMEOUT_MSA)
    alignments = await clients.msa_search(nim_url(msa_env), sequence)

    await clients.wait_until_ready(nim_url(openfold2_env), timeout=config.TIMEOUT_OPENFOLD2)
    return await clients.openfold2_predict(nim_url(openfold2_env), sequence, alignments)


# ---------------------------------------------------------------------------
# Generation + docking
# ---------------------------------------------------------------------------


@screen_env.task(retries=2)
async def generate_molecules(
    seed_smiles: str,
    num_molecules: int = config.DEFAULT_NUM_MOLECULES,
    temperature: float = config.DEFAULT_TEMPERATURE,
    noise: float = config.DEFAULT_NOISE,
    step_size: int = config.DEFAULT_STEP_SIZE,
    scoring: str = config.DEFAULT_SCORING,
) -> list[Candidate]:
    """Generate molecules with GenMol and attach local RDKit descriptors."""
    await clients.wait_until_ready(nim_url(genmol_env), timeout=config.TIMEOUT_GENMOL)

    generated = await clients.genmol_generate(
        nim_url(genmol_env),
        seed_smiles,
        num_molecules=num_molecules,
        temperature=temperature,
        noise=noise,
        step_size=step_size,
        scoring=scoring,
    )

    return [
        Candidate(smiles=mol.smiles, genmol_score=mol.score, **_describe(mol.smiles))  # type: ignore[arg-type]
        for mol in generated
    ]


@screen_env.task(retries=2)
async def dock_candidates(
    protein_pdb: str,
    candidates: list[Candidate],
    num_poses: int = config.DEFAULT_NUM_POSES,
    time_divisions: int = config.DEFAULT_TIME_DIVISIONS,
    num_steps: int = config.DEFAULT_NUM_STEPS,
) -> list[Candidate]:
    """
    Batch-dock every candidate against the receptor and fold the results back in.

    DiffDock returns results positionally, so we match on the returned SMILES where
    we can and fall back to index. A mismatch would silently attach the wrong pose to
    the wrong molecule, which is the kind of bug that survives a demo and embarrasses
    you later.
    """
    if not candidates:
        return []

    await clients.wait_until_ready(nim_url(diffdock_env), timeout=config.TIMEOUT_DIFFDOCK)

    docked = await clients.diffdock_generate(
        nim_url(diffdock_env),
        protein_pdb,
        [c.smiles for c in candidates],
        num_poses=num_poses,
        time_divisions=time_divisions,
        num_steps=num_steps,
    )

    by_smiles = {d.smiles: d for d in docked}

    updated: list[Candidate] = []
    for idx, cand in enumerate(candidates):
        result = by_smiles.get(cand.smiles)
        if result is None and idx < len(docked):
            result = docked[idx]

        if result is None:
            updated.append(cand)
            continue

        cand.docking_status = result.status
        cand.pose_confidence = result.best_confidence
        cand.best_pose = result.best_pose
        cand.pose_count = len(result.poses)
        cand.poses = result.poses
        cand.confidences = result.confidences
        updated.append(cand)

    return updated


def _rank(candidates: list[Candidate]) -> list[Candidate]:
    """
    Rank candidates: successfully docked first, then by pose confidence, then by QED.

    Deliberately simple and explainable. A visitor asking "why is that one on top?"
    gets a one-sentence answer, which matters more at a booth than a cleverer score.
    """
    return sorted(
        candidates,
        key=lambda c: (
            c.docked,
            c.pose_confidence if c.pose_confidence is not None else float("-inf"),
            c.qed if c.qed is not None else 0.0,
        ),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


@screen_env.task(report=True)
async def screen(
    target_key: str,
    target_label: str,
    protein_pdb: str,
    seed_smiles: str,
    num_molecules: int = config.DEFAULT_NUM_MOLECULES,
    temperature: float = config.DEFAULT_TEMPERATURE,
    noise: float = config.DEFAULT_NOISE,
    step_size: int = config.DEFAULT_STEP_SIZE,
    scoring: str = config.DEFAULT_SCORING,
    num_poses: int = config.DEFAULT_NUM_POSES,
) -> ScreenResult:
    """
    The booth loop: generate, dock, rank. Takes a precomputed receptor structure.

    Kept as a single task with two awaits rather than a fan-out, because with one warm
    replica of each NIM there is nothing to gain from parallelism here and a flat run
    graph is easier to narrate to someone watching over your shoulder.
    """
    loop = asyncio.get_event_loop()

    t0 = loop.time()
    candidates = await generate_molecules(
        seed_smiles,
        num_molecules=num_molecules,
        temperature=temperature,
        noise=noise,
        step_size=step_size,
        scoring=scoring,
    )
    t_gen = loop.time() - t0

    t0 = loop.time()
    candidates = await dock_candidates(protein_pdb, candidates, num_poses=num_poses)
    t_dock = loop.time() - t0

    ranked = _rank(candidates)
    result = ScreenResult(
        target_key=target_key,
        target_label=target_label,
        seed_smiles=seed_smiles,
        protein_pdb=protein_pdb,
        candidates=ranked,
        timings={"generate": round(t_gen, 2), "dock": round(t_dock, 2)},
    )

    await _write_report(result)
    return result


@fold_env.task(report=True)
async def full_pipeline(
    target_key: str,
    target_label: str,
    sequence: str,
    seed_smiles: str,
    num_molecules: int = config.DEFAULT_NUM_MOLECULES,
    num_poses: int = config.DEFAULT_NUM_POSES,
) -> ScreenResult:
    """
    The complete blueprint: MSA -> OpenFold2 -> GenMol -> DiffDock.

    This is the full DAG for when someone wants to see the whole thing rather than the
    fast path. Folding dominates the runtime.
    """
    loop = asyncio.get_event_loop()

    t0 = loop.time()
    protein_pdb = await fold_target(sequence)
    t_fold = loop.time() - t0

    result = await screen(
        target_key=target_key,
        target_label=target_label,
        protein_pdb=protein_pdb,
        seed_smiles=seed_smiles,
        num_molecules=num_molecules,
        num_poses=num_poses,
    )
    result.timings["fold"] = round(t_fold, 2)
    result.folded_in_run = True
    return result


# ---------------------------------------------------------------------------
# Flyte report
#
# A readable summary inside the Union run page, so the run itself is worth showing
# and not just the Vue app in front of it.
# ---------------------------------------------------------------------------


async def _write_report(result: ScreenResult) -> None:
    def cell(value: object) -> str:
        return "&mdash;" if value is None else str(value)

    rows = "".join(
        f"<tr>"
        f"<td style='text-align:right'>{i}</td>"
        f"<td style='font-family:ui-monospace,monospace;font-size:12px'>{c.smiles}</td>"
        f"<td style='text-align:right'>{cell(c.pose_confidence)}</td>"
        f"<td style='text-align:right'>{cell(c.qed)}</td>"
        f"<td style='text-align:right'>{cell(c.molecular_weight)}</td>"
        f"<td style='text-align:right'>{cell(c.logp)}</td>"
        f"<td>{c.docking_status}</td>"
        f"</tr>"
        for i, c in enumerate(result.candidates, start=1)
    )

    timing_bits = " &middot; ".join(f"{k}: {v}s" for k, v in result.timings.items())

    html = f"""
    <div style="font-family:system-ui,sans-serif;max-width:1000px">
      <h2 style="margin-bottom:4px">{result.target_label}</h2>
      <p style="color:#555;margin-top:0">
        Seed <code>{result.seed_smiles}</code><br>
        {len(result.candidates)} candidates &middot; {timing_bits}
      </p>
      <p style="background:#fff8e1;border-left:3px solid #f0b400;padding:8px 12px;color:#444">
        <strong>Pose confidence</strong> is DiffDock's estimate that the predicted geometry
        is correct. It is not a binding affinity.
      </p>
      <table style="border-collapse:collapse;width:100%;font-size:13px">
        <thead>
          <tr style="background:#f4f4f5;text-align:left">
            <th style="padding:6px">#</th><th style="padding:6px">SMILES</th>
            <th style="padding:6px">Pose conf.</th><th style="padding:6px">QED</th>
            <th style="padding:6px">MW</th><th style="padding:6px">logP</th>
            <th style="padding:6px">Status</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """

    # `replace` is @syncify'd, so the async form is `.aio`.
    await flyte.report.replace.aio(html, do_flush=True)
