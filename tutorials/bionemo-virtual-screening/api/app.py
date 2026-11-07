"""
The playground's HTTP API, served as a Union app.

It does three things: hand the UI the target/seed catalogue, launch screening runs on
Flyte and report their progress, and keep the leaderboard.

Deploy with::

    flyte deploy api/app.py api_env
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import pathlib
from contextlib import asynccontextmanager

import flyte
import flyte.app
import flyte.errors
import flyte.remote as remote
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from flyte.app.extras import FastAPIAppEnvironment
from pydantic import BaseModel, Field

from api import leaderboard
from playground import config
from playground.screening import ScreenResult, full_pipeline, screen, screen_env
from playground.targets import SEEDS, load_targets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _bundled_files() -> tuple[str, ...]:
    """
    Every non-Python file this app needs, as explicit individual paths.

    The code bundle ships Python only, so data files must be named in `include`. The
    documented shorthands do not work reliably here: passing a *directory* bundled
    nothing at all, and a `**/*` glob bundled the nested `assets/` subtree while
    silently dropping the top-level files beside it. The result was an app that
    started cleanly and then served a blank page (no index.html) and reported
    "5 targets, 0 with structures" (no receptors) — both failures invisible until you
    open the UI.

    Expanding the globs here and handing over a flat list of concrete file paths is
    the form that demonstrably survives bundling.
    """
    paths = [_PROJECT_ROOT / "data" / "targets.json"]
    paths += sorted((_PROJECT_ROOT / "data" / "structures").glob("*.pdb"))
    dist = _PROJECT_ROOT / "ui" / "dist"
    paths += sorted(p for p in dist.rglob("*") if p.is_file())
    return tuple(str(p) for p in paths)

# Targets are read once at import. The cache is baked into the code bundle, so there
# is nothing to refresh at runtime.
TARGETS = load_targets()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize Flyte with the app's own in-cluster identity.

    Not passthrough auth: the booth UI is unauthenticated (it's a kiosk), so there are
    no caller credentials to pass through. Runs are launched as the app's identity.
    """
    await flyte.init_in_cluster.aio(
        org=config.ORG,
        project=flyte.current_project() or config.PROJECT,
        domain=flyte.current_domain() or config.DOMAIN,
    )
    logger.info(
        "Flyte initialized; %d targets, %d with structures",
        len(TARGETS),
        sum(1 for t in TARGETS.values() if t.has_structure),
    )

    # Restore the board before creating tables, so a saved database is adopted rather
    # than shadowed by an empty one. init_db is idempotent and only adds what's missing.
    from flyte.app import ctx as app_ctx

    uri = leaderboard.resolve_uri(getattr(app_ctx(), "raw_data_path", None))
    logger.info("Leaderboard persistence: %s", uri or "DISABLED (no blob path available)")
    restored = await leaderboard.restore()
    leaderboard.init_db()
    stats = leaderboard.stats()
    logger.info(
        "Leaderboard %s: %d runs, %d molecules",
        "restored" if restored else "started empty",
        stats["runs"],
        stats["molecules"],
    )

    yield


app = FastAPI(
    title="BioNeMo Playground API",
    description="Generative virtual screening on Union",
    version="0.1.0",
    lifespan=lifespan,
)

# The Vue dev server runs on a different origin during local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "targets": len(TARGETS)}


@app.get("/api/catalogue")
async def catalogue() -> dict:
    """Everything the UI needs to build its control panel."""
    return {
        "targets": [
            {
                "key": t.key,
                "label": t.label,
                "blurb": t.blurb,
                "uniprot": t.uniprot,
                "length": t.length,
                "source": t.source,
                # Targets without a cached structure can only be run through the
                # slow full pipeline, and the UI greys out the fast path for them.
                "has_structure": t.has_structure,
            }
            for t in TARGETS.values()
        ],
        "seeds": [{"key": s.key, "label": s.label, "smiles": s.smiles, "blurb": s.blurb} for s in SEEDS],
        "defaults": {
            "num_molecules": config.DEFAULT_NUM_MOLECULES,
            "temperature": config.DEFAULT_TEMPERATURE,
            "noise": config.DEFAULT_NOISE,
            "step_size": config.DEFAULT_STEP_SIZE,
            "scoring": config.DEFAULT_SCORING,
            "num_poses": config.DEFAULT_NUM_POSES,
        },
    }


@app.get("/api/structure/{target_key}")
async def structure(target_key: str) -> dict:
    """The receptor PDB, for the 3D viewer."""
    target = TARGETS.get(target_key)
    if target is None:
        raise HTTPException(404, f"Unknown target {target_key!r}")
    if not target.has_structure:
        raise HTTPException(404, f"No cached structure for {target_key!r}")
    return {"target_key": target_key, "pdb": target.structure_pdb}


# ---------------------------------------------------------------------------
# Launching runs
# ---------------------------------------------------------------------------


class ScreenRequest(BaseModel):
    target_key: str
    seed_smiles: str
    nickname: str = Field(default="anonymous", max_length=32)
    num_molecules: int = Field(default=config.DEFAULT_NUM_MOLECULES, ge=1, le=50)
    temperature: float = Field(default=config.DEFAULT_TEMPERATURE, gt=0, le=5)
    noise: float = Field(default=config.DEFAULT_NOISE, ge=0, le=2)
    step_size: int = Field(default=config.DEFAULT_STEP_SIZE, ge=1, le=16)
    scoring: str = Field(default=config.DEFAULT_SCORING)
    num_poses: int = Field(default=config.DEFAULT_NUM_POSES, ge=1, le=40)
    # When true, fold the target in-run instead of using the cached structure.
    # Minutes rather than seconds; the UI makes that clear before you click.
    fold_live: bool = False


@app.post("/api/screen")
async def launch_screen(req: ScreenRequest) -> dict:
    """Launch a screening run and return its identity immediately."""
    target = TARGETS.get(req.target_key)
    if target is None:
        raise HTTPException(404, f"Unknown target {req.target_key!r}")

    if req.fold_live:
        task = full_pipeline
        kwargs: dict = {
            "target_key": target.key,
            "target_label": target.label,
            "sequence": target.sequence,
            "seed_smiles": req.seed_smiles,
            "num_molecules": req.num_molecules,
            "num_poses": req.num_poses,
        }
    else:
        if not target.has_structure:
            raise HTTPException(
                409,
                f"{target.label} has no cached structure. Re-request with fold_live=true, "
                "or precompute it with scripts/precompute_structures.py.",
            )
        task = screen
        kwargs = {
            "target_key": target.key,
            "target_label": target.label,
            "protein_pdb": target.structure_pdb,
            "seed_smiles": req.seed_smiles,
            "num_molecules": req.num_molecules,
            "temperature": req.temperature,
            "noise": req.noise,
            "step_size": req.step_size,
            "scoring": req.scoring,
            "num_poses": req.num_poses,
        }

    try:
        run = await flyte.run.aio(task, **kwargs)
    except flyte.errors.RuntimeSystemError as exc:
        logger.exception("Failed to launch run")
        raise HTTPException(502, f"Could not launch run: {exc}") from exc

    logger.info("Launched %s for %r on %s", run.name, req.nickname, target.key)

    # The nickname belongs to the submission, not the task signature, so it is held
    # here and attached when the run's results reach the leaderboard.
    _NICKNAMES[run.name] = req.nickname

    return {
        "run_name": run.name,
        "run_url": run.url,
        "target_key": target.key,
        "target_label": target.label,
        "mode": "full_pipeline" if req.fold_live else "screen",
    }


# run_name -> nickname, for runs launched by this process.
_NICKNAMES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Polling
# ---------------------------------------------------------------------------


def _as_dicts(result: ScreenResult) -> list[dict]:
    """Candidates as plain dicts, with the bulky pose text dropped."""
    out = []
    for cand in result.candidates:
        d = dataclasses.asdict(cand)
        # Pose geometries are large — ten per molecule, ~1 KB each. The UI fetches them
        # on demand for the selected molecule rather than receiving the whole matrix.
        d.pop("best_pose", None)
        d.pop("poses", None)
        out.append(d)
    return out


@app.get("/api/run/{run_name}")
async def run_status(run_name: str) -> dict:
    """
    Poll a run. Returns its phase, and its results once it has finished.

    On the first successful poll the results are also written to the leaderboard.
    """
    try:
        run = await remote.Run.get.aio(name=run_name)
    except Exception as exc:  # noqa: BLE001 - the SDK raises various types here
        raise HTTPException(404, f"Run {run_name!r} not found: {exc}") from exc

    # `run.phase` is an enum, not a str: it renders as "ActionPhase.SUCCEEDED". Comparing
    # it directly against "SUCCEEDED" is always False, which made every *successful* run
    # come back to the UI as an error with no candidates — a green SUCCEEDED pill sitting
    # next to a red "Run finished in phase ActionPhase.SUCCEEDED" box. Normalise to the
    # bare name once, and use it for both the comparison and what the UI displays.
    phase = str(run.phase).rsplit(".", 1)[-1]

    payload: dict = {"run_name": run_name, "phase": phase, "url": run.url, "done": run.done()}

    if not run.done():
        return payload

    if phase != "SUCCEEDED":
        payload["error"] = f"Run finished in phase {phase}"
        return payload

    try:
        outputs = await run.typed_outputs.aio({"o0": ScreenResult})
        result: ScreenResult = outputs["o0"]
    except Exception as exc:  # noqa: BLE001 - surface rather than 500
        logger.exception("Could not deserialize outputs for %s", run_name)
        payload["error"] = f"Run succeeded but its output could not be read: {exc}"
        return payload

    candidates = _as_dicts(result)
    payload.update(
        {
            "target_key": result.target_key,
            "target_label": result.target_label,
            "seed_smiles": result.seed_smiles,
            "timings": result.timings,
            "folded_in_run": result.folded_in_run,
            "candidates": candidates,
        }
    )

    inserted = leaderboard.record_run(
        run_name=run_name,
        run_url=run.url,
        nickname=_NICKNAMES.get(run_name, "anonymous"),
        target_key=result.target_key,
        target_label=result.target_label,
        seed_smiles=result.seed_smiles,
        candidates=candidates,
    )
    payload["leaderboard_rows_added"] = inserted
    if inserted:
        # Mirror to blob storage so the board outlives this pod.
        await leaderboard.persist()

    return payload


@app.get("/api/run/{run_name}/pose/{index}")
async def run_pose(run_name: str, index: int) -> dict:
    """Fetch one pose's geometry, for the 3D viewer. Kept out of the polling payload."""
    try:
        run = await remote.Run.get.aio(name=run_name)
        outputs = await run.typed_outputs.aio({"o0": ScreenResult})
        result: ScreenResult = outputs["o0"]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"Could not read run {run_name!r}: {exc}") from exc

    if not 0 <= index < len(result.candidates):
        raise HTTPException(404, f"No candidate at index {index}")

    cand = result.candidates[index]
    # Every pose, best first, so the UI can offer rank-by-rank inspection the way the
    # NVIDIA blueprint page does. `pose` stays for the common "just show me the best
    # one" case.
    return {
        "smiles": cand.smiles,
        "pose": cand.best_pose,
        "pose_confidence": cand.pose_confidence,
        "poses": cand.poses,
        "confidences": cand.confidences,
        "protein_pdb": result.protein_pdb,
    }


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


@app.get("/api/leaderboard")
async def get_leaderboard(target_key: str | None = None, limit: int = 20) -> dict:
    entries = leaderboard.top_entries(target_key=target_key, limit=min(limit, 100))
    return {
        "entries": [dataclasses.asdict(e) for e in entries],
        "stats": leaderboard.stats(),
    }


# ---------------------------------------------------------------------------
# NIM health, for the booth operator
# ---------------------------------------------------------------------------


@app.get("/api/nims")
async def nim_health() -> dict:
    """
    Report whether each NIM is up, so you can see at a glance whether the demo is
    ready before someone walks up to it.
    """
    import httpx

    from playground.nims import ALL_NIMS

    async def probe(env) -> dict:
        try:
            endpoint = env.endpoint
        except Exception as exc:  # noqa: BLE001 - app not deployed yet
            return {"name": env.name, "status": "not deployed", "detail": str(exc)}

        url = f"{endpoint.rstrip('/')}/v1/health/ready"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
            return {
                "name": env.name,
                "status": "ready" if resp.status_code == 200 else "not ready",
                "detail": f"HTTP {resp.status_code}",
            }
        except httpx.RequestError as exc:
            # A scale-to-zero NIM that is cold looks exactly like this.
            return {"name": env.name, "status": "unreachable", "detail": type(exc).__name__}

    results = await asyncio.gather(*(probe(env) for env in ALL_NIMS))
    return {"nims": results}


# ---------------------------------------------------------------------------
# The booth UI, served by this same app
#
# Deliberately one app rather than two. When the UI was its own Union app it lived on a
# different subdomain, and with auth enabled the browser's fetch to this API was
# redirected to a login page on yet another origin — which the browser reports simply as
# "Failed to fetch". Serving the built bundle from here makes every request same-origin,
# so one Union login covers the whole demo and there is no CORS surface at all.
#
# Mounted last: StaticFiles at "/" would otherwise shadow the /api routes above.
# html=True makes it serve index.html for unknown paths, which a single-page app needs.
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _no_cache_html(request, call_next):
    """
    Stop browsers caching the app shell.

    Vite fingerprints the JS and CSS, so those are safe to cache forever — but
    index.html is a fixed name whose contents change every build. A cached shell keeps
    pointing at the previous bundle, so a redeploy appears to do nothing: the server
    serves the new build and the browser keeps running the old one. That cost real time
    to diagnose once; at a booth it would mean a fix never reaching the screen in front
    of you.
    """
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
    return response


_UI_DIST = _PROJECT_ROOT / "ui" / "dist"

if _UI_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(_UI_DIST), html=True), name="ui")
else:
    # Not fatal: the API is still usable, and this is the state during local API-only
    # work. Loud, because a deploy that forgets `npm run build` would otherwise ship a
    # booth demo with no UI and no explanation.
    logger.warning("No built UI at %s — run `npm run build` in ui/ before deploying.", _UI_DIST)


# ---------------------------------------------------------------------------
# App environment
# ---------------------------------------------------------------------------

api_image = flyte.Image.from_debian_base(python_version=(3, 12), name="gvs-api").with_pip_packages(
    "fastapi", "uvicorn", "httpx", "pydantic"
)

api_env = FastAPIAppEnvironment(
    name="bionemo-api",
    app=app,
    description="BioNeMo playground API: launches screening runs and keeps the leaderboard",
    image=api_image,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    # Pinned to one replica: the leaderboard is a SQLite file on this container, so a
    # second replica would keep a second, divergent board.
    scaling=flyte.app.Scaling(replicas=1),
    # Authenticated: this endpoint launches GPU workloads, so it is not something to
    # leave open to anyone who finds the URL.
    requires_auth=True,
    # Deploying the API brings the pipeline environment (and through it the NIMs) along.
    depends_on=[screen_env],
    env_vars={"LEADERBOARD_PATH": os.environ.get("LEADERBOARD_PATH", "/tmp/bionemo-leaderboard.sqlite")},
    # The code bundle ships Python files only, so the target catalogue and the cached
    # receptors have to be asked for explicitly. Without them the app imports fine
    # locally and then CrashLoopBackOffs on the cluster with "No target cache at
    # /home/flyte/data/targets.json". Absolute paths because `include` resolves
    # relative entries against api/, and these live one level up at the project root.
    include=_bundled_files(),
)
