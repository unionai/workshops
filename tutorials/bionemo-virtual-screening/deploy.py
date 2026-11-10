"""
Deploy the whole playground, in dependency order.

    python deploy.py                 # booth setup: warm NIMs + API + UI
    python deploy.py --with-folding   # also deploy MSA-Search and OpenFold2
    python deploy.py --nims-only      # just the NIMs (useful when iterating on them)
    python deploy.py --app-only       # just API + UI (fast iteration on the demo)

Order matters: the NIMs come up first so the API's `.endpoint` lookups resolve, and
the UI goes last because it needs the API's endpoint baked into its bundle.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import flyte  # noqa: E402
import flyte.remote  # noqa: E402

from playground.nims import BOOTH_NIMS, FOLDING_NIMS  # noqa: E402

# NOTE: `api.app` is imported inside main(), *after* the UI build. Importing it here
# would evaluate its `include` list — which globs ui/dist — against the previous
# build's output, so every deploy would ship a one-build-stale bundle. The symptom is
# nasty: the deploy reports success and the browser keeps serving the old UI, so a
# fix you just made appears not to work.


def banner(text: str) -> None:
    print(f"\n{'=' * 64}\n{text}\n{'=' * 64}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-folding", action="store_true", help="Also deploy MSA-Search and OpenFold2")
    parser.add_argument("--nims-only", action="store_true", help="Deploy only the NIM apps")
    parser.add_argument("--app-only", action="store_true", help="Deploy only the API and UI")
    args = parser.parse_args()

    flyte.init_from_config(str(ROOT / "config.yaml"))

    if not args.app_only:
        targets = list(BOOTH_NIMS) + (list(FOLDING_NIMS) if args.with_folding else [])
        banner(f"Deploying {len(targets)} NIM(s)")
        print("First deploy pulls multi-gigabyte images and downloads model weights.")
        print("Expect this to take a while, and watch the app logs in the Union UI.\n")

        flyte.deploy(*targets)
        # Report from the app records rather than the Deployment objects, which don't
        # carry a name or url.
        for env in targets:
            app = flyte.remote.App.get(name=env.name)
            print(f"  {env.name:<20} {app.url}")
            print(f"  {'':<20} {app.endpoint}")

    if args.nims_only:
        print("\nNIMs deployed. Re-run without --nims-only to bring up the API and UI.")
        return 0

    banner("Building UI")
    # The API serves the built bundle, so the build has to happen before the deploy
    # bundles it up. Doing it here rather than in a container start-up script means a
    # broken build fails on your laptop instead of as a CrashLoopBackOff on the cluster.
    ui_dir = ROOT / "ui"
    for cmd in (["npm", "install", "--no-audit", "--no-fund"], ["npm", "run", "build"]):
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=ui_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            raise SystemExit(f"UI build failed: {' '.join(cmd)}")
    print(f"  built -> {ui_dir / 'dist'}")

    # Imported only now, so its include list sees the build we just produced.
    from api.app import api_env

    banner("Deploying API (serves the UI too)")
    flyte.deploy(api_env)
    # `Deployment` carries neither a name nor a url; read them back from the app record.
    api_app = flyte.remote.App.get(name=api_env.name)
    api_endpoint = api_app.endpoint
    print(f"  bionemo-api          {api_app.url}")
    print(f"  endpoint             {api_endpoint}")

    banner("Done")
    print(f"Booth UI:  {api_endpoint}")
    print(f"API:       {api_endpoint}/api/catalogue")
    print(f"Console:   {api_app.url}")
    print("\nBefore the booth opens:")
    print("  1. Check /api/nims shows genmol and diffdock ready.")
    print("  2. Run one screen from the UI. The first request after a deploy pays")
    print("     model load time; do not let a visitor be the one who pays it.")
    print("\nReceptors come from the RCSB PDB and are committed to the repo, so there")
    print("is nothing to precompute and no MSA databases to provision. Run")
    print("`python scripts/fetch_structures.py` only if you add a target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
