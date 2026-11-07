"""
Fold every demo target once and cache the structures to data/structures/.

This is the step that makes the booth demo fast. MSA-Search plus OpenFold2 takes
minutes per target, so we pay that cost here — days before the conference — and the
booth loop only ever does generation and docking.

Run it well ahead of time:

    python scripts/precompute_structures.py

It needs the folding NIMs deployed:

    flyte deploy playground/nims.py msa_env openfold2_env

Structures already on disk are skipped, so re-running is cheap and safe. Once the
cache is full you can tear the folding NIMs down and reclaim their GPUs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import flyte  # noqa: E402

from playground.screening import fold_target  # noqa: E402
from playground.targets import STRUCTURES_DIR, load_targets, structure_path  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="Fold just this target key", default=None)
    parser.add_argument("--force", action="store_true", help="Refold even if a structure is cached")
    args = parser.parse_args()

    flyte.init_from_config(str(Path(__file__).parent.parent / "config.yaml"))

    targets = load_targets()
    if args.only:
        if args.only not in targets:
            print(f"Unknown target {args.only!r}. Known: {', '.join(sorted(targets))}")
            return 1
        targets = {args.only: targets[args.only]}

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for key, target in targets.items():
        out = structure_path(key)
        if out.exists() and not args.force:
            print(f"  skip  {key:<24} already cached ({out.stat().st_size // 1024} KB)")
            continue

        print(f"  fold  {key:<24} {target.length} aa ... ", end="", flush=True)
        try:
            run = flyte.run(fold_target, sequence=target.sequence)
            print(f"\n        run: {run.url}")
            run.wait()

            pdb = run.outputs()
            if not isinstance(pdb, str) or "ATOM" not in pdb:
                raise ValueError(f"output does not look like PDB text (got {type(pdb).__name__})")

            out.write_text(pdb)
            print(f"        done -> {out} ({len(pdb) // 1024} KB)")
        except Exception as exc:  # noqa: BLE001 - report and continue to the next target
            print(f"\n        FAILED: {exc}")
            failures.append(key)

    if failures:
        print(f"\n{len(failures)} target(s) failed: {', '.join(failures)}")
        return 1

    print("\nAll structures cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
