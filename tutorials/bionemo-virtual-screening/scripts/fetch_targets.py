"""
Resolve the demo target library against UniProt and cache it to data/targets.json.

Run this once before deploying:

    python scripts/fetch_targets.py

Re-run it if you add a target to playground/targets.py. It is idempotent and only
writes the cache if every target resolved, so a partial network failure can't leave
you with a half-populated library that fails mysteriously at the booth.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.targets import DATA_DIR, TARGETS, TARGETS_CACHE  # noqa: E402

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"


def fetch_sequence(accession: str) -> str:
    """Fetch a sequence from UniProt and return it as a bare residue string."""
    resp = httpx.get(UNIPROT_FASTA.format(accession=accession), timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    lines = resp.text.strip().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"Unexpected FASTA payload for {accession}: {resp.text[:200]!r}")

    sequence = "".join(line.strip() for line in lines[1:])
    if not sequence.isalpha():
        raise ValueError(f"Non-alphabetic characters in sequence for {accession}")
    return sequence.upper()


def main() -> int:
    resolved: dict[str, dict] = {}
    failures: list[str] = []

    for target in TARGETS:
        if target.is_literal:
            sequence = target.literal_sequence
            source = "blueprint-notebook"
            assert sequence is not None
        else:
            assert target.uniprot is not None, f"{target.key} has neither a literal sequence nor an accession"
            try:
                sequence = fetch_sequence(target.uniprot)
                source = f"uniprot:{target.uniprot}"
            except Exception as exc:  # noqa: BLE001 - we want to report and continue
                print(f"  FAIL  {target.key:<24} {exc}")
                failures.append(target.key)
                continue

        if target.region is not None:
            start, end = target.region
            sequence = sequence[start - 1 : end]
            source += f"#{start}-{end}"

        resolved[target.key] = {
            "label": target.label,
            "blurb": target.blurb,
            "uniprot": target.uniprot,
            "sequence": sequence,
            "source": source,
        }
        print(f"  ok    {target.key:<24} {len(sequence):>5} aa   ({source})")

    if failures:
        print(f"\n{len(failures)} target(s) failed to resolve: {', '.join(failures)}")
        print("Cache not written. Fix the failures and re-run.")
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TARGETS_CACHE.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {len(resolved)} targets to {TARGETS_CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
