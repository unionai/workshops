"""
Fetch experimental receptor structures from the RCSB PDB.

This replaces folding for the booth. Every demo target is a heavily studied protein
with solved crystal structures, so predicting one with OpenFold2 would be slower, need
1.3 TB of MSA databases and two extra GPUs, and produce a *worse* receptor than the
experimental coordinates. Docking is sensitive to binding-site geometry, and predicted
models frequently place side chains in non-physiological rotamers.

The folding path (scripts/precompute_structures.py) still exists for targets with no
experimental structure. It is simply not on the booth's critical path any more.

    python scripts/fetch_structures.py              # fetch all targets
    python scripts/fetch_structures.py --only cdk2
    python scripts/fetch_structures.py --dry-run    # show candidates, download nothing

Structures are chosen by resolution from X-ray entries matching the target's UniProt
accession, unless the target pins an explicit PDB ID. Every download is verified against
our own sequence before it is written, so a wrong ID fails loudly instead of quietly
poisoning the demo.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from playground.targets import STRUCTURES_DIR, TARGETS_BY_KEY, load_targets, structure_path  # noqa: E402

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Three-letter to one-letter, for checking a downloaded structure really is our protein.
AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified residues, mapped to their parent so they don't break the check.
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
    "HYP": "P",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}

# Waters and the usual crystallisation junk. Dropped from the receptor.
SOLVENT = {"HOH", "WAT", "DOD"}


@dataclass
class Candidate:
    pdb_id: str
    resolution: float | None
    title: str


def search_by_uniprot(accession: str, limit: int = 5) -> list[Candidate]:
    """Find X-ray entries for a UniProt accession, best resolution first."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers"
                            ".reference_sequence_identifiers.database_accession"
                        ),
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": (
                            "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name"
                        ),
                        "operator": "exact_match",
                        "value": "UniProt",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
            "paginate": {"start": 0, "rows": limit},
        },
    }

    resp = httpx.post(RCSB_SEARCH, json=query, timeout=60.0)
    if resp.status_code == 204:
        return []
    resp.raise_for_status()

    ids = [row["identifier"] for row in resp.json().get("result_set", [])]
    return [describe(pdb_id) for pdb_id in ids]


def describe(pdb_id: str) -> Candidate:
    """Fetch an entry's title and resolution."""
    resp = httpx.get(RCSB_ENTRY.format(pdb_id=pdb_id), timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    resolutions = data.get("rcsb_entry_info", {}).get("resolution_combined") or []
    return Candidate(
        pdb_id=pdb_id,
        resolution=resolutions[0] if resolutions else None,
        title=data.get("struct", {}).get("title", "").strip(),
    )


def download(pdb_id: str) -> str:
    resp = httpx.get(RCSB_DOWNLOAD.format(pdb_id=pdb_id), timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def chain_residues(pdb_text: str) -> dict[str, list[tuple[int, str, str]]]:
    """
    Extract each protein chain's observed residues, keyed by PDB residue number.

    Built from ATOM rather than SEQRES on purpose: SEQRES lists the construct, while
    ATOM lists what was actually resolved, and the receptor we hand DiffDock is the
    latter.

    Returns, per chain, an ordered list of (residue number, insertion code, residue).

    Insertion codes are carried rather than discarded because some entries need them
    to be read correctly. Bovine trypsin is numbered by the chymotrypsinogen
    convention, which inserts residues as 184/184A/188/188A rather than renumbering
    the chain. Keying on the numeric field alone makes those collide, silently drops
    every inserted residue, and leaves the sequence looking ~75% identical to itself.
    """
    chains: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, int, str]] = set()

    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        resname = line[17:20].strip()
        chain = line[21].strip() or "A"
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        icode = line[26]

        key = (chain, resnum, icode)
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(chain, []).append((resnum, icode, AA3TO1.get(resname, "X")))

    return chains


def identity(observed: list[tuple[int, str, str]], reference: str) -> tuple[float, int, str]:
    """
    Compare a chain's observed residues against our reference sequence.

    Returns (identity, residues compared, method).

    Two strategies, because PDB entries break sequence comparison in two different
    and mutually exclusive ways:

    * ``numbering`` — trust the PDB residue numbers as indices into the reference.
      This is the only approach that survives *gaps*: CDK2's activation loop is
      unresolved, so its observed residues are not contiguous, and any flat-string
      comparison slides out of register at the first missing residue.

    * ``offset`` — slide the observed residues along the reference and take the best
      alignment. This is the only approach that survives *renumbering*: mature
      proteins rarely share UniProt's numbering. Lysozyme's crystal is numbered 1-129
      for the mature chain while UniProt P00698 counts an 18-residue signal peptide,
      and bovine trypsin uses historical chymotrypsinogen numbering that matches
      nothing at all.

    Neither alone covers the set, so we score both and keep the better. A real
    alignment would handle gaps and offsets together, but that means a dependency and
    this is a sanity check, not science: we only need to answer "is this the right
    protein", and a wrong protein scores near zero under both.
    """
    if not observed or not reference:
        return 0.0, 0, "none"

    # Strategy 1: PDB numbering as a direct index. Inserted residues (those carrying
    # an insertion code) have no position in the canonical sequence, so they sit out.
    num_matches = 0
    num_compared = 0
    for resnum, icode, aa in observed:
        if icode.strip():
            continue
        if 1 <= resnum <= len(reference):
            num_compared += 1
            if reference[resnum - 1] == aa:
                num_matches += 1
    num_score = (num_matches / num_compared) if num_compared else 0.0

    # Strategy 2: best contiguous offset, over every observed residue in order.
    flat = "".join(aa for _, _, aa in observed)
    best_score = 0.0
    best_compared = 0
    for offset in range(-len(flat) + 1, len(reference)):
        matches = 0
        compared = 0
        for i, aa in enumerate(flat):
            j = offset + i
            if 0 <= j < len(reference):
                compared += 1
                if reference[j] == aa:
                    matches += 1
        if compared >= 20:
            score = matches / compared
            if score > best_score:
                best_score, best_compared = score, compared

    if num_score >= best_score and num_compared >= 20:
        return num_score, num_compared, "numbering"
    return best_score, best_compared, "offset"


def clean_receptor(pdb_text: str, chain: str) -> tuple[str, dict[str, int]]:
    """
    Reduce a crystal structure to a docking receptor: one protein chain, no solvent,
    no co-crystallised ligand.

    Dropping the bound ligand matters. Leaving it in would hand DiffDock a receptor
    whose pocket is already occupied, and the generated molecules would have nowhere
    to dock. This is the single most important step in this script.

    Known limitation: this also strips catalytic metals. Carbonic anhydrase II has a
    structural Zn(2+) that its real inhibitors coordinate to, so docking against the
    metal-free receptor is chemically incomplete. Called out in the README; keeping
    HETATM metals is a change for after the conference, once there is time to check
    DiffDock actually handles them.
    """
    kept: list[str] = []
    counts = {"atom": 0, "dropped_hetatm": 0, "dropped_other_chain": 0, "solvent": 0}

    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            if (line[21].strip() or "A") != chain:
                counts["dropped_other_chain"] += 1
                continue
            # Keep only altloc blank or 'A', so we emit one conformer per atom.
            if line[16] not in (" ", "A"):
                continue
            kept.append(line)
            counts["atom"] += 1
        elif line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname in SOLVENT:
                counts["solvent"] += 1
            else:
                counts["dropped_hetatm"] += 1
        elif line.startswith(("TER", "END")):
            continue

    kept.append("END")
    return "\n".join(kept) + "\n", counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", help="Just this target key")
    parser.add_argument("--dry-run", action="store_true", help="Show candidates, download nothing")
    parser.add_argument("--force", action="store_true", help="Overwrite cached structures")
    parser.add_argument("--min-identity", type=float, default=0.8, help="Reject below this identity")
    args = parser.parse_args()

    targets = load_targets()
    if args.only:
        if args.only not in targets:
            print(f"Unknown target {args.only!r}. Known: {', '.join(sorted(targets))}")
            return 1
        targets = {args.only: targets[args.only]}

    STRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for key, target in targets.items():
        print(f"\n{'=' * 70}\n{key}  ({target.label}, {target.length} aa)\n{'=' * 70}")

        out = structure_path(key)
        if out.exists() and not args.force and not args.dry_run:
            print(f"  skip   already cached ({out.stat().st_size // 1024} KB)")
            continue

        spec = TARGETS_BY_KEY.get(key)
        pinned = getattr(spec, "pdb_id", None) if spec else None

        try:
            if pinned:
                candidates = [describe(pinned)]
                print(f"  pinned PDB ID {pinned}")
            elif target.uniprot:
                candidates = search_by_uniprot(target.uniprot)
                print(f"  search UniProt {target.uniprot} -> {len(candidates)} X-ray candidates")
            else:
                print("  FAIL   no pinned PDB ID and no UniProt accession")
                failures.append(key)
                continue
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL   lookup error: {exc}")
            failures.append(key)
            continue

        if not candidates:
            print("  FAIL   no candidate structures found")
            failures.append(key)
            continue

        for cand in candidates:
            res = f"{cand.resolution:.2f} A" if cand.resolution else "n/a"
            print(f"         {cand.pdb_id}  {res:>8}  {cand.title[:60]}")

        if args.dry_run:
            continue

        # Walk candidates best-resolution first, taking the first that verifies.
        chosen = None
        for cand in candidates:
            try:
                pdb_text = download(cand.pdb_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  ..     {cand.pdb_id}: download failed ({exc})")
                continue

            chains = chain_residues(pdb_text)
            if not chains:
                print(f"  ..     {cand.pdb_id}: no protein chains")
                continue

            scored = [(c, *identity(res, target.sequence)) for c, res in chains.items()]
            best_chain, best_id, compared, method = max(scored, key=lambda t: (t[1], t[2]))
            print(
                f"  ..     {cand.pdb_id}: chain {best_chain}, identity {best_id:.2f} "
                f"over {compared} residues ({method})"
            )

            if best_id >= args.min_identity:
                chosen = (cand, pdb_text, best_chain, best_id)
                break

        if chosen is None:
            print(f"  FAIL   no candidate reached identity {args.min_identity}")
            failures.append(key)
            continue

        cand, pdb_text, chain, ident = chosen
        receptor, counts = clean_receptor(pdb_text, chain)
        out.write_text(receptor)

        print(
            f"  ok     {cand.pdb_id} chain {chain}  identity {ident:.2f}  "
            f"{counts['atom']} atoms kept, {counts['dropped_hetatm']} ligand/metal "
            f"and {counts['solvent']} solvent atoms dropped"
        )
        print(f"         -> {out}")

    if failures:
        print(f"\n{len(failures)} target(s) failed: {', '.join(failures)}")
        return 1

    if not args.dry_run:
        print("\nAll receptors cached. No folding, no MSA databases, no extra GPUs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
