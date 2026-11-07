"""
The demo target library.

Design note: protein sequences are **never** hardcoded here from memory. Each target
is identified by its UniProt accession and the sequence is fetched from the UniProt
REST API by ``scripts/fetch_targets.py``, which caches the result into
``data/targets.json``. A transcription error in a 300-residue sequence is invisible
at a booth and would quietly poison every fold downstream, so the sequences come
from the authoritative source or not at all.

The one exception is ``sars-cov-2-mpro``, whose sequence is copied verbatim from the
upstream blueprint notebook so that our output can be diffed against theirs for
parity checking. It is marked ``literal=True``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
TARGETS_CACHE = DATA_DIR / "targets.json"
STRUCTURES_DIR = DATA_DIR / "structures"


@dataclass(frozen=True)
class Target:
    """A protein target available in the playground."""

    key: str
    label: str
    uniprot: str | None
    blurb: str
    # Residue range to submit, 1-indexed inclusive. OpenFold2 gets slow and MSA gets
    # expensive on long sequences, so for large multi-domain proteins we fold only the
    # relevant domain. None means "the whole thing".
    region: tuple[int, int] | None = None
    # Sequence copied verbatim from an authoritative source (the blueprint notebook).
    literal_sequence: str | None = None
    # Pin a specific PDB entry instead of searching RCSB by UniProt accession. Needed
    # when the accession doesn't identify the protein cleanly — SARS-CoV-2 Mpro is a
    # cleaved product of the replicase polyprotein, so its accession returns every
    # non-structural protein in the genome. scripts/fetch_structures.py verifies
    # whatever it downloads against our sequence either way.
    pdb_id: str | None = None

    @property
    def is_literal(self) -> bool:
        return self.literal_sequence is not None


# The upstream blueprint's own example, kept verbatim for parity checking.
# Source: generative-virtual-screening/src/generative-virtual-screening.ipynb, cell 8.
_MPRO_BLUEPRINT = (
    "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRKSNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVR"
    "IQPGQTFSVLACYNGSPSGVYQCAMRPNFTIKGSFLNGSCGSVGFNIDYDCVSFCYMHHMELPTGVHAGTDLEGNFYGPFVDRQTAQAAGTDTTITVNVLAWLYA"
    "AVINGDRWFLNRFTTTLNDFNLVAMKYNYEPLTQDHVDILGPLSAQTGIAVLDMCASLKELLQNGMNGRTILGSALLEDEFTPFDVVRQCSGVTFQ"
)


TARGETS: tuple[Target, ...] = (
    Target(
        key="sars-cov-2-mpro",
        label="SARS-CoV-2 main protease",
        uniprot=None,
        blurb="The upstream blueprint's reference target. Kept for parity with NVIDIA's notebook.",
        literal_sequence=_MPRO_BLUEPRINT,
        # Apo main protease. Verified by sequence identity at fetch time.
        pdb_id="6Y2E",
    ),
    Target(
        key="lysozyme",
        label="Hen egg-white lysozyme",
        uniprot="P00698",
        blurb="The classic structural-biology workhorse. Small, fast to fold, folds well.",
    ),
    Target(
        key="carbonic-anhydrase-2",
        label="Carbonic anhydrase II",
        uniprot="P00918",
        blurb="A textbook druggable enzyme with decades of co-crystal structures.",
    ),
    Target(
        key="cdk2",
        label="CDK2 kinase domain",
        uniprot="P24941",
        blurb="A well-studied oncology kinase. Good showcase for blind docking into an ATP site.",
    ),
    Target(
        key="trypsin",
        label="Bovine trypsin",
        uniprot="P00760",
        blurb="A small serine protease. Reliable fold, well-defined pocket.",
    ),
)

TARGETS_BY_KEY: dict[str, Target] = {t.key: t for t in TARGETS}


# ---------------------------------------------------------------------------
# Seed molecules for GenMol
#
# GenMol consumes SAFE (Sequential Attachment-based Fragment Embedding) strings
# with a `[*{m-n}]` mask marking where new atoms may be generated. As with the
# protein sequences, we do not invent these: the scaffold seed is copied verbatim
# from the blueprint, and the alternative is pure de-novo generation, which needs
# no chemistry knowledge on our part at all.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Seed:
    key: str
    label: str
    smiles: str
    blurb: str


SEEDS: tuple[Seed, ...] = (
    Seed(
        key="blueprint-scaffold",
        label="Blueprint scaffold",
        # Verbatim from the blueprint notebook, cell 18 (fragmented nirmatrelvir).
        smiles="C12OC3C(O)C1O.C3O.[*{25-25}]",
        blurb="NVIDIA's reference seed. Keeps a fixed fragment and grows 25 atoms around it.",
    ),
    Seed(
        key="de-novo-small",
        label="De novo (small)",
        smiles="[*{20-30}]",
        blurb="No scaffold at all. GenMol generates 20-30 atoms from scratch.",
    ),
    Seed(
        key="de-novo-large",
        label="De novo (large)",
        smiles="[*{35-50}]",
        blurb="Unconstrained generation in a larger size range. More diverse, less drug-like.",
    ),
)

SEEDS_BY_KEY: dict[str, Seed] = {s.key: s for s in SEEDS}


# ---------------------------------------------------------------------------
# Cache access
# ---------------------------------------------------------------------------


@dataclass
class TargetRecord:
    """A target plus its resolved sequence and (optionally) precomputed structure."""

    key: str
    label: str
    blurb: str
    sequence: str
    uniprot: str | None = None
    source: str = "unknown"
    structure_pdb: str | None = field(default=None, repr=False)

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def has_structure(self) -> bool:
        return bool(self.structure_pdb)


def load_targets(require_structure: bool = False) -> dict[str, TargetRecord]:
    """
    Load resolved targets from the on-disk cache.

    Raises if the cache is missing, because every alternative — silently falling back
    to a subset, or fabricating a sequence — produces a demo that looks fine and is
    wrong. Run ``python scripts/fetch_targets.py`` first.
    """
    if not TARGETS_CACHE.exists():
        raise FileNotFoundError(
            f"No target cache at {TARGETS_CACHE}. Run `python scripts/fetch_targets.py` to populate it."
        )

    raw = json.loads(TARGETS_CACHE.read_text())
    records: dict[str, TargetRecord] = {}

    for key, entry in raw.items():
        pdb_path = STRUCTURES_DIR / f"{key}.pdb"
        structure = pdb_path.read_text() if pdb_path.exists() else None
        if require_structure and structure is None:
            continue
        records[key] = TargetRecord(
            key=key,
            label=entry["label"],
            blurb=entry["blurb"],
            sequence=entry["sequence"],
            uniprot=entry.get("uniprot"),
            source=entry.get("source", "unknown"),
            structure_pdb=structure,
        )

    return records


def structure_path(target_key: str) -> Path:
    return STRUCTURES_DIR / f"{target_key}.pdb"
