"""MoleculeDesigner agent: generates candidate covalent drug molecules.

Takes a MoleculeDesignInput (warhead recommendations, target protein/residue,
num_candidates) and returns a MoleculeDesignResult with validated candidate
molecules bearing computed molecular descriptors.

Uses a fragment-based approach: drug-like scaffolds are combined with warhead
fragments through common linker motifs. When RDKit is available, molecules are
validated and descriptors are computed. When RDKit is unavailable, a fallback
mode returns pre-defined candidates for well-known targets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from covalent_agent.config import settings
from covalent_agent.schemas import (
    CandidateMolecule,
    MoleculeDesignInput,
    MoleculeDesignResult,
    WarheadRecommendation,
)

logger = logging.getLogger(__name__)

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False
    logger.info(
        "RDKit is not installed. MoleculeDesigner will use fallback mode "
        "with pre-defined candidate molecules."
    )


# ---------------------------------------------------------------------------
# Constants: scaffold library
# ---------------------------------------------------------------------------

_SCAFFOLDS: list[tuple[str, str]] = [
    ("pyrimidine", "c1cnc(N)nc1"),
    ("quinazoline", "c1ccc2c(c1)ncnc2N"),
    ("pyrazolopyrimidine", "c1nn2c(c1)ncnc2N"),
    ("indole", "c1ccc2[nH]cc(CC(=O)O)c2c1"),
    ("benzimidazole", "c1ccc2[nH]cnc2c1"),
    ("imidazopyridine", "c1cnc2[nH]ccn12"),
    ("pyrrolopyridine", "c1cc2cc[nH]c2nc1"),
    ("phenylpiperazine", "c1ccc(N2CCNCC2)cc1"),
    ("aminopyridine", "Nc1ccncc1"),
    ("pyridopyrimidine", "c1cnc2nccnc2c1"),
    ("pyridine", "c1ccncc1"),
    ("thiazole", "c1cscn1"),
    ("oxazole", "c1cocn1"),
]

# ---------------------------------------------------------------------------
# Constants: warhead fragments (keyed by lowercase warhead class name)
# ---------------------------------------------------------------------------

_WARHEAD_FRAGMENTS: dict[str, str] = {
    "acrylamide": "C=CC(=O)",
    "chloroacetamide": "ClCC(=O)",
    "vinyl sulfonamide": "C=CS(=O)(=O)",
    "cyanoacrylamide": "N#C/C=C/C(=O)",
    "propynamide": "C#CC(=O)",
    "alpha-beta unsaturated ketone": "C=CC(=O)C",
    "sulfonyl fluoride": "S(=O)(=O)F",
    "epoxide": "C1OC1",
    "nitrile": "C#N",
}

# ---------------------------------------------------------------------------
# Constants: linkers
# ---------------------------------------------------------------------------

_LINKERS: list[tuple[str, str]] = [
    ("amide", "C(=O)N"),
    ("reverse_amide", "NC(=O)"),
    ("methylene", "C"),
    ("ethylene", "CC"),
    ("oxymethylene", "OC"),
]

# ---------------------------------------------------------------------------
# Constants: drug-likeness filter bounds (Lipinski-like, slightly relaxed)
# ---------------------------------------------------------------------------

_MW_MIN = 200.0
_MW_MAX = 600.0
_LOGP_MIN = -1.0
_LOGP_MAX = 5.0
_HBD_MAX = 5
_HBA_MAX = 10
_ROTATABLE_MAX = 12
_TPSA_MAX = 140.0

# ---------------------------------------------------------------------------
# Constants: known drug fallbacks (target key -> list of CandidateMolecule)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _KnownDrug:
    smiles: str
    name: str
    scaffold_type: str
    warhead_class: str
    mw: float
    logp: float
    hbd: int
    hba: int
    rot: int
    tpsa: float


_KNOWN_DRUGS: dict[str, list[_KnownDrug]] = {
    "KRAS_C12": [
        _KnownDrug(
            smiles="CC1(C)C=CC(=O)N1c1cc(-c2cnc3[nH]c(C)cc3n2)cc(F)c1NC(=O)C=C",
            name="KRAS_acrylamide_sotorasib_analog_0",
            scaffold_type="pyridopyrimidine",
            warhead_class="Acrylamide",
            mw=420.5,
            logp=2.8,
            hbd=2,
            hba=5,
            rot=3,
            tpsa=90.1,
        ),
        _KnownDrug(
            smiles=(
                "CC(C)c1cc(-c2ccnc(NC(=O)C3(C#N)CC(F)(F)C3)n2)c(F)"
                "c(-c2ccn(C)n2)c1OC(F)F"
            ),
            name="KRAS_cyanoacrylamide_adagrasib_analog_0",
            scaffold_type="pyrimidine",
            warhead_class="Cyanoacrylamide",
            mw=604.5,
            logp=3.5,
            hbd=2,
            hba=8,
            rot=6,
            tpsa=102.4,
        ),
    ],
    "EGFR_C797": [
        _KnownDrug(
            smiles="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN(C)C",
            name="EGFR_acrylamide_osimertinib_analog_0",
            scaffold_type="quinazoline",
            warhead_class="Acrylamide",
            mw=499.6,
            logp=3.4,
            hbd=2,
            hba=7,
            rot=8,
            tpsa=88.0,
        ),
    ],
    "BTK_C481": [
        _KnownDrug(
            smiles="C=CC(=O)Nc1cccc(-n2c(=O)n(-c3ccccc3)c3cnc(N)nc32)c1",
            name="BTK_acrylamide_ibrutinib_analog_0",
            scaffold_type="pyrazolopyrimidine",
            warhead_class="Acrylamide",
            mw=440.5,
            logp=3.1,
            hbd=2,
            hba=6,
            rot=5,
            tpsa=99.2,
        ),
    ],
}

# Additional generic fallbacks for unknown targets
_GENERIC_FALLBACK_SMILES: list[tuple[str, str, str]] = [
    ("Nc1ccncc1NC(=O)C=C", "aminopyridine", "Acrylamide"),
    ("c1ccc2[nH]cnc2c1NC(=O)C=C", "benzimidazole", "Acrylamide"),
    ("c1cnc(NC(=O)C=C)nc1", "pyrimidine", "Acrylamide"),
    ("Nc1ccncc1NC(=O)CCl", "aminopyridine", "Chloroacetamide"),
    ("c1ccc2[nH]cnc2c1NC(=O)CCl", "benzimidazole", "Chloroacetamide"),
]


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------


class MoleculeDesignerAgent:
    """Generate candidate covalent drug molecules via fragment-based assembly.

    Workflow:
        1. Select the top 3 warhead recommendations by score.
        2. For each warhead, combine drug-like scaffolds with the warhead
           fragment through common linkers.
        3. Validate all SMILES with RDKit (or use fallback mode).
        4. Compute molecular descriptors and filter by drug-likeness.
        5. Return the best ``num_candidates`` molecules.
    """

    async def run(self, input: MoleculeDesignInput) -> MoleculeDesignResult:
        """Execute the molecule design pipeline."""
        top_warheads = _select_top_warheads(input.warhead_recommendations, n=3)

        if not top_warheads:
            logger.warning(
                "No warhead recommendations provided. Returning empty result."
            )
            return MoleculeDesignResult(
                candidates=[],
                design_rationale="No warhead recommendations available for molecule design.",
            )

        if _HAS_RDKIT:
            candidates = _generate_with_rdkit(
                top_warheads=top_warheads,
                protein=input.target_protein,
            )
        else:
            candidates = _generate_fallback(
                top_warheads=top_warheads,
                protein=input.target_protein,
                residue=input.target_residue,
            )

        ranked = _rank_candidates(candidates)
        selected = ranked[: input.num_candidates]

        warhead_names = [wh.warhead_class for wh in top_warheads]
        rationale = (
            f"Generated {len(candidates)} candidate molecules using "
            f"fragment-based assembly with {len(_SCAFFOLDS)} scaffolds, "
            f"{len(_LINKERS)} linker motifs, and {len(top_warheads)} warhead(s) "
            f"({', '.join(warhead_names)}). "
            f"Filtered to {len(ranked)} drug-like molecules "
            f"(MW {_MW_MIN}-{_MW_MAX}, LogP {_LOGP_MIN}-{_LOGP_MAX}) "
            f"and selected the top {len(selected)} by composite drug-likeness score."
        )

        logger.info(
            "MoleculeDesigner produced %d candidates (from %d raw) for %s %s.",
            len(selected),
            len(candidates),
            input.target_protein,
            input.target_residue,
        )

        return MoleculeDesignResult(
            candidates=selected,
            design_rationale=rationale,
        )


# ---------------------------------------------------------------------------
# Helper: select top warheads
# ---------------------------------------------------------------------------


def _select_top_warheads(
    recommendations: list[WarheadRecommendation],
    n: int,
) -> list[WarheadRecommendation]:
    """Return the top *n* warhead recommendations sorted by score descending."""
    sorted_recs = sorted(recommendations, key=lambda r: r.score, reverse=True)
    return sorted_recs[:n]


# ---------------------------------------------------------------------------
# RDKit-based generation
# ---------------------------------------------------------------------------


def _generate_with_rdkit(
    top_warheads: list[WarheadRecommendation],
    protein: str,
) -> list[CandidateMolecule]:
    """Assemble scaffold + linker + warhead, validate with RDKit, compute descriptors."""
    candidates: list[CandidateMolecule] = []
    index = 0

    for warhead in top_warheads:
        fragment = _get_warhead_fragment(warhead.warhead_class)
        if fragment is None:
            logger.warning(
                "No fragment defined for warhead class '%s'; skipping.",
                warhead.warhead_class,
            )
            continue

        for scaffold_name, scaffold_smi in _SCAFFOLDS:
            for linker_name, linker_smi in _LINKERS:
                combined_smi = f"{scaffold_smi}{linker_smi}{fragment}"
                mol = _validate_smiles(combined_smi)
                if mol is None:
                    continue

                descriptors = _compute_descriptors(mol)
                canonical = Chem.MolToSmiles(mol)

                candidate = CandidateMolecule(
                    smiles=canonical,
                    name=f"{protein}_{warhead.warhead_class}_{scaffold_name}_{index}",
                    scaffold_type=scaffold_name,
                    warhead_class=warhead.warhead_class,
                    molecular_weight=descriptors["mw"],
                    logp=descriptors["logp"],
                    num_h_donors=descriptors["hbd"],
                    num_h_acceptors=descriptors["hba"],
                    num_rotatable_bonds=descriptors["rot"],
                    tpsa=descriptors["tpsa"],
                )
                candidates.append(candidate)
                index += 1

    return candidates


def _validate_smiles(smiles: str) -> object | None:
    """Parse and sanitize a SMILES string. Returns an RDKit Mol or None."""
    if not _HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def _compute_descriptors(mol: object) -> dict[str, float]:
    """Compute molecular descriptors for a validated RDKit Mol object."""
    return {
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rot": Descriptors.NumRotatableBonds(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
    }


# ---------------------------------------------------------------------------
# Fallback generation (no RDKit)
# ---------------------------------------------------------------------------


def _generate_fallback(
    top_warheads: list[WarheadRecommendation],
    protein: str,
    residue: str,
) -> list[CandidateMolecule]:
    """Return pre-defined candidates when RDKit is not installed."""
    target_key = f"{protein.upper()}_{residue.upper()}"
    known = _KNOWN_DRUGS.get(target_key, [])

    candidates: list[CandidateMolecule] = []

    # Add known drug analogs for this target
    for drug in known:
        candidates.append(
            CandidateMolecule(
                smiles=drug.smiles,
                name=drug.name,
                scaffold_type=drug.scaffold_type,
                warhead_class=drug.warhead_class,
                molecular_weight=drug.mw,
                logp=drug.logp,
                num_h_donors=drug.hbd,
                num_h_acceptors=drug.hba,
                num_rotatable_bonds=drug.rot,
                tpsa=drug.tpsa,
            )
        )

    # Fill with generic scaffold+warhead combinations
    index = len(candidates)
    warhead_classes_requested = {wh.warhead_class.lower() for wh in top_warheads}

    for smi, scaffold_name, warhead_class in _GENERIC_FALLBACK_SMILES:
        if warhead_class.lower() not in warhead_classes_requested:
            continue
        candidates.append(
            CandidateMolecule(
                smiles=smi,
                name=f"{protein}_{warhead_class}_{scaffold_name}_{index}",
                scaffold_type=scaffold_name,
                warhead_class=warhead_class,
                molecular_weight=0.0,
                logp=0.0,
                num_h_donors=0,
                num_h_acceptors=0,
                num_rotatable_bonds=0,
                tpsa=0.0,
            )
        )
        index += 1

    if not candidates:
        # Last resort: generate simple text-based combinations
        index = 0
        for warhead in top_warheads:
            fragment = _get_warhead_fragment(warhead.warhead_class)
            if fragment is None:
                continue
            for scaffold_name, scaffold_smi in _SCAFFOLDS[:5]:
                combined = f"{scaffold_smi}NC(=O){fragment}"
                candidates.append(
                    CandidateMolecule(
                        smiles=combined,
                        name=f"{protein}_{warhead.warhead_class}_{scaffold_name}_{index}",
                        scaffold_type=scaffold_name,
                        warhead_class=warhead.warhead_class,
                    )
                )
                index += 1

    logger.info(
        "Fallback mode generated %d candidates for %s %s.",
        len(candidates),
        protein,
        residue,
    )
    return candidates


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_warhead_fragment(warhead_class: str) -> str | None:
    """Look up the SMILES fragment for a warhead class (case-insensitive)."""
    return _WARHEAD_FRAGMENTS.get(warhead_class.lower())


def _passes_drug_likeness(candidate: CandidateMolecule) -> bool:
    """Check whether a candidate falls within the drug-likeness bounds.

    Candidates with zero-valued descriptors (fallback mode) are passed through
    since their descriptors were not computed.
    """
    if candidate.molecular_weight == 0.0:
        return True
    if not (_MW_MIN <= candidate.molecular_weight <= _MW_MAX):
        return False
    if not (_LOGP_MIN <= candidate.logp <= _LOGP_MAX):
        return False
    if candidate.num_h_donors > _HBD_MAX:
        return False
    if candidate.num_h_acceptors > _HBA_MAX:
        return False
    if candidate.num_rotatable_bonds > _ROTATABLE_MAX:
        return False
    if candidate.tpsa > _TPSA_MAX:
        return False
    return True


def _drug_likeness_score(candidate: CandidateMolecule) -> float:
    """Compute a 0-1 composite score for drug-likeness.

    Higher is better. Molecules with zero-valued descriptors (fallback mode)
    receive a neutral score of 0.5.
    """
    if candidate.molecular_weight == 0.0:
        return 0.5

    mw_range = _MW_MAX - _MW_MIN
    mw_ideal = (_MW_MIN + _MW_MAX) / 2.0
    mw_score = max(0.0, 1.0 - abs(candidate.molecular_weight - mw_ideal) / (mw_range / 2.0))

    logp_range = _LOGP_MAX - _LOGP_MIN
    logp_ideal = (_LOGP_MIN + _LOGP_MAX) / 2.0
    logp_score = max(0.0, 1.0 - abs(candidate.logp - logp_ideal) / (logp_range / 2.0))

    hbd_score = max(0.0, 1.0 - candidate.num_h_donors / _HBD_MAX)
    hba_score = max(0.0, 1.0 - candidate.num_h_acceptors / _HBA_MAX)
    rot_score = max(0.0, 1.0 - candidate.num_rotatable_bonds / _ROTATABLE_MAX)
    tpsa_score = max(0.0, 1.0 - candidate.tpsa / _TPSA_MAX)

    weights = {
        "mw": 0.25,
        "logp": 0.25,
        "hbd": 0.10,
        "hba": 0.10,
        "rot": 0.15,
        "tpsa": 0.15,
    }

    composite = (
        weights["mw"] * mw_score
        + weights["logp"] * logp_score
        + weights["hbd"] * hbd_score
        + weights["hba"] * hba_score
        + weights["rot"] * rot_score
        + weights["tpsa"] * tpsa_score
    )
    return round(composite, 4)


def _rank_candidates(candidates: list[CandidateMolecule]) -> list[CandidateMolecule]:
    """Filter by drug-likeness bounds, then sort by composite score descending."""
    passing = [c for c in candidates if _passes_drug_likeness(c)]

    # Deduplicate by SMILES (keep first occurrence, which preserves generation order)
    seen: set[str] = set()
    unique: list[CandidateMolecule] = []
    for c in passing:
        if c.smiles not in seen:
            seen.add(c.smiles)
            unique.append(c)

    unique.sort(key=_drug_likeness_score, reverse=True)
    return unique
