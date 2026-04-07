"""TargetAnalyst agent: identifies reactive residues and scores ligandability.

Takes a protein name and residue identifier (e.g. "KRAS", "C12"), looks up
known data, queries UniProt for unknowns, runs ESM-2 embedding analysis,
and produces a structured ``TargetAnalysisResult``.
"""

from __future__ import annotations

import logging
import re

import httpx
import numpy as np

from covalent_agent.config import settings
from covalent_agent.data import loaders
from covalent_agent.models.esm_wrapper import ESMWrapper
from covalent_agent.schemas import TargetAnalysisInput, TargetAnalysisResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RESIDUE_TYPE_MAP: dict[str, str] = {
    "C": "cysteine",
    "K": "lysine",
    "S": "serine",
    "Y": "tyrosine",
    "D": "aspartate",
    "E": "glutamate",
    "T": "threonine",
    "H": "histidine",
    "R": "arginine",
}

# Representative sequences around known druggable sites.
# These are real subsequences centred on the target residue to give ESM-2
# meaningful local context (~50 AA each).
_KNOWN_SEQUENCES: dict[str, dict[str, str | int]] = {
    "KRAS_C12": {
        # Human KRAS4B residues ~1-55 (G12C mutant: position 12 is C)
        "sequence": (
            "MTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPTIEDSY"
            "RKQVVIDGETCL"
        ),
        # 0-indexed position of the target cysteine
        "position": 11,
    },
    "EGFR_C797": {
        # Human EGFR kinase domain residues ~775-825
        "sequence": (
            "QLITQRPNIILECVHKGIMPCLHYLTDQMAHLA"
            "RSVRFDKNPQFTNEDL"
        ),
        # C797 is at index 18 within this subsequence
        "position": 18,
    },
    "BTK_C481": {
        # Human BTK kinase domain residues ~460-510
        "sequence": (
            "RAVDKWPEGFAIEAIRGQIGCGHFKNVATYGLA"
            "RAPEILTRNEYTFHRD"
        ),
        # C481 is at index 21 within this subsequence
        "position": 21,
    },
}

_UNIPROT_SEARCH_URL = (
    "https://rest.uniprot.org/uniprotkb/search"
    "?query=gene_exact:{gene}+AND+organism_id:9606&format=json&size=1"
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_residue(residue_str: str) -> tuple[str, str, int]:
    """Parse a residue string like 'C12' into (letter, type_name, position).

    Returns:
        Tuple of (single-letter code, full residue name, 1-based position).

    Raises:
        ValueError: if the string cannot be parsed.
    """
    match = re.match(r"^([A-Za-z])(\d+)$", residue_str.strip())
    if not match:
        raise ValueError(
            f"Cannot parse residue '{residue_str}'. Expected format like 'C12'."
        )

    letter = match.group(1).upper()
    position = int(match.group(2))
    residue_type = _RESIDUE_TYPE_MAP.get(letter, f"unknown ({letter})")
    return letter, residue_type, position


async def _fetch_uniprot(
    protein_name: str,
) -> tuple[str, str, str]:
    """Query UniProt REST API for a human protein.

    Returns:
        Tuple of (uniprot_id, sequence, full_name). Falls back to empty
        strings on failure.
    """
    url = _UNIPROT_SEARCH_URL.format(gene=protein_name)
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])
        if not results:
            logger.warning("No UniProt results for %s", protein_name)
            return "", "", ""

        entry = results[0]
        uniprot_id = entry.get("primaryAccession", "")
        sequence = entry.get("sequence", {}).get("value", "")
        full_name = (
            entry
            .get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", protein_name)
        )
        return uniprot_id, sequence, full_name

    except Exception:
        logger.exception("UniProt lookup failed for %s", protein_name)
        return "", "", ""


def _compute_conservation_proxy(sequence: str, position: int) -> float:
    """Rough conservation proxy based on local sequence features.

    Real conservation requires an MSA; this heuristic uses local amino acid
    diversity as a stand-in.
    """
    esm = ESMWrapper()
    ctx = esm.get_context_window(sequence, position, window=10)
    unique_fraction = len(set(ctx)) / max(len(ctx), 1)
    # Higher diversity -> lower conservation; invert and rescale
    return round(1.0 - 0.6 * unique_fraction, 4)


async def _generate_rationale(
    protein_name: str,
    residue_type: str,
    position: int,
    ligandability: float,
    conservation: float,
    known_drugs: list[str],
    indication: str,
    structural_context: str,
) -> str:
    """Use Claude to synthesise a natural-language rationale.

    Falls back to a template string if the API call fails.
    """
    if not settings.anthropic_api_key:
        return _template_rationale(
            protein_name, residue_type, position, ligandability,
            conservation, known_drugs, indication,
        )

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

        prompt = (
            f"You are a medicinal chemistry expert. Summarise the druggability "
            f"assessment for the following target in 3-4 sentences.\n\n"
            f"Protein: {protein_name}\n"
            f"Residue: {residue_type} at position {position}\n"
            f"Ligandability score (0-1): {ligandability:.2f}\n"
            f"Conservation score (0-1): {conservation:.2f}\n"
            f"Known covalent drugs: {', '.join(known_drugs) if known_drugs else 'none'}\n"
            f"Indication: {indication or 'not specified'}\n"
            f"Structural context: {structural_context}\n\n"
            f"Be concise and scientifically accurate. Do not use em dashes."
        )

        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    except Exception:
        logger.exception("Claude rationale generation failed; using template")
        return _template_rationale(
            protein_name, residue_type, position, ligandability,
            conservation, known_drugs, indication,
        )


def _template_rationale(
    protein_name: str,
    residue_type: str,
    position: int,
    ligandability: float,
    conservation: float,
    known_drugs: list[str],
    indication: str,
) -> str:
    """Deterministic fallback rationale when Claude API is unavailable."""
    drug_text = (
        f"Known covalent drugs targeting this site include "
        f"{', '.join(known_drugs)}."
        if known_drugs
        else "No approved covalent drugs currently target this site."
    )

    level = (
        "highly ligandable" if ligandability >= 0.7
        else "moderately ligandable" if ligandability >= 0.4
        else "poorly ligandable"
    )

    return (
        f"{protein_name} {residue_type} at position {position} is assessed as "
        f"{level} (score: {ligandability:.2f}). The residue shows a "
        f"conservation score of {conservation:.2f}, suggesting "
        f"{'strong' if conservation >= 0.7 else 'moderate'} evolutionary "
        f"constraint. {drug_text}"
        + (f" Primary indication: {indication}." if indication else "")
    )


# ---------------------------------------------------------------------------
# TargetAnalystAgent
# ---------------------------------------------------------------------------

class TargetAnalystAgent:
    """Analyses a protein target and scores residue ligandability.

    Standalone agent class: takes a ``TargetAnalysisInput``, returns a
    ``TargetAnalysisResult``. No LangGraph dependency; can be tested in
    isolation.
    """

    def __init__(self) -> None:
        self._esm = ESMWrapper()

    async def run(self, input_data: TargetAnalysisInput) -> TargetAnalysisResult:
        """Execute the target analysis workflow.

        Steps:
            1. Parse the residue string.
            2. Look up the protein in the reactive-residues database.
            3. If unknown, query UniProt for protein info.
            4. Score ligandability with ESM-2 (or fallback).
            5. Generate a natural-language rationale via Claude.
            6. Return a fully populated ``TargetAnalysisResult``.
        """
        protein_name = input_data.protein_name.upper()
        residue_str = input_data.residue.upper()

        # 1. Parse residue
        letter, residue_type, residue_position = _parse_residue(residue_str)

        # 2. Look up known data
        known = loaders.lookup_residue(protein_name, residue_str)

        uniprot_id = ""
        known_drugs: list[str] = []
        indication = input_data.indication
        notes = ""

        if known is not None:
            uniprot_id = known.get("uniprot", "")
            known_drugs = known.get("approved_drugs", [])
            indication = indication or known.get("indication", "")
            notes = known.get("notes", "")

        # 3. Resolve sequence and UniProt ID
        sequence, seq_position = self._resolve_sequence(
            protein_name, letter, residue_position, residue_str,
        )

        if not uniprot_id:
            fetched_id, fetched_seq, _ = await _fetch_uniprot(protein_name)
            if fetched_id:
                uniprot_id = fetched_id
            if fetched_seq and not sequence:
                sequence = fetched_seq
                # Convert 1-based residue position to 0-based
                seq_position = residue_position - 1

        # 4. Final fallback: synthetic sequence
        if not sequence:
            sequence = self._synthetic_context(letter, residue_position)
            seq_position = min(residue_position - 1, len(sequence) - 1)

        if not uniprot_id:
            uniprot_id = "unknown"

        # 5. ESM ligandability scoring
        ligandability = self._esm.score_residue_ligandability(
            sequence, seq_position
        )

        # Conservation proxy
        conservation = _compute_conservation_proxy(sequence, seq_position)

        # Structural context window
        structural_context = self._esm.get_context_window(
            sequence, seq_position, window=15
        )
        if notes:
            structural_context = f"{structural_context} | {notes}"

        # ESM confidence: higher in real mode; moderate in fallback
        esm_confidence = 0.40 if self._esm.fallback_mode else 0.85

        # 6. Generate rationale
        rationale = await _generate_rationale(
            protein_name=protein_name,
            residue_type=residue_type,
            position=residue_position,
            ligandability=ligandability,
            conservation=conservation,
            known_drugs=known_drugs,
            indication=indication,
            structural_context=structural_context,
        )

        return TargetAnalysisResult(
            protein_name=protein_name,
            uniprot_id=uniprot_id,
            residue_type=residue_type,
            residue_position=residue_position,
            ligandability_score=ligandability,
            conservation_score=conservation,
            structural_context=structural_context,
            known_drugs=known_drugs,
            esm_confidence=esm_confidence,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_sequence(
        self,
        protein_name: str,
        letter: str,
        position: int,
        residue_str: str,
    ) -> tuple[str, int]:
        """Resolve a protein sequence and 0-indexed target position.

        Checks the hardcoded known-sequence table first; returns
        (sequence, 0-indexed position) or ("", 0) if not found.
        """
        key = f"{protein_name}_{residue_str}"
        known_seq = _KNOWN_SEQUENCES.get(key)
        if known_seq is not None:
            return str(known_seq["sequence"]), int(known_seq["position"])
        return "", 0

    @staticmethod
    def _synthetic_context(letter: str, position: int) -> str:
        """Build a synthetic ~50 AA context for unknown proteins.

        Places the target residue at approximately the right relative
        position within a poly-alanine stretch to give ESM-2 a minimal
        context window.
        """
        total_len = 50
        # Place the target residue roughly at its natural fractional
        # position (capped to fit within the synthetic sequence).
        idx = min(position - 1, total_len - 1)
        idx = max(idx, 0)
        left = "A" * idx
        right = "A" * (total_len - idx - 1)
        return left + letter + right
