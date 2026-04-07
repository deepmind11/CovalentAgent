"""PropertyPredictor agent: ADMET and drug-likeness scoring for candidate molecules.

Takes a PropertyPredictionInput (list of CandidateMolecule) and returns a
PropertyPredictionResult with full MoleculeProperties for each valid candidate.
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from covalent_agent.models.chemprop_wrapper import ChempropWrapper
from covalent_agent.schemas import (
    ADMETProfile,
    MoleculeProperties,
    PropertyPredictionInput,
    PropertyPredictionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights for the overall composite score
# ---------------------------------------------------------------------------
_W_QED = 0.30
_W_TOX = 0.25
_W_DRUGLIKE = 0.20
_W_SA = 0.15
_W_SELECTIVITY = 0.10


class PropertyPredictorAgent:
    """Score candidate molecules on drug-likeness, ADMET, and synthetic accessibility.

    Workflow per candidate:
        1. Parse SMILES with RDKit
        2. Compute physicochemical descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds)
        3. Compute QED score
        4. Count Lipinski rule-of-five violations
        5. Predict ADMET via ChempropWrapper (descriptor fallback if chemprop absent)
        6. Compute synthetic accessibility score
        7. Calculate weighted composite overall_score
    """

    def __init__(self) -> None:
        self._chemprop = ChempropWrapper()

    async def run(self, input: PropertyPredictionInput) -> PropertyPredictionResult:
        """Evaluate all candidate molecules and return property predictions."""
        predictions: list[MoleculeProperties] = []

        for candidate in input.candidates:
            mol = Chem.MolFromSmiles(candidate.smiles)
            if mol is None:
                logger.warning(
                    "Skipping candidate '%s': invalid SMILES '%s'",
                    candidate.name,
                    candidate.smiles,
                )
                continue

            try:
                props = self._evaluate_molecule(candidate.smiles, mol)
                predictions.append(props)
            except Exception:
                logger.exception(
                    "Unexpected error evaluating '%s' (%s); skipping.",
                    candidate.name,
                    candidate.smiles,
                )

        return PropertyPredictionResult(predictions=predictions)

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _evaluate_molecule(self, smiles: str, mol: Chem.Mol) -> MoleculeProperties:
        """Run the full property evaluation pipeline for a single molecule."""
        # Physicochemical descriptors
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        # QED
        qed_score = QED.qed(mol)

        # Lipinski violations
        lipinski_violations = _count_lipinski_violations(mw, logp, hbd, hba)

        # Drug-likeness from Lipinski
        drug_likeness = _clamp(1.0 - (lipinski_violations / 4.0), 0.0, 1.0)

        # ADMET prediction (chemprop or descriptor heuristic)
        admet_raw = self._chemprop.predict_admet(smiles)
        admet = ADMETProfile(**admet_raw)

        # Synthetic accessibility
        sa_props = self._chemprop.predict_properties(smiles)
        sa_score = sa_props["synthetic_accessibility"]

        # SA normalization: map [1, 10] to [0, 1] where 0 = easiest
        sa_normalized = _clamp((sa_score - 1.0) / 9.0, 0.0, 1.0)

        # Selectivity bonus: placeholder heuristic based on TPSA and MW
        # Molecules with moderate size and polarity tend to be more selective
        selectivity_bonus = _estimate_selectivity_bonus(mw, tpsa, logp)

        # Weighted composite score
        overall = (
            _W_QED * qed_score
            + _W_TOX * (1.0 - admet.toxicity_risk)
            + _W_DRUGLIKE * drug_likeness
            + _W_SA * (1.0 - sa_normalized)
            + _W_SELECTIVITY * selectivity_bonus
        )
        overall = round(_clamp(overall, 0.0, 1.0), 3)

        return MoleculeProperties(
            smiles=smiles,
            drug_likeness_score=round(drug_likeness, 3),
            qed_score=round(qed_score, 3),
            lipinski_violations=lipinski_violations,
            admet=admet,
            synthetic_accessibility=round(sa_score, 2),
            overall_score=overall,
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _count_lipinski_violations(
    mw: float,
    logp: float,
    hbd: int,
    hba: int,
) -> int:
    """Count Lipinski rule-of-five violations.

    Rules: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.
    """
    violations = 0
    if mw > 500:
        violations += 1
    if logp > 5:
        violations += 1
    if hbd > 5:
        violations += 1
    if hba > 10:
        violations += 1
    return violations


def _estimate_selectivity_bonus(mw: float, tpsa: float, logp: float) -> float:
    """Heuristic selectivity bonus based on physicochemical space.

    Molecules in the "drug-like" sweet spot (moderate MW, balanced polarity)
    tend to have better selectivity profiles. This is a rough approximation;
    real selectivity requires docking or experimental data.
    """
    # Optimal ranges: MW 300-500, TPSA 60-120, LogP 1-4
    mw_score = 1.0 - _clamp(abs(mw - 400) / 300.0, 0.0, 1.0)
    tpsa_score = 1.0 - _clamp(abs(tpsa - 90) / 100.0, 0.0, 1.0)
    logp_score = 1.0 - _clamp(abs(logp - 2.5) / 4.0, 0.0, 1.0)

    return round((mw_score + tpsa_score + logp_score) / 3.0, 3)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(value, hi))
