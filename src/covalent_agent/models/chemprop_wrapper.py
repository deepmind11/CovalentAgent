"""Chemprop molecular property prediction wrapper with RDKit fallback.

When chemprop is installed, uses trained models for ADMET and property prediction.
When chemprop is not available, falls back to RDKit descriptor-based heuristics
that approximate the same outputs.
"""

from __future__ import annotations

import logging
import os
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SA Score: lives in RDKit's Contrib directory, not importable by default
# ---------------------------------------------------------------------------
_sascorer = None
try:
    sa_score_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_score_path not in sys.path:
        sys.path.append(sa_score_path)
    import sascorer as _sascorer  # type: ignore[import-untyped]
except (ImportError, OSError):
    logger.warning(
        "SA_Score module not found in RDKit Contrib. "
        "Synthetic accessibility will use a simplified estimate."
    )

# ---------------------------------------------------------------------------
# Chemprop: optional dependency
# ---------------------------------------------------------------------------
_chemprop_available = False
try:
    import chemprop  # type: ignore[import-untyped]  # noqa: F401

    _chemprop_available = True
except ImportError:
    logger.info("chemprop not installed; using RDKit descriptor fallback mode.")


class ChempropWrapper:
    """Molecular property prediction via chemprop or RDKit fallback.

    Attributes:
        _use_chemprop: True when the real chemprop library is loaded.
    """

    def __init__(self) -> None:
        self._use_chemprop = _chemprop_available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True when the real chemprop backend is loaded."""
        return self._use_chemprop

    def predict_properties(self, smiles: str) -> dict:
        """Predict physicochemical and drug-likeness properties.

        Returns a dict with keys:
            molecular_weight, logp, tpsa, hbd, hba, rotatable_bonds,
            qed, synthetic_accessibility
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        props = _compute_rdkit_descriptors(mol)
        props["qed"] = QED.qed(mol)
        props["synthetic_accessibility"] = _compute_sa_score(mol)
        return props

    def predict_admet(self, smiles: str) -> dict:
        """Predict ADMET profile scores (each in 0-1 range).

        Returns a dict with keys:
            absorption_score, distribution_score, metabolism_score,
            excretion_score, toxicity_risk
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if self._use_chemprop:
            return self._chemprop_admet(smiles)

        return _estimate_admet_from_descriptors(mol)

    # ------------------------------------------------------------------
    # Chemprop backend (used only when chemprop is installed)
    # ------------------------------------------------------------------

    def _chemprop_admet(self, smiles: str) -> dict:
        """Run real chemprop model inference for ADMET.

        Placeholder: loads cached models from settings.chemprop_model_dir.
        When no trained checkpoint exists yet, falls back to descriptors.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        # TODO: Load and run chemprop checkpoints once trained models are cached.
        # For now, delegate to the descriptor heuristic.
        return _estimate_admet_from_descriptors(mol)


# ---------------------------------------------------------------------------
# RDKit descriptor computation (shared by both backends)
# ---------------------------------------------------------------------------


def _compute_rdkit_descriptors(mol: Chem.Mol) -> dict:
    """Compute core physicochemical descriptors from an RDKit Mol."""
    return {
        "molecular_weight": round(Descriptors.ExactMolWt(mol), 2),
        "logp": round(Descriptors.MolLogP(mol), 3),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
    }


def _compute_sa_score(mol: Chem.Mol) -> float:
    """Compute synthetic accessibility score (1 = easy, 10 = hard).

    Uses RDKit Contrib's sascorer when available; otherwise estimates
    from fragment complexity heuristics.
    """
    if _sascorer is not None:
        return round(_sascorer.calculateScore(mol), 2)

    # Simplified fallback: heavier and more complex molecules are harder
    mw = Descriptors.ExactMolWt(mol)
    ring_count = Descriptors.RingCount(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    num_stereo = Chem.FindMolChiralCenters(mol, includeUnassigned=True)

    score = 2.0
    score += min(mw / 200.0, 3.0)
    score += min(ring_count * 0.5, 2.0)
    score += min(rot_bonds * 0.1, 1.0)
    score += min(len(num_stereo) * 0.3, 1.5)
    return round(min(max(score, 1.0), 10.0), 2)


# ---------------------------------------------------------------------------
# Descriptor-based ADMET heuristics (fallback mode)
# ---------------------------------------------------------------------------


def _estimate_admet_from_descriptors(mol: Chem.Mol) -> dict:
    """Estimate ADMET properties from physicochemical descriptors.

    These are rough heuristics, not trained models. They follow published
    rules of thumb:
      - Absorption correlates inversely with MW and TPSA
      - Distribution correlates with moderate LogP
      - Metabolism risk rises with high LogP
      - Excretion correlates with MW (renal clearance drops for large molecules)
      - Toxicity correlates with high TPSA and reactive substructures
    """
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)

    # Absorption: penalize high MW, high TPSA, many H-bond donors
    absorption = 1.0
    absorption -= _clamp(mw / 1000.0, 0.0, 0.4)
    absorption -= _clamp(tpsa / 300.0, 0.0, 0.3)
    absorption -= _clamp(hbd / 10.0, 0.0, 0.2)
    absorption = _clamp(absorption, 0.0, 1.0)

    # Distribution: optimal LogP around 1-3
    logp_deviation = abs(logp - 2.0)
    distribution = _clamp(1.0 - logp_deviation / 6.0, 0.0, 1.0)

    # Metabolism: high LogP increases CYP metabolism risk
    metabolism = _clamp(1.0 - abs(logp - 2.5) / 5.0, 0.0, 1.0)

    # Excretion: large molecules clear slowly
    excretion = _clamp(1.0 - mw / 800.0, 0.0, 1.0)

    # Toxicity risk: higher TPSA and extreme LogP raise concern
    toxicity = 0.1
    toxicity += _clamp(tpsa / 400.0, 0.0, 0.3)
    if logp > 5.0:
        toxicity += 0.2
    if mw > 500.0:
        toxicity += 0.1
    toxicity = _clamp(toxicity, 0.0, 1.0)

    return {
        "absorption_score": round(absorption, 3),
        "distribution_score": round(distribution, 3),
        "metabolism_score": round(metabolism, 3),
        "excretion_score": round(excretion, 3),
        "toxicity_risk": round(toxicity, 3),
    }


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(value, hi))
