"""Tests for PropertyPredictorAgent.

Covers property prediction validity, score ranges (QED, SA, overall),
Lipinski violations, and handling of invalid SMILES.
RDKit is required for these tests (PropertyPredictor hard-imports it).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from covalent_agent.schemas import (
    CandidateMolecule,
    MoleculeProperties,
    PropertyPredictionInput,
    PropertyPredictionResult,
)


# ---------------------------------------------------------------------------
# Skip entire module if RDKit is absent
# ---------------------------------------------------------------------------

rdkit = pytest.importorskip("rdkit", reason="PropertyPredictor requires RDKit")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(smiles: str, name: str = "test_mol") -> CandidateMolecule:
    """Create a CandidateMolecule with minimal fields."""
    return CandidateMolecule(
        smiles=smiles,
        name=name,
        scaffold_type="pyrimidine",
        warhead_class="Acrylamide",
    )


# Real drug-like SMILES for testing
_ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
_SOTORASIB_ANALOG = "c1cnc(NC(=O)C=C)nc1"
_BENZIMIDAZOLE_ACRYLAMIDE = "c1ccc2[nH]cnc2c1NC(=O)C=C"
_SIMPLE_ACRYLAMIDE = "Nc1ccncc1NC(=O)C=C"
_INVALID_SMILES = "NOT_A_REAL_SMILES_STRING"
_NONSENSE_SMILES = "XYZZY"


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestPropertyPredictorAgent:
    """Tests for PropertyPredictorAgent.run() with real RDKit computations."""

    async def test_predict_properties_valid(self):
        """Agent should return valid MoleculeProperties for a good SMILES."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [_make_candidate(_ASPIRIN_SMILES, "aspirin")]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        assert isinstance(result, PropertyPredictionResult)
        assert len(result.predictions) == 1

        props = result.predictions[0]
        assert isinstance(props, MoleculeProperties)
        assert props.smiles == _ASPIRIN_SMILES

    async def test_qed_range(self):
        """QED score must be between 0 and 1 for all valid molecules."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SOTORASIB_ANALOG, "sotorasib_analog"),
            _make_candidate(_BENZIMIDAZOLE_ACRYLAMIDE, "benzimidazole"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        for props in result.predictions:
            assert 0.0 <= props.qed_score <= 1.0, (
                f"QED {props.qed_score} out of range for {props.smiles}"
            )

    async def test_lipinski_violations_count(self):
        """Lipinski violations must be >= 0 for all molecules."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SIMPLE_ACRYLAMIDE, "simple"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        for props in result.predictions:
            assert props.lipinski_violations >= 0, (
                f"Lipinski violations {props.lipinski_violations} < 0 "
                f"for {props.smiles}"
            )

    async def test_sa_score_range(self):
        """Synthetic accessibility score must be between 1 and 10."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SOTORASIB_ANALOG, "sotorasib_analog"),
            _make_candidate(_BENZIMIDAZOLE_ACRYLAMIDE, "benzimidazole"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        for props in result.predictions:
            assert 1.0 <= props.synthetic_accessibility <= 10.0, (
                f"SA score {props.synthetic_accessibility} out of [1, 10] "
                f"for {props.smiles}"
            )

    async def test_overall_score_range(self):
        """Overall composite score must be between 0 and 1."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SOTORASIB_ANALOG, "sotorasib_analog"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        for props in result.predictions:
            assert 0.0 <= props.overall_score <= 1.0, (
                f"Overall score {props.overall_score} out of range "
                f"for {props.smiles}"
            )

    async def test_handles_invalid_smiles(self):
        """Invalid SMILES should be skipped gracefully, not crash the agent."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_INVALID_SMILES, "invalid_mol"),
            _make_candidate(_NONSENSE_SMILES, "nonsense_mol"),
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        # Only the valid SMILES (aspirin) should produce a prediction
        assert len(result.predictions) == 1
        assert result.predictions[0].smiles == _ASPIRIN_SMILES

    async def test_empty_candidates_returns_empty(self):
        """Empty candidate list should return empty predictions."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        input_data = PropertyPredictionInput(candidates=[])

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        assert isinstance(result, PropertyPredictionResult)
        assert len(result.predictions) == 0

    async def test_admet_profile_scores_in_range(self):
        """All ADMET profile scores should be between 0 and 1."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [_make_candidate(_ASPIRIN_SMILES, "aspirin")]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        props = result.predictions[0]
        admet = props.admet
        assert 0.0 <= admet.absorption_score <= 1.0
        assert 0.0 <= admet.distribution_score <= 1.0
        assert 0.0 <= admet.metabolism_score <= 1.0
        assert 0.0 <= admet.excretion_score <= 1.0
        assert 0.0 <= admet.toxicity_risk <= 1.0

    async def test_drug_likeness_score_range(self):
        """Drug-likeness score must be between 0 and 1."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SOTORASIB_ANALOG, "sotorasib_analog"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        for props in result.predictions:
            assert 0.0 <= props.drug_likeness_score <= 1.0

    async def test_aspirin_low_lipinski_violations(self):
        """Aspirin (a drug-like molecule) should have 0 Lipinski violations."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [_make_candidate(_ASPIRIN_SMILES, "aspirin")]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        assert result.predictions[0].lipinski_violations == 0

    async def test_multiple_candidates_all_predicted(self):
        """All valid candidates should receive property predictions."""
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        candidates = [
            _make_candidate(_ASPIRIN_SMILES, "aspirin"),
            _make_candidate(_SOTORASIB_ANALOG, "sotorasib_analog"),
            _make_candidate(_BENZIMIDAZOLE_ACRYLAMIDE, "benzimidazole"),
        ]
        input_data = PropertyPredictionInput(candidates=candidates)

        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)

        assert len(result.predictions) == 3
        predicted_smiles = {p.smiles for p in result.predictions}
        expected_smiles = {_ASPIRIN_SMILES, _SOTORASIB_ANALOG, _BENZIMIDAZOLE_ACRYLAMIDE}
        assert predicted_smiles == expected_smiles
