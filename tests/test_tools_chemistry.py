"""Tests for the chemistry MCP tool dispatcher.

Covers TOOLS schema, validate_smiles, compute_descriptors, and the
execute_tool router for select_warheads and predict_properties (with
mocked agent classes).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.schemas import (
    ADMETProfile,
    MoleculeProperties,
    PropertyPredictionResult,
    WarheadRecommendation,
    WarheadSelectionResult,
)
from covalent_agent.tools import chemistry_tools


# ---------------------------------------------------------------------------
# TOOLS schema sanity
# ---------------------------------------------------------------------------


class TestToolsSchema:
    """Tests for the chemistry TOOLS metadata."""

    def test_tools_list_is_non_empty(self):
        assert len(chemistry_tools.TOOLS) >= 4

    @pytest.mark.parametrize(
        "name",
        [
            "select_warheads",
            "predict_properties",
            "validate_smiles",
            "compute_descriptors",
        ],
    )
    def test_tool_is_registered(self, name):
        names = [t["name"] for t in chemistry_tools.TOOLS]
        assert name in names

    def test_each_tool_has_input_schema(self):
        for tool in chemistry_tools.TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteToolDispatch:
    async def test_unknown_tool_returns_error(self):
        result = await chemistry_tools.execute_tool("madeup", {})
        assert "error" in result

    async def test_dispatches_validate_smiles(self):
        result = await chemistry_tools.execute_tool(
            "validate_smiles", {"smiles": "CCO"}
        )
        assert result["valid"] is True

    async def test_dispatches_compute_descriptors(self):
        result = await chemistry_tools.execute_tool(
            "compute_descriptors", {"smiles": "CCO"}
        )
        assert result["valid"] is True
        assert result["molecular_weight"] > 0


# ---------------------------------------------------------------------------
# validate_smiles tool
# ---------------------------------------------------------------------------


class TestValidateSmiles:
    def test_valid_smiles_returns_metadata(self):
        result = chemistry_tools._execute_validate_smiles({"smiles": "c1ccccc1"})
        assert result["valid"] is True
        assert result["num_atoms"] > 0
        assert result["molecular_formula"]

    def test_invalid_smiles_returns_error(self):
        result = chemistry_tools._execute_validate_smiles({"smiles": "not-a-molecule"})
        assert result["valid"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# compute_descriptors tool
# ---------------------------------------------------------------------------


class TestComputeDescriptors:
    def test_ethanol_descriptors(self):
        result = chemistry_tools._execute_compute_descriptors({"smiles": "CCO"})
        assert result["valid"] is True
        assert result["molecular_weight"] > 40
        assert result["molecular_weight"] < 50  # ethanol ~46.07
        assert result["num_h_donors"] == 1
        assert result["num_h_acceptors"] == 1
        assert result["lipinski_violations"] == 0

    def test_invalid_smiles_returns_error(self):
        result = chemistry_tools._execute_compute_descriptors(
            {"smiles": "not-valid"}
        )
        assert result["valid"] is False
        assert "error" in result

    def test_lipinski_violations_count(self):
        # Heptadecanoic acid C17 — should violate at least the LogP rule
        result = chemistry_tools._execute_compute_descriptors(
            {"smiles": "CCCCCCCCCCCCCCCCC(=O)O"}
        )
        assert result["valid"] is True
        assert result["lipinski_violations"] >= 1


# ---------------------------------------------------------------------------
# select_warheads tool dispatch
# ---------------------------------------------------------------------------


class TestSelectWarheadsTool:
    async def test_calls_warhead_selector_agent(self):
        """The fixed import should now invoke WarheadSelectorAgent."""
        fake_result = WarheadSelectionResult(
            target_residue="cysteine",
            recommendations=[
                WarheadRecommendation(
                    warhead_class="Acrylamide",
                    smarts="[CH2]=[CH]-C(=O)-N",
                    reactivity="moderate",
                    selectivity="high",
                    score=0.9,
                    rationale="Top pick",
                    examples=["sotorasib"],
                    mechanism="Michael addition",
                )
            ],
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch(
            "covalent_agent.agents.warhead_selector.WarheadSelectorAgent",
            return_value=mock_agent,
        ):
            result = await chemistry_tools._execute_select_warheads(
                {
                    "residue_type": "cysteine",
                    "ligandability_score": 0.85,
                    "structural_context": "switch II",
                    "protein_name": "KRAS",
                }
            )

        assert result["target_residue"] == "cysteine"
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["warhead_class"] == "Acrylamide"
        mock_agent.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# predict_properties tool dispatch
# ---------------------------------------------------------------------------


class TestPredictPropertiesTool:
    async def test_calls_property_predictor_agent(self):
        """The fixed import should now invoke PropertyPredictorAgent."""
        fake_result = PropertyPredictionResult(
            predictions=[
                MoleculeProperties(
                    smiles="CCO",
                    drug_likeness_score=0.7,
                    qed_score=0.5,
                    lipinski_violations=0,
                    admet=ADMETProfile(),
                    synthetic_accessibility=2.0,
                    overall_score=0.65,
                )
            ]
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch(
            "covalent_agent.agents.property_predictor.PropertyPredictorAgent",
            return_value=mock_agent,
        ):
            result = await chemistry_tools._execute_predict_properties(
                {
                    "candidates": [
                        {
                            "smiles": "CCO",
                            "name": "ethanol",
                            "scaffold_type": "alcohol",
                            "warhead_class": "none",
                        }
                    ]
                }
            )

        assert "predictions" in result
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["smiles"] == "CCO"
        mock_agent.run.assert_awaited_once()
