"""Tests for CovalentAgentPipeline (LangGraph supervisor).

Covers graph construction, full mocked pipeline execution, and error handling.
All agent .run() methods are mocked to return fixture data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.schemas import (
    FinalReport,
    LiteratureResult,
    MoleculeDesignResult,
    PropertyPredictionResult,
    RankedCandidate,
    SupervisorState,
    TargetAnalysisInput,
    TargetAnalysisResult,
    WarheadSelectionResult,
)
from covalent_agent.supervisor import CovalentAgentPipeline, _check_error


# ---------------------------------------------------------------------------
# Tests for the _check_error routing function
# ---------------------------------------------------------------------------


class TestCheckError:
    """Tests for the _check_error conditional edge function."""

    def test_no_error_returns_continue(self):
        """When state has no error, routing should return 'continue'."""
        state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name="KRAS", residue="C12", indication=""
            ),
            "target_analysis": None,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "warhead_selection",
            "error": None,
        }
        assert _check_error(state) == "continue"

    def test_with_error_returns_end(self):
        """When state has an error string, routing should return 'end'."""
        state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name="KRAS", residue="C12", indication=""
            ),
            "target_analysis": None,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "target_analysis",
            "error": "Something went wrong",
        }
        assert _check_error(state) == "end"

    def test_empty_error_string_returns_continue(self):
        """Empty string error should be treated as no error (falsy)."""
        state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name="KRAS", residue="C12", indication=""
            ),
            "target_analysis": None,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "target_analysis",
            "error": "",
        }
        assert _check_error(state) == "continue"


# ---------------------------------------------------------------------------
# Tests for graph construction
# ---------------------------------------------------------------------------


class TestPipelineBuilds:
    """Tests for CovalentAgentPipeline graph construction."""

    def test_pipeline_builds_without_error(self):
        """The LangGraph state machine should compile without errors."""
        pipeline = CovalentAgentPipeline()
        assert pipeline.graph is not None

    def test_pipeline_graph_has_nodes(self):
        """Compiled graph should contain all expected node names."""
        pipeline = CovalentAgentPipeline()
        # LangGraph compiled graphs expose nodes via .nodes attribute
        graph_nodes = pipeline.graph.nodes
        expected_nodes = [
            "analyze_target",
            "select_warheads",
            "design_molecules",
            "predict_properties",
            "search_literature",
            "generate_report",
        ]
        for node_name in expected_nodes:
            assert node_name in graph_nodes, (
                f"Node '{node_name}' missing from compiled graph"
            )


# ---------------------------------------------------------------------------
# Full pipeline integration test (all agents mocked)
# ---------------------------------------------------------------------------


class TestFullPipelineMocked:
    """Run the full pipeline with all agent .run() methods returning fixtures."""

    async def test_full_pipeline_mocked(
        self,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_property_prediction_result,
        sample_literature_result,
    ):
        """Mocked pipeline should produce a FinalReport without errors."""
        # Build a minimal FinalReport that the reporter agent would produce
        mock_final_report = FinalReport(
            target_protein="KRAS",
            target_residue="C12",
            indication="NSCLC",
            ligandability_assessment="Ligandability score: 0.85. Validated target.",
            num_candidates_generated=3,
            num_candidates_passing=2,
            ranked_candidates=[
                RankedCandidate(
                    rank=1,
                    smiles="c1cnc(NC(=O)C=C)nc1",
                    name="KRAS_Acrylamide_pyrimidine_0",
                    composite_score=0.72,
                    warhead_class="Acrylamide",
                    drug_likeness=0.75,
                    qed_score=0.62,
                    admet_summary="Absorption: 0.85, Toxicity: 0.15",
                    synthetic_accessibility=2.5,
                    literature_support="Supported by KRAS G12C studies",
                    rationale="Top-ranked candidate for KRAS G12C.",
                ),
            ],
            methodology_summary="Multi-agent pipeline for covalent drug design.",
            citations=[],
        )

        with (
            patch(
                "covalent_agent.supervisor.TargetAnalystAgent"
            ) as mock_target_cls,
            patch(
                "covalent_agent.supervisor.WarheadSelectorAgent"
            ) as mock_warhead_cls,
            patch(
                "covalent_agent.supervisor.MoleculeDesignerAgent"
            ) as mock_designer_cls,
            patch(
                "covalent_agent.supervisor.PropertyPredictorAgent"
            ) as mock_property_cls,
            patch(
                "covalent_agent.supervisor.LiteratureRAGAgent"
            ) as mock_lit_cls,
            patch(
                "covalent_agent.supervisor.ReporterAgent"
            ) as mock_reporter_cls,
        ):
            # Configure each mock agent's .run() to return fixture data
            mock_target_cls.return_value.run = AsyncMock(
                return_value=sample_target_result
            )
            mock_warhead_cls.return_value.run = AsyncMock(
                return_value=sample_warhead_selection_result
            )
            mock_designer_cls.return_value.run = AsyncMock(
                return_value=sample_molecule_design_result
            )
            mock_property_cls.return_value.run = AsyncMock(
                return_value=sample_property_prediction_result
            )
            mock_lit_cls.return_value.run = AsyncMock(
                return_value=sample_literature_result
            )
            mock_reporter_cls.return_value.run = AsyncMock(
                return_value=mock_final_report
            )

            pipeline = CovalentAgentPipeline()
            report = await pipeline.run(
                target="KRAS",
                residue="C12",
                indication="NSCLC",
            )

            assert isinstance(report, FinalReport)
            assert report.target_protein == "KRAS"
            assert report.target_residue == "C12"
            assert report.indication == "NSCLC"
            assert len(report.ranked_candidates) >= 1
            assert report.ranked_candidates[0].rank == 1

    async def test_pipeline_error_handling(self, sample_target_result):
        """Pipeline should raise RuntimeError when an agent step fails."""
        with (
            patch(
                "covalent_agent.supervisor.TargetAnalystAgent"
            ) as mock_target_cls,
            patch(
                "covalent_agent.supervisor.WarheadSelectorAgent"
            ) as mock_warhead_cls,
        ):
            # Target analysis succeeds
            mock_target_cls.return_value.run = AsyncMock(
                return_value=sample_target_result
            )
            # Warhead selection raises an exception
            mock_warhead_cls.return_value.run = AsyncMock(
                side_effect=RuntimeError("Anthropic API unavailable")
            )

            pipeline = CovalentAgentPipeline()

            with pytest.raises(RuntimeError, match="Pipeline failed"):
                await pipeline.run(
                    target="KRAS",
                    residue="C12",
                    indication="NSCLC",
                )

    async def test_pipeline_target_analysis_failure(self):
        """Pipeline should surface errors from the very first step."""
        with patch(
            "covalent_agent.supervisor.TargetAnalystAgent"
        ) as mock_target_cls:
            mock_target_cls.return_value.run = AsyncMock(
                side_effect=ValueError("Invalid residue format")
            )

            pipeline = CovalentAgentPipeline()

            with pytest.raises(RuntimeError, match="Pipeline failed"):
                await pipeline.run(
                    target="KRAS",
                    residue="INVALID",
                    indication="",
                )


# ---------------------------------------------------------------------------
# Supervisor state validation
# ---------------------------------------------------------------------------


class TestSupervisorState:
    """Tests for the SupervisorState TypedDict structure."""

    def test_initial_state_structure(self):
        """Initial state should have all required keys with None defaults."""
        state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name="KRAS", residue="C12", indication="NSCLC"
            ),
            "target_analysis": None,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "target_analysis",
            "error": None,
        }

        assert state["target_input"].protein_name == "KRAS"
        assert state["target_analysis"] is None
        assert state["current_step"] == "target_analysis"
        assert state["error"] is None

    def test_state_accepts_result_objects(self, sample_target_result):
        """State should accept properly-typed result objects."""
        state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name="KRAS", residue="C12", indication=""
            ),
            "target_analysis": sample_target_result,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "warhead_selection",
            "error": None,
        }

        assert state["target_analysis"].protein_name == "KRAS"
        assert state["target_analysis"].ligandability_score == 0.85
