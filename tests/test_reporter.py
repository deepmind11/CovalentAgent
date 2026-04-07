"""Tests for ReporterAgent.

Covers helper functions (formatting, literature support search, rationale
construction, methodology template) and the full run() pipeline that merges
molecules with property predictions, ranks them, and produces a FinalReport.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.agents.reporter import (
    ReporterAgent,
    _build_candidate_rationale,
    _build_methodology_template,
    _find_literature_support,
    _format_admet,
)
from covalent_agent.schemas import (
    ADMETProfile,
    CandidateMolecule,
    FinalReport,
    MoleculeProperties,
    PropertyPredictionResult,
    RankedCandidate,
)


# ---------------------------------------------------------------------------
# _format_admet
# ---------------------------------------------------------------------------


class TestFormatAdmet:
    """Tests for the ADMET formatting helper."""

    def test_includes_all_components(self):
        admet = ADMETProfile(
            absorption_score=0.85,
            distribution_score=0.70,
            metabolism_score=0.65,
            excretion_score=0.80,
            toxicity_risk=0.15,
        )
        text = _format_admet(admet)
        assert "Absorption: 0.8" in text
        assert "Distribution: 0.7" in text
        # 0.65 formats to 0.7 under .1f (round-half-to-even)
        assert "Metabolism: 0.7" in text
        assert "Excretion: 0.8" in text
        assert "Toxicity risk: 0.1" in text


# ---------------------------------------------------------------------------
# _find_literature_support
# ---------------------------------------------------------------------------


class TestFindLiteratureSupport:
    """Tests for warhead-citation matching."""

    def test_matches_warhead_in_title(self, sample_citations):
        # Add a citation that mentions Acrylamide
        from covalent_agent.schemas import Citation

        citations = sample_citations + [
            Citation(
                title="Acrylamide warheads in drug design",
                authors=["X"],
                journal="JMC",
                year=2022,
                pmid="42",
                abstract="Review of acrylamide chemistry",
            )
        ]
        result = _find_literature_support("Acrylamide", citations)
        assert "Acrylamide warheads in drug design" in result
        assert result.startswith("Supported by:")

    def test_matches_warhead_in_abstract(self):
        from covalent_agent.schemas import Citation

        citations = [
            Citation(
                title="Generic paper",
                authors=["X"],
                journal="J",
                year=2020,
                pmid="1",
                abstract="Discusses chloroacetamide reactivity in detail.",
            )
        ]
        result = _find_literature_support("Chloroacetamide", citations)
        assert "Generic paper" in result

    def test_no_match_returns_specific_message(self, sample_citations):
        result = _find_literature_support("Nonexistentwarhead", sample_citations)
        assert "No specific literature found" in result
        assert "Nonexistentwarhead" in result

    def test_empty_citations_returns_no_citations_message(self):
        result = _find_literature_support("Acrylamide", [])
        assert result == "No literature citations available."


# ---------------------------------------------------------------------------
# _build_candidate_rationale
# ---------------------------------------------------------------------------


class TestBuildCandidateRationale:
    """Tests for candidate rationale construction."""

    def _make_props(self, drug_likeness=0.75, qed=0.6, sa=2.5, tox=0.15):
        return MoleculeProperties(
            smiles="C",
            drug_likeness_score=drug_likeness,
            qed_score=qed,
            lipinski_violations=0,
            admet=ADMETProfile(toxicity_risk=tox),
            synthetic_accessibility=sa,
            overall_score=0.7,
        )

    def test_includes_warhead_rationale_when_matched(
        self, sample_warhead_recommendations
    ):
        props = self._make_props()
        rationale = _build_candidate_rationale(
            "Acrylamide", sample_warhead_recommendations, props
        )
        # Acrylamide rationale from fixture mentions sotorasib
        assert "Sotorasib" in rationale or "sotorasib" in rationale
        # Property assessment also present
        assert "Drug-likeness" in rationale

    def test_omits_warhead_rationale_when_no_match(
        self, sample_warhead_recommendations
    ):
        props = self._make_props()
        rationale = _build_candidate_rationale(
            "Nonexistent", sample_warhead_recommendations, props
        )
        assert "Drug-likeness" in rationale

    @pytest.mark.parametrize(
        "score,expected_label",
        [
            (0.9, "excellent"),
            (0.6, "good"),
            (0.4, "moderate"),
            (0.1, "poor"),
        ],
    )
    def test_drug_likeness_labeling(self, score, expected_label):
        props = self._make_props(drug_likeness=score)
        rationale = _build_candidate_rationale("Any", [], props)
        assert expected_label in rationale

    @pytest.mark.parametrize(
        "sa,expected_label",
        [
            (2.0, "easy"),
            (4.0, "moderate"),
            (6.0, "difficult"),
            (9.0, "very difficult"),
        ],
    )
    def test_synthetic_accessibility_labeling(self, sa, expected_label):
        props = self._make_props(sa=sa)
        rationale = _build_candidate_rationale("Any", [], props)
        assert expected_label in rationale


# ---------------------------------------------------------------------------
# _build_methodology_template
# ---------------------------------------------------------------------------


class TestBuildMethodologyTemplate:
    """Tests for the deterministic methodology fallback."""

    def test_template_includes_all_inputs(self):
        text = _build_methodology_template(
            protein="KRAS",
            residue="C12",
            score=0.85,
            n_warheads=5,
            top_warhead="Acrylamide",
            n_candidates=10,
            n_passing=7,
            n_citations=3,
        )
        assert "KRAS" in text
        assert "C12" in text
        assert "0.85" in text
        assert "Acrylamide" in text
        assert "10 candidate" in text
        assert "7 candidates passed" in text
        assert "3 relevant publications" in text


# ---------------------------------------------------------------------------
# ReporterAgent.run() — full pipeline
# ---------------------------------------------------------------------------


class TestReporterRun:
    """End-to-end tests for ReporterAgent.run()."""

    async def test_run_produces_final_report(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_property_prediction_result,
        sample_literature_result,
    ):
        agent = ReporterAgent()

        # Force the methodology template path by clearing the api key
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            report = await agent.run(
                target_analysis=sample_target_result,
                warhead_selection=sample_warhead_selection_result,
                molecule_design=sample_molecule_design_result,
                property_prediction=sample_property_prediction_result,
                literature=sample_literature_result,
                target_input=sample_target_input,
            )

        assert isinstance(report, FinalReport)
        assert report.target_protein == "KRAS"
        assert report.target_residue == "C12"
        assert report.indication == "NSCLC"
        # All 3 fixture candidates have matching property predictions
        assert report.num_candidates_generated == 3
        assert len(report.ranked_candidates) == 3
        # Ranked highest first
        scores = [rc.composite_score for rc in report.ranked_candidates]
        assert scores == sorted(scores, reverse=True)
        assert report.ranked_candidates[0].rank == 1
        assert report.ranked_candidates[1].rank == 2

    async def test_run_skips_candidates_without_predictions(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_literature_result,
    ):
        """Candidates without matching property predictions should be skipped."""
        from covalent_agent.schemas import MoleculeDesignResult

        # 2 candidates: one will have a prediction, one won't
        candidates = [
            CandidateMolecule(
                smiles="CCO",
                name="ethanol",
                scaffold_type="alcohol",
                warhead_class="Acrylamide",
            ),
            CandidateMolecule(
                smiles="CCN",
                name="ethylamine",
                scaffold_type="amine",
                warhead_class="Acrylamide",
            ),
        ]
        molecule_design = MoleculeDesignResult(
            candidates=candidates, design_rationale="test"
        )
        property_prediction = PropertyPredictionResult(
            predictions=[
                MoleculeProperties(
                    smiles="CCO",
                    drug_likeness_score=0.75,
                    qed_score=0.6,
                    lipinski_violations=0,
                    admet=ADMETProfile(),
                    synthetic_accessibility=2.0,
                    overall_score=0.7,
                )
                # Note: no prediction for CCN
            ]
        )
        agent = ReporterAgent()
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            report = await agent.run(
                target_analysis=sample_target_result,
                warhead_selection=sample_warhead_selection_result,
                molecule_design=molecule_design,
                property_prediction=property_prediction,
                literature=sample_literature_result,
                target_input=sample_target_input,
            )

        # 2 generated, but only 1 ranked (the other was skipped)
        assert report.num_candidates_generated == 2
        assert len(report.ranked_candidates) == 1
        assert report.ranked_candidates[0].smiles == "CCO"

    async def test_run_counts_passing_candidates(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_literature_result,
    ):
        """num_candidates_passing counts molecules with overall_score > 0.5."""
        # Two pass (0.7, 0.55), one fails (0.4)
        properties = PropertyPredictionResult(
            predictions=[
                MoleculeProperties(
                    smiles="c1cnc(NC(=O)C=C)nc1",
                    drug_likeness_score=0.75,
                    qed_score=0.6,
                    lipinski_violations=0,
                    admet=ADMETProfile(),
                    synthetic_accessibility=2.5,
                    overall_score=0.7,
                ),
                MoleculeProperties(
                    smiles="c1ccc2[nH]cnc2c1NC(=O)C=C",
                    drug_likeness_score=0.5,
                    qed_score=0.5,
                    lipinski_violations=0,
                    admet=ADMETProfile(),
                    synthetic_accessibility=3.5,
                    overall_score=0.55,
                ),
                MoleculeProperties(
                    smiles="Nc1ccncc1NC(=O)C=C",
                    drug_likeness_score=0.3,
                    qed_score=0.3,
                    lipinski_violations=2,
                    admet=ADMETProfile(toxicity_risk=0.6),
                    synthetic_accessibility=5.0,
                    overall_score=0.4,
                ),
            ]
        )
        agent = ReporterAgent()
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            report = await agent.run(
                target_analysis=sample_target_result,
                warhead_selection=sample_warhead_selection_result,
                molecule_design=sample_molecule_design_result,
                property_prediction=properties,
                literature=sample_literature_result,
                target_input=sample_target_input,
            )

        assert report.num_candidates_passing == 2

    async def test_run_uses_anthropic_when_key_present(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_property_prediction_result,
        sample_literature_result,
    ):
        """When an api key is set, ReporterAgent should call Claude."""
        mock_text = MagicMock()
        mock_text.text = "Generated methodology summary text."
        mock_response = MagicMock()
        mock_response.content = [mock_text]
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = ReporterAgent()
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = "fake-key"
            with patch("anthropic.AsyncAnthropic", return_value=mock_client):
                report = await agent.run(
                    target_analysis=sample_target_result,
                    warhead_selection=sample_warhead_selection_result,
                    molecule_design=sample_molecule_design_result,
                    property_prediction=sample_property_prediction_result,
                    literature=sample_literature_result,
                    target_input=sample_target_input,
                )

        assert report.methodology_summary == "Generated methodology summary text."
        mock_client.messages.create.assert_awaited_once()

    async def test_run_falls_back_when_anthropic_raises(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_property_prediction_result,
        sample_literature_result,
    ):
        """If anthropic raises mid-call, fall back to the template summary."""
        agent = ReporterAgent()
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = "fake-key"
            with patch(
                "anthropic.AsyncAnthropic",
                side_effect=RuntimeError("network error"),
            ):
                report = await agent.run(
                    target_analysis=sample_target_result,
                    warhead_selection=sample_warhead_selection_result,
                    molecule_design=sample_molecule_design_result,
                    property_prediction=sample_property_prediction_result,
                    literature=sample_literature_result,
                    target_input=sample_target_input,
                )

        # Falls back to template — KRAS appears in template
        assert "KRAS" in report.methodology_summary
        assert "C12" in report.methodology_summary

    async def test_ranked_candidate_has_full_rationale(
        self,
        sample_target_input,
        sample_target_result,
        sample_warhead_selection_result,
        sample_molecule_design_result,
        sample_property_prediction_result,
        sample_literature_result,
    ):
        agent = ReporterAgent()
        with patch("covalent_agent.agents.reporter.settings") as mock_settings:
            mock_settings.anthropic_api_key = ""
            report = await agent.run(
                target_analysis=sample_target_result,
                warhead_selection=sample_warhead_selection_result,
                molecule_design=sample_molecule_design_result,
                property_prediction=sample_property_prediction_result,
                literature=sample_literature_result,
                target_input=sample_target_input,
            )

        top = report.ranked_candidates[0]
        assert isinstance(top, RankedCandidate)
        assert top.warhead_class
        assert top.admet_summary
        assert top.literature_support
        assert top.rationale
