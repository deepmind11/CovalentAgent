"""Tests for MoleculeDesignerAgent.

Covers candidate generation, candidate count constraints, schema validation,
SMILES validity, molecular weight ranges, and fallback mode for known targets.
RDKit availability is tested in both states (present and absent).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from covalent_agent.schemas import (
    CandidateMolecule,
    MoleculeDesignInput,
    MoleculeDesignResult,
    WarheadRecommendation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_warhead_recs(
    classes: list[tuple[str, float]] | None = None,
) -> list[WarheadRecommendation]:
    """Build a list of WarheadRecommendation objects for testing."""
    if classes is None:
        classes = [
            ("Acrylamide", 0.85),
            ("Chloroacetamide", 0.72),
            ("Cyanoacrylamide", 0.68),
        ]
    return [
        WarheadRecommendation(
            warhead_class=name,
            smarts="[CH2]=[CH]-C(=O)-N",
            reactivity="moderate",
            selectivity="high",
            score=score,
            rationale=f"Test rationale for {name}",
            examples=[],
            mechanism="Michael addition",
        )
        for name, score in classes
    ]


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


class TestMoleculeDesignerAgent:
    """Tests for MoleculeDesignerAgent.run() in both RDKit and fallback modes."""

    async def test_generates_candidates(self, sample_warhead_recommendations):
        """Agent should return at least one candidate molecule."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        input_data = MoleculeDesignInput(
            warhead_recommendations=sample_warhead_recommendations,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        assert isinstance(result, MoleculeDesignResult)
        assert len(result.candidates) >= 1

    async def test_candidate_count_respects_parameter(
        self, sample_warhead_recommendations
    ):
        """Number of returned candidates should not exceed num_candidates."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        for num in [1, 3, 5]:
            input_data = MoleculeDesignInput(
                warhead_recommendations=sample_warhead_recommendations,
                target_protein="KRAS",
                target_residue="C12",
                num_candidates=num,
            )

            agent = MoleculeDesignerAgent()
            result = await agent.run(input_data)

            assert len(result.candidates) <= num, (
                f"Requested {num} candidates but got {len(result.candidates)}"
            )

    async def test_candidate_schema_valid(self, sample_warhead_recommendations):
        """Each candidate must be a valid CandidateMolecule with all fields."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        input_data = MoleculeDesignInput(
            warhead_recommendations=sample_warhead_recommendations,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        for candidate in result.candidates:
            assert isinstance(candidate, CandidateMolecule)
            assert candidate.smiles  # non-empty
            assert candidate.name  # non-empty
            assert candidate.scaffold_type  # non-empty
            assert candidate.warhead_class  # non-empty

    async def test_smiles_not_empty(self, sample_warhead_recommendations):
        """All candidate SMILES strings must be non-empty."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        input_data = MoleculeDesignInput(
            warhead_recommendations=sample_warhead_recommendations,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        for candidate in result.candidates:
            assert len(candidate.smiles) > 0, (
                f"Candidate '{candidate.name}' has empty SMILES"
            )

    async def test_design_rationale_present(self, sample_warhead_recommendations):
        """MoleculeDesignResult should include a non-empty design rationale."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        input_data = MoleculeDesignInput(
            warhead_recommendations=sample_warhead_recommendations,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        assert result.design_rationale
        assert len(result.design_rationale) > 10

    async def test_empty_warheads_returns_empty_result(self):
        """Empty warhead list should return empty candidates with a rationale."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        input_data = MoleculeDesignInput(
            warhead_recommendations=[],
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        assert isinstance(result, MoleculeDesignResult)
        assert len(result.candidates) == 0
        assert result.design_rationale  # should explain why empty


class TestMoleculeDesignerFallback:
    """Tests specifically for the fallback mode (simulating no RDKit)."""

    async def test_fallback_known_targets_kras(self):
        """Fallback mode should return known drug analogs for KRAS C12."""
        from covalent_agent.agents import molecule_designer

        # Temporarily force fallback mode
        original_flag = molecule_designer._HAS_RDKIT
        molecule_designer._HAS_RDKIT = False

        try:
            recs = _make_warhead_recs([("Acrylamide", 0.9)])
            input_data = MoleculeDesignInput(
                warhead_recommendations=recs,
                target_protein="KRAS",
                target_residue="C12",
                num_candidates=5,
            )

            agent = molecule_designer.MoleculeDesignerAgent()
            result = await agent.run(input_data)

            assert len(result.candidates) >= 1
            # At least one candidate should be a known drug analog
            names = [c.name for c in result.candidates]
            has_known = any("sotorasib" in n.lower() or "KRAS" in n for n in names)
            assert has_known, f"Expected known KRAS drug analog in names: {names}"
        finally:
            molecule_designer._HAS_RDKIT = original_flag

    async def test_fallback_known_targets_egfr(self):
        """Fallback mode should return known drug analogs for EGFR C797."""
        from covalent_agent.agents import molecule_designer

        original_flag = molecule_designer._HAS_RDKIT
        molecule_designer._HAS_RDKIT = False

        try:
            recs = _make_warhead_recs([("Acrylamide", 0.9)])
            input_data = MoleculeDesignInput(
                warhead_recommendations=recs,
                target_protein="EGFR",
                target_residue="C797",
                num_candidates=5,
            )

            agent = molecule_designer.MoleculeDesignerAgent()
            result = await agent.run(input_data)

            assert len(result.candidates) >= 1
            names = [c.name for c in result.candidates]
            has_known = any("EGFR" in n or "osimertinib" in n.lower() for n in names)
            assert has_known, f"Expected EGFR drug analog in names: {names}"
        finally:
            molecule_designer._HAS_RDKIT = original_flag

    async def test_fallback_unknown_target_generic(self):
        """Fallback for unknown targets should still return generic candidates."""
        from covalent_agent.agents import molecule_designer

        original_flag = molecule_designer._HAS_RDKIT
        molecule_designer._HAS_RDKIT = False

        try:
            recs = _make_warhead_recs([("Acrylamide", 0.9)])
            input_data = MoleculeDesignInput(
                warhead_recommendations=recs,
                target_protein="UNKNOWNPROT",
                target_residue="C100",
                num_candidates=5,
            )

            agent = molecule_designer.MoleculeDesignerAgent()
            result = await agent.run(input_data)

            assert isinstance(result, MoleculeDesignResult)
            assert len(result.candidates) >= 1
        finally:
            molecule_designer._HAS_RDKIT = original_flag


class TestMoleculeDesignerRDKit:
    """Tests that only run when RDKit is available."""

    @pytest.fixture(autouse=True)
    def _require_rdkit(self):
        """Skip tests if RDKit is not installed."""
        pytest.importorskip("rdkit")

    async def test_molecular_weight_range(self):
        """RDKit-generated candidates with computed MW should be 100-800."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        recs = _make_warhead_recs()
        input_data = MoleculeDesignInput(
            warhead_recommendations=recs,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=10,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        for candidate in result.candidates:
            if candidate.molecular_weight > 0:
                assert 100.0 <= candidate.molecular_weight <= 800.0, (
                    f"MW {candidate.molecular_weight} for '{candidate.name}' "
                    f"outside [100, 800]"
                )

    async def test_rdkit_candidates_have_descriptors(self):
        """RDKit mode should populate molecular descriptors (MW, LogP, etc.)."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        recs = _make_warhead_recs([("Acrylamide", 0.9)])
        input_data = MoleculeDesignInput(
            warhead_recommendations=recs,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=5,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        # At least one candidate should have non-zero descriptors
        has_descriptors = any(c.molecular_weight > 0 for c in result.candidates)
        assert has_descriptors, "Expected at least one candidate with computed MW"

    async def test_candidates_are_deduplicated(self):
        """Candidate SMILES should be unique (no duplicates)."""
        from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent

        recs = _make_warhead_recs()
        input_data = MoleculeDesignInput(
            warhead_recommendations=recs,
            target_protein="KRAS",
            target_residue="C12",
            num_candidates=10,
        )

        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)

        smiles_list = [c.smiles for c in result.candidates]
        assert len(smiles_list) == len(set(smiles_list)), (
            "Duplicate SMILES found in candidates"
        )
