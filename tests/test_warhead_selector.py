"""Tests for WarheadSelectorAgent.

Covers residue-type filtering, score ranges, sort order, schema validation,
and the warhead library integration. The Anthropic API is mocked for all tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.data.warhead_library import WarheadLibrary
from covalent_agent.schemas import (
    WarheadRecommendation,
    WarheadSelectionInput,
    WarheadSelectionResult,
)


# ---------------------------------------------------------------------------
# WarheadLibrary unit tests (no mocking needed, uses real data/warheads.json)
# ---------------------------------------------------------------------------


class TestWarheadLibrary:
    """Tests for the WarheadLibrary query interface over warheads.json."""

    def setup_method(self):
        self.library = WarheadLibrary()

    def test_cysteine_warheads_include_acrylamide(self):
        """Cysteine-targeting warheads must include Acrylamide."""
        warheads = self.library.get_warheads_for_residue("cysteine")
        names = [w["name"] for w in warheads]
        assert "Acrylamide" in names

    def test_cysteine_warheads_include_chloroacetamide(self):
        """Cysteine-targeting warheads must include Chloroacetamide."""
        warheads = self.library.get_warheads_for_residue("cysteine")
        names = [w["name"] for w in warheads]
        assert "Chloroacetamide" in names

    def test_cysteine_warheads_count(self):
        """Cysteine should have at least 5 compatible warhead classes."""
        warheads = self.library.get_warheads_for_residue("cysteine")
        assert len(warheads) >= 5

    def test_lysine_warheads_include_vinyl_sulfonamide(self):
        """Lysine-targeting warheads must include Vinyl sulfonamide."""
        warheads = self.library.get_warheads_for_residue("lysine")
        names = [w["name"] for w in warheads]
        assert "Vinyl sulfonamide" in names

    def test_lysine_warheads_include_sulfonyl_fluoride(self):
        """Lysine-targeting warheads must include Sulfonyl fluoride."""
        warheads = self.library.get_warheads_for_residue("lysine")
        names = [w["name"] for w in warheads]
        assert "Sulfonyl fluoride" in names

    def test_serine_warheads(self):
        """Serine should have compatible warheads (sulfonyl fluorides)."""
        warheads = self.library.get_warheads_for_residue("serine")
        assert len(warheads) >= 1
        names = [w["name"] for w in warheads]
        assert "Sulfonyl fluoride" in names

    def test_unknown_residue_returns_empty(self):
        """An unknown residue type should return an empty list."""
        warheads = self.library.get_warheads_for_residue("alanine")
        assert warheads == []

    def test_score_warhead_for_context_range(self):
        """All warhead scores must be between 0 and 1."""
        warheads = self.library.get_warheads_for_residue("cysteine")
        for w in warheads:
            score = self.library.score_warhead_for_context(w, "cysteine", 0.8)
            assert 0.0 <= score <= 1.0, f"{w['name']} scored {score}, out of range"

    def test_score_warhead_higher_ligandability_increases_score(self):
        """Higher ligandability should generally increase the warhead score."""
        warhead = self.library.get_warhead_by_name("Acrylamide")
        assert warhead is not None

        score_low = self.library.score_warhead_for_context(warhead, "cysteine", 0.2)
        score_high = self.library.score_warhead_for_context(warhead, "cysteine", 0.9)
        assert score_high > score_low

    def test_get_warhead_by_name_case_insensitive(self):
        """Lookup by name should be case-insensitive."""
        w1 = self.library.get_warhead_by_name("acrylamide")
        w2 = self.library.get_warhead_by_name("ACRYLAMIDE")
        assert w1 is not None
        assert w1 == w2

    def test_get_warhead_by_name_nonexistent(self):
        """Nonexistent warhead name should return None."""
        assert self.library.get_warhead_by_name("nonexistent_warhead") is None


# ---------------------------------------------------------------------------
# WarheadSelectorAgent integration tests (Anthropic mocked)
# ---------------------------------------------------------------------------


class TestWarheadSelectorAgent:
    """Tests for the WarheadSelectorAgent.run() method."""

    @pytest.fixture(autouse=True)
    def _mock_anthropic(self, mock_anthropic):
        """Patch the Anthropic client for all tests."""
        self.patcher = patch(
            "covalent_agent.agents.warhead_selector.anthropic.AsyncAnthropic",
            mock_anthropic,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_cysteine_warheads_returns_recommendations(
        self, mock_settings, sample_warhead_input
    ):
        """Cysteine input should return non-empty recommendations."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(sample_warhead_input)

        assert isinstance(result, WarheadSelectionResult)
        assert result.target_residue == "cysteine"
        assert len(result.recommendations) >= 1

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_recommendations_sorted_by_score_descending(
        self, mock_settings, sample_warhead_input
    ):
        """Recommendations must be sorted by score in descending order."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(sample_warhead_input)

        scores = [rec.score for rec in result.recommendations]
        assert scores == sorted(scores, reverse=True), (
            f"Recommendations not sorted descending: {scores}"
        )

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_score_range(self, mock_settings, sample_warhead_input):
        """All recommendation scores must be between 0 and 1."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(sample_warhead_input)

        for rec in result.recommendations:
            assert 0.0 <= rec.score <= 1.0, (
                f"{rec.warhead_class} score {rec.score} out of [0, 1]"
            )

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_output_schema_valid(self, mock_settings, sample_warhead_input):
        """Each recommendation must be a valid WarheadRecommendation."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(sample_warhead_input)

        assert isinstance(result, WarheadSelectionResult)
        for rec in result.recommendations:
            assert isinstance(rec, WarheadRecommendation)
            assert rec.warhead_class
            assert rec.smarts
            assert rec.mechanism
            assert rec.reactivity in ("low", "moderate", "high")
            assert rec.selectivity in ("low", "moderate", "high", "broad")

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_max_recommendations_cap(self, mock_settings, sample_warhead_input):
        """Should return at most 5 recommendations (the _MAX_RECOMMENDATIONS cap)."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(sample_warhead_input)

        assert len(result.recommendations) <= 5

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_empty_residue_type_returns_empty(self, mock_settings):
        """Unknown residue type should produce empty recommendations."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        input_data = WarheadSelectionInput(
            residue_type="alanine",
            ligandability_score=0.5,
            structural_context="Unknown context",
            protein_name="TESTPROT",
        )

        agent = WarheadSelectorAgent()
        result = await agent.run(input_data)

        assert isinstance(result, WarheadSelectionResult)
        assert len(result.recommendations) == 0


class TestWarheadSelectorAgentLysine:
    """Targeted tests for lysine warhead selection."""

    @pytest.fixture(autouse=True)
    def _mock_anthropic(self, mock_anthropic):
        self.patcher = patch(
            "covalent_agent.agents.warhead_selector.anthropic.AsyncAnthropic",
            mock_anthropic,
        )
        self.patcher.start()
        yield
        self.patcher.stop()

    @patch("covalent_agent.agents.warhead_selector.settings")
    async def test_lysine_warheads(self, mock_settings):
        """Lysine targeting should return vinyl sulfonamide and/or sulfonyl fluoride."""
        mock_settings.anthropic_api_key = "test-key"

        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        input_data = WarheadSelectionInput(
            residue_type="lysine",
            ligandability_score=0.7,
            structural_context="Exposed lysine in active site",
            protein_name="TESTPROT",
        )

        agent = WarheadSelectorAgent()
        result = await agent.run(input_data)

        assert len(result.recommendations) >= 1
        warhead_names = {rec.warhead_class for rec in result.recommendations}
        # At least one of these should appear for lysine
        expected = {"Vinyl sulfonamide", "Sulfonyl fluoride", "Aryl sulfonyl fluoride"}
        assert warhead_names & expected, (
            f"Expected at least one of {expected} for lysine, got {warhead_names}"
        )
