"""Tests for TargetAnalystAgent.

Covers residue parsing, ligandability scoring, schema validation,
residue type mapping, and both known/unknown protein paths.
All external dependencies (Anthropic, ESM-2, httpx) are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.agents.target_analyst import (
    TargetAnalystAgent,
    _RESIDUE_TYPE_MAP,
    _parse_residue,
)
from covalent_agent.schemas import TargetAnalysisInput, TargetAnalysisResult


# ---------------------------------------------------------------------------
# Unit tests: _parse_residue helper
# ---------------------------------------------------------------------------


class TestParseResidue:
    """Tests for the _parse_residue helper function."""

    def test_parse_residue_cysteine(self):
        """C12 should parse to letter='C', type='cysteine', position=12."""
        letter, residue_type, position = _parse_residue("C12")
        assert letter == "C"
        assert residue_type == "cysteine"
        assert position == 12

    def test_parse_residue_lysine(self):
        """K481 should parse to letter='K', type='lysine', position=481."""
        letter, residue_type, position = _parse_residue("K481")
        assert letter == "K"
        assert residue_type == "lysine"
        assert position == 481

    def test_parse_residue_serine(self):
        """S195 should parse to letter='S', type='serine', position=195."""
        letter, residue_type, position = _parse_residue("S195")
        assert letter == "S"
        assert residue_type == "serine"
        assert position == 195

    def test_parse_residue_lowercase_input(self):
        """Lowercase input 'c12' should be normalised to uppercase."""
        letter, residue_type, position = _parse_residue("c12")
        assert letter == "C"
        assert residue_type == "cysteine"
        assert position == 12

    def test_parse_residue_unknown_amino_acid(self):
        """Unknown single-letter code should produce 'unknown (X)' type."""
        letter, residue_type, position = _parse_residue("X99")
        assert letter == "X"
        assert "unknown" in residue_type
        assert position == 99

    def test_parse_residue_invalid_format_raises(self):
        """Invalid residue strings should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse residue"):
            _parse_residue("invalid")

    def test_parse_residue_empty_string_raises(self):
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse residue"):
            _parse_residue("")


class TestResidueTypeMapping:
    """Verify that all expected single-letter codes map to full names."""

    @pytest.mark.parametrize(
        "letter,expected_type",
        [
            ("C", "cysteine"),
            ("K", "lysine"),
            ("S", "serine"),
            ("Y", "tyrosine"),
            ("D", "aspartate"),
            ("E", "glutamate"),
            ("T", "threonine"),
            ("H", "histidine"),
            ("R", "arginine"),
        ],
    )
    def test_residue_type_mapping(self, letter: str, expected_type: str):
        """Each letter in _RESIDUE_TYPE_MAP should produce the correct full name."""
        assert _RESIDUE_TYPE_MAP[letter] == expected_type

    def test_residue_type_map_coverage(self):
        """Map should contain exactly 9 residue types."""
        assert len(_RESIDUE_TYPE_MAP) == 9


# ---------------------------------------------------------------------------
# Integration tests: TargetAnalystAgent.run()
# ---------------------------------------------------------------------------


class TestTargetAnalystAgentKnownProtein:
    """Tests for TargetAnalystAgent with a known protein (KRAS C12)."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, mock_esm_wrapper):
        """Patch ESMWrapper and Anthropic for all tests in this class."""
        self.esm_instance = mock_esm_wrapper

    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_parse_residue_known_protein(
        self,
        mock_conservation,
        mock_settings,
        sample_target_input,
    ):
        """KRAS C12 should return known data including UniProt ID and drugs."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.72

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input)

        assert isinstance(result, TargetAnalysisResult)
        assert result.protein_name == "KRAS"
        assert result.uniprot_id == "P01116"
        assert result.residue_type == "cysteine"
        assert result.residue_position == 12
        assert "sotorasib" in result.known_drugs
        assert "adagrasib" in result.known_drugs

    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_ligandability_score_range(
        self,
        mock_conservation,
        mock_settings,
        sample_target_input,
    ):
        """Ligandability score must be between 0 and 1."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.72

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input)

        assert 0.0 <= result.ligandability_score <= 1.0

    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_output_schema_valid(
        self,
        mock_conservation,
        mock_settings,
        sample_target_input,
    ):
        """Result must be a valid TargetAnalysisResult with all required fields."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.72

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input)

        assert isinstance(result, TargetAnalysisResult)
        assert result.protein_name
        assert result.uniprot_id
        assert result.residue_type
        assert result.residue_position > 0
        assert 0.0 <= result.conservation_score <= 1.0
        assert 0.0 <= result.esm_confidence <= 1.0
        assert result.rationale  # non-empty rationale

    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_esm_confidence_fallback_mode(
        self,
        mock_conservation,
        mock_settings,
        sample_target_input,
    ):
        """ESM confidence should be 0.40 in fallback mode (no torch)."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.72

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input)

        assert result.esm_confidence == 0.40


class TestTargetAnalystAgentUnknownProtein:
    """Tests for TargetAnalystAgent with an unknown protein."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, mock_esm_wrapper):
        """Patch ESMWrapper for all tests in this class."""
        self.esm_instance = mock_esm_wrapper

    @patch("covalent_agent.agents.target_analyst._fetch_uniprot")
    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_parse_residue_unknown_protein(
        self,
        mock_conservation,
        mock_settings,
        mock_fetch_uniprot,
        sample_target_input_unknown,
    ):
        """Unknown protein should gracefully fall back, not crash."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.50
        mock_fetch_uniprot.return_value = ("", "", "")

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input_unknown)

        assert isinstance(result, TargetAnalysisResult)
        assert result.protein_name == "FAKEPROT"
        assert result.uniprot_id == "unknown"
        assert result.residue_type == "cysteine"
        assert result.residue_position == 42

    @patch("covalent_agent.agents.target_analyst._fetch_uniprot")
    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_unknown_protein_uses_uniprot_fallback(
        self,
        mock_conservation,
        mock_settings,
        mock_fetch_uniprot,
        sample_target_input_unknown,
    ):
        """When UniProt returns data for an unknown protein, it should be used."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.55
        mock_fetch_uniprot.return_value = ("Q99999", "AAACAAAA", "Fake protein")

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input_unknown)

        assert result.uniprot_id == "Q99999"
        mock_fetch_uniprot.assert_called_once_with("FAKEPROT")

    @patch("covalent_agent.agents.target_analyst._fetch_uniprot")
    @patch("covalent_agent.agents.target_analyst.settings")
    @patch("covalent_agent.agents.target_analyst._compute_conservation_proxy")
    async def test_unknown_protein_ligandability_in_range(
        self,
        mock_conservation,
        mock_settings,
        mock_fetch_uniprot,
        sample_target_input_unknown,
    ):
        """Ligandability score should still be in [0, 1] for unknown proteins."""
        mock_settings.anthropic_api_key = ""
        mock_conservation.return_value = 0.50
        mock_fetch_uniprot.return_value = ("", "", "")

        agent = TargetAnalystAgent()
        result = await agent.run(sample_target_input_unknown)

        assert 0.0 <= result.ligandability_score <= 1.0
