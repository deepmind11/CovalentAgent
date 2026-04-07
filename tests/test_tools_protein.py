"""Tests for the protein MCP tool dispatcher.

Covers TOOLS schema sanity, the execute_tool router, and each individual
tool function: lookup_residue, analyze_target (with mocked TargetAnalystAgent),
and fetch_protein_info (with mocked httpx + on-failure DB fallback).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.schemas import TargetAnalysisResult
from covalent_agent.tools import protein_tools


# ---------------------------------------------------------------------------
# TOOLS schema sanity
# ---------------------------------------------------------------------------


class TestToolsSchema:
    """Tests for the TOOLS metadata list."""

    def test_tools_list_is_non_empty(self):
        assert len(protein_tools.TOOLS) >= 3

    @pytest.mark.parametrize(
        "name", ["analyze_target", "lookup_residue", "fetch_protein_info"]
    )
    def test_tool_is_registered(self, name):
        names = [t["name"] for t in protein_tools.TOOLS]
        assert name in names

    def test_each_tool_has_input_schema(self):
        for tool in protein_tools.TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"
            assert "properties" in tool["input_schema"]
            assert "required" in tool["input_schema"]


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteToolDispatch:
    """Tests for the execute_tool router function."""

    async def test_unknown_tool_returns_error(self):
        result = await protein_tools.execute_tool("nonexistent", {})
        assert "error" in result
        assert "nonexistent" in result["error"]

    async def test_dispatches_lookup_residue(self):
        result = await protein_tools.execute_tool(
            "lookup_residue", {"protein_name": "KRAS", "residue": "C12"}
        )
        assert "found" in result


# ---------------------------------------------------------------------------
# lookup_residue tool
# ---------------------------------------------------------------------------


class TestLookupResidue:
    """Tests for the lookup_residue tool implementation."""

    def test_known_protein_residue_found(self):
        result = protein_tools._execute_lookup_residue(
            {"protein_name": "KRAS", "residue": "C12"}
        )
        assert result["found"] is True
        assert result["protein"] == "KRAS"
        assert result["residue"] == "C12"
        assert "sotorasib" in result.get("approved_drugs", [])

    def test_unknown_protein_returns_not_found(self):
        result = protein_tools._execute_lookup_residue(
            {"protein_name": "FAKEPROT", "residue": "C42"}
        )
        assert result["found"] is False
        assert result["protein_name"] == "FAKEPROT"
        assert "message" in result


# ---------------------------------------------------------------------------
# analyze_target tool
# ---------------------------------------------------------------------------


class TestAnalyzeTargetTool:
    """Tests for the analyze_target tool dispatch."""

    async def test_calls_target_analyst_agent(self):
        """The fixed import should now invoke TargetAnalystAgent."""
        fake_result = TargetAnalysisResult(
            protein_name="KRAS",
            uniprot_id="P01116",
            residue_type="cysteine",
            residue_position=12,
            ligandability_score=0.9,
            conservation_score=0.7,
            structural_context="switch II",
            known_drugs=["sotorasib"],
            esm_confidence=0.8,
            rationale="Validated target",
        )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch(
            "covalent_agent.agents.target_analyst.TargetAnalystAgent",
            return_value=mock_agent,
        ):
            result = await protein_tools._execute_analyze_target(
                {
                    "protein_name": "KRAS",
                    "residue": "C12",
                    "indication": "NSCLC",
                }
            )

        assert result["protein_name"] == "KRAS"
        assert result["residue_type"] == "cysteine"
        assert result["ligandability_score"] == 0.9
        mock_agent.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# fetch_protein_info tool
# ---------------------------------------------------------------------------


class TestFetchProteinInfoTool:
    """Tests for the fetch_protein_info tool implementation."""

    async def test_uniprot_success_path(self):
        """A successful UniProt response should be parsed into the result dict."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "primaryAccession": "P01116",
                    "genes": [{"geneName": {"value": "KRAS"}}],
                    "proteinDescription": {
                        "recommendedName": {
                            "fullName": {"value": "GTPase KRas"},
                        },
                    },
                    "sequence": {"length": 189},
                    "organism": {"scientificName": "Homo sapiens"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await protein_tools._execute_fetch_protein_info(
                {"protein_name": "KRAS"}
            )

        assert result["found"] is True
        assert result["uniprot_id"] == "P01116"
        assert result["gene_names"] == "KRAS"
        assert result["protein_name"] == "GTPase KRas"
        assert result["length"] == 189

    async def test_uniprot_empty_results(self):
        """When UniProt returns no results, surface a not-found message."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await protein_tools._execute_fetch_protein_info(
                {"protein_name": "ZZZNOSUCH"}
            )

        assert result["found"] is False
        assert "ZZZNOSUCH" in result["protein_name"]

    async def test_uniprot_failure_falls_back_to_local_db(self):
        """When UniProt raises, the tool should look up the local DB."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("network down"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await protein_tools._execute_fetch_protein_info(
                {"protein_name": "KRAS"}
            )

        # KRAS is in the local DB, so should be found via fallback
        assert result["found"] is True
        assert result.get("source") == "local_database"

    async def test_uniprot_failure_unknown_protein(self):
        """When UniProt raises and the protein is not in the local DB, return error."""
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await protein_tools._execute_fetch_protein_info(
                {"protein_name": "ZZZNOSUCH"}
            )

        assert result["found"] is False
        assert "error" in result
