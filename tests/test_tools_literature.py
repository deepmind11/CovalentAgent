"""Tests for the literature MCP tool dispatcher.

Covers TOOLS schema, the execute_tool router, search_literature dispatch
(with mocked LiteratureRAGAgent.run), and get_citation lookups against the
real STARTER_CORPUS.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.schemas import Citation, LiteratureResult
from covalent_agent.tools import literature_tools


# ---------------------------------------------------------------------------
# TOOLS schema sanity
# ---------------------------------------------------------------------------


class TestToolsSchema:
    def test_tools_list_is_non_empty(self):
        assert len(literature_tools.TOOLS) >= 2

    @pytest.mark.parametrize("name", ["search_literature", "get_citation"])
    def test_tool_is_registered(self, name):
        names = [t["name"] for t in literature_tools.TOOLS]
        assert name in names

    def test_each_tool_has_input_schema(self):
        for tool in literature_tools.TOOLS:
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# execute_tool dispatcher
# ---------------------------------------------------------------------------


class TestExecuteToolDispatch:
    async def test_unknown_tool_returns_error(self):
        result = await literature_tools.execute_tool("madeup", {})
        assert "error" in result

    async def test_dispatches_get_citation_known_pmid(self):
        # Sotorasib paper from STARTER_CORPUS
        result = await literature_tools.execute_tool(
            "get_citation", {"pmid": "31645765"}
        )
        assert result["found"] is True


# ---------------------------------------------------------------------------
# search_literature tool dispatch
# ---------------------------------------------------------------------------


class TestSearchLiteratureTool:
    async def test_dispatches_to_literature_rag_agent(self):
        """search_literature should construct a LiteratureQuery and call run()."""
        fake_result = LiteratureResult(
            query="KRAS",
            citations=[
                Citation(
                    title="Test Paper",
                    authors=["Smith J"],
                    journal="Nature",
                    year=2024,
                    pmid="123",
                )
            ],
            summary="Test summary",
            key_findings=["finding 1"],
        )
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=fake_result)

        with patch(
            "covalent_agent.agents.literature_rag.LiteratureRAGAgent",
            return_value=mock_agent,
        ):
            result = await literature_tools._execute_search_literature(
                {
                    "query": "KRAS covalent inhibitor",
                    "protein_name": "KRAS",
                    "warhead_class": "acrylamide",
                    "max_results": 3,
                }
            )

        assert result["query"] == "KRAS"
        assert len(result["citations"]) == 1
        assert result["summary"] == "Test summary"
        mock_agent.run.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_citation tool
# ---------------------------------------------------------------------------


class TestGetCitationTool:
    def test_known_pmid_hit(self):
        # Use a real PMID from STARTER_CORPUS
        result = literature_tools._execute_get_citation({"pmid": "27309814"})
        assert result["found"] is True
        assert result["pmid"] == "27309814"
        assert "title" in result

    def test_unknown_pmid_miss(self):
        result = literature_tools._execute_get_citation({"pmid": "999999999"})
        assert result["found"] is False
        assert result["pmid"] == "999999999"
        assert "message" in result
