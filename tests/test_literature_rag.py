"""Tests for LiteratureRAGAgent.

Covers retrieval (Chroma + keyword fallback), corpus seeding, LLM synthesis
parsing, the deterministic fallback path, and PMID lookup. ChromaDB and
Anthropic are mocked so no external services are required.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.agents.literature_rag import (
    STARTER_CORPUS,
    LiteratureRAGAgent,
    _corpus_entry_to_citation,
)
from covalent_agent.schemas import Citation, LiteratureQuery, LiteratureResult


# ---------------------------------------------------------------------------
# _corpus_entry_to_citation helper
# ---------------------------------------------------------------------------


class TestCorpusEntryToCitation:
    """Tests for the _corpus_entry_to_citation helper."""

    def test_full_entry_maps_all_fields(self):
        entry = STARTER_CORPUS[0]
        citation = _corpus_entry_to_citation(entry, relevance=0.9)
        assert isinstance(citation, Citation)
        assert citation.title == entry["title"]
        assert citation.authors == entry["authors"]
        assert citation.journal == entry["journal"]
        assert citation.year == entry["year"]
        assert citation.pmid == entry["pmid"]
        assert citation.relevance_score == 0.9

    def test_default_relevance_is_half(self):
        citation = _corpus_entry_to_citation(STARTER_CORPUS[0])
        assert citation.relevance_score == 0.5

    def test_missing_optional_fields_use_defaults(self):
        sparse = {"title": "X", "pmid": "1"}
        citation = _corpus_entry_to_citation(sparse, relevance=0.4)
        assert citation.title == "X"
        assert citation.authors == []
        assert citation.journal == ""
        assert citation.year == 0
        assert citation.doi == ""
        assert citation.abstract == ""


# ---------------------------------------------------------------------------
# Agent initialisation: ChromaDB unavailable -> fallback mode
# ---------------------------------------------------------------------------


class TestAgentInit:
    """Tests for LiteratureRAGAgent initialisation behavior."""

    def test_init_falls_back_when_chroma_unavailable(self):
        """When chromadb cannot be imported, agent runs in fallback mode."""
        with patch.object(
            LiteratureRAGAgent,
            "_try_init_chroma",
            lambda self: setattr(self, "_chroma_available", False),
        ):
            agent = LiteratureRAGAgent()
            assert agent._chroma_available is False
            assert agent._collection is None

    def test_init_falls_back_on_chroma_exception(self):
        """If ChromaDB raises during init, agent should fall back gracefully."""
        with patch(
            "chromadb.PersistentClient", side_effect=RuntimeError("disk error")
        ):
            agent = LiteratureRAGAgent()
            assert agent._chroma_available is False


# ---------------------------------------------------------------------------
# Keyword-based fallback retrieval
# ---------------------------------------------------------------------------


class TestRetrieveFallback:
    """Tests for the keyword-based fallback retrieval path."""

    @pytest.fixture
    def fallback_agent(self):
        """Build an agent with chroma forcibly disabled."""
        with patch.object(
            LiteratureRAGAgent,
            "_try_init_chroma",
            lambda self: setattr(self, "_chroma_available", False),
        ):
            return LiteratureRAGAgent()

    def test_retrieve_kras_finds_relevant_papers(self, fallback_agent):
        query = LiteratureQuery(
            query="KRAS G12C covalent inhibitor",
            protein_name="KRAS",
            warhead_class="acrylamide",
            max_results=5,
        )
        citations = fallback_agent._retrieve(query)
        assert len(citations) > 0
        assert len(citations) <= 5
        # All citations should be valid Citation objects
        for c in citations:
            assert isinstance(c, Citation)
            assert c.title
        # The top hit should mention KRAS in title or abstract
        top = citations[0]
        searchable = f"{top.title} {top.abstract}".lower()
        assert "kras" in searchable

    def test_retrieve_respects_max_results(self, fallback_agent):
        query = LiteratureQuery(query="covalent", max_results=3)
        citations = fallback_agent._retrieve(query)
        assert len(citations) <= 3

    def test_retrieve_unmatched_query_returns_empty(self, fallback_agent):
        query = LiteratureQuery(query="zzznosuchterm123", max_results=5)
        citations = fallback_agent._retrieve(query)
        assert citations == []

    def test_retrieve_sorts_by_relevance_descending(self, fallback_agent):
        query = LiteratureQuery(
            query="KRAS sotorasib G12C",
            protein_name="KRAS",
            max_results=5,
        )
        citations = fallback_agent._retrieve(query)
        if len(citations) >= 2:
            scores = [c.relevance_score for c in citations]
            assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# ChromaDB retrieval
# ---------------------------------------------------------------------------


class TestRetrieveChroma:
    """Tests for the ChromaDB-backed retrieval path."""

    def test_retrieve_chroma_parses_results(self):
        """Mock a chroma collection and verify result parsing."""
        agent = LiteratureRAGAgent.__new__(LiteratureRAGAgent)
        agent._chroma_available = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 3
        mock_collection.query.return_value = {
            "ids": [["pmid_1", "pmid_2"]],
            "distances": [[0.1, 0.4]],
            "metadatas": [
                [
                    {
                        "title": "Paper 1",
                        "authors": "Smith J, Doe A",
                        "journal": "Nature",
                        "year": 2020,
                        "pmid": "1",
                    },
                    {
                        "title": "Paper 2",
                        "authors": "Lee K",
                        "journal": "Cell",
                        "year": 2021,
                        "pmid": "2",
                    },
                ]
            ],
            "documents": [["doc 1 text", "doc 2 text"]],
        }
        agent._collection = mock_collection

        query = LiteratureQuery(
            query="test", protein_name="KRAS", warhead_class="acrylamide", max_results=2
        )
        citations = agent._retrieve_chroma(query)

        assert len(citations) == 2
        assert citations[0].title == "Paper 1"
        assert citations[0].authors == ["Smith J", "Doe A"]
        assert citations[0].journal == "Nature"
        assert citations[0].year == 2020
        # 1.0 - 0.1 = 0.9
        assert citations[0].relevance_score == 0.9
        # 1.0 - 0.4 = 0.6
        assert citations[1].relevance_score == 0.6

    def test_retrieve_chroma_empty_results(self):
        agent = LiteratureRAGAgent.__new__(LiteratureRAGAgent)
        agent._chroma_available = True

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {"ids": []}
        agent._collection = mock_collection

        citations = agent._retrieve_chroma(
            LiteratureQuery(query="anything", max_results=5)
        )
        assert citations == []


# ---------------------------------------------------------------------------
# _parse_synthesis: parse Claude's structured response
# ---------------------------------------------------------------------------


class TestParseSynthesis:
    """Tests for the _parse_synthesis static helper."""

    def test_parses_well_formed_response(self):
        text = (
            "SUMMARY:\nKRAS G12C is a validated covalent target.\n\n"
            "KEY FINDINGS:\n"
            "- Sotorasib is the first approved KRAS G12C inhibitor\n"
            "- Acrylamide warheads are validated\n"
            "- Selectivity is excellent\n"
        )
        summary, findings = LiteratureRAGAgent._parse_synthesis(text)
        assert "KRAS G12C is a validated covalent target." in summary
        assert len(findings) == 3
        assert findings[0].startswith("Sotorasib")
        assert findings[1].startswith("Acrylamide")

    def test_parses_asterisk_bullets(self):
        text = (
            "SUMMARY:\nShort summary.\n\n"
            "KEY FINDINGS:\n* Finding A\n* Finding B\n"
        )
        summary, findings = LiteratureRAGAgent._parse_synthesis(text)
        assert summary == "Short summary."
        assert findings == ["Finding A", "Finding B"]

    def test_unstructured_response_falls_back_to_summary_only(self):
        text = "Just a free-form paragraph with no structure."
        summary, findings = LiteratureRAGAgent._parse_synthesis(text)
        assert summary == text
        assert findings == []


# ---------------------------------------------------------------------------
# _fallback_synthesis
# ---------------------------------------------------------------------------


class TestFallbackSynthesis:
    """Tests for the deterministic fallback summary path."""

    def test_fallback_summary_includes_query_and_count(self, sample_citations):
        summary, findings = LiteratureRAGAgent._fallback_synthesis(
            "covalent KRAS", sample_citations
        )
        assert "covalent KRAS" in summary
        assert str(len(sample_citations)) in summary
        assert len(findings) <= 5

    def test_fallback_with_no_citations(self):
        summary, findings = LiteratureRAGAgent._fallback_synthesis("any", [])
        assert "Found 0" in summary
        assert findings == []


# ---------------------------------------------------------------------------
# _synthesize: full LLM synthesis path
# ---------------------------------------------------------------------------


class TestSynthesize:
    """Tests for the _synthesize method."""

    @pytest.fixture
    def fallback_agent(self):
        with patch.object(
            LiteratureRAGAgent,
            "_try_init_chroma",
            lambda self: setattr(self, "_chroma_available", False),
        ):
            return LiteratureRAGAgent()

    async def test_synthesize_no_citations_returns_empty(self, fallback_agent):
        summary, findings = await fallback_agent._synthesize("query", [])
        assert "No relevant literature" in summary
        assert findings == []

    async def test_synthesize_uses_anthropic_when_available(
        self, fallback_agent, sample_citations
    ):
        """Synthesis should call AsyncAnthropic and parse the response."""
        mock_text = MagicMock()
        mock_text.text = (
            "SUMMARY:\nKRAS G12C summary.\n\nKEY FINDINGS:\n- F1\n- F2\n"
        )
        mock_response = MagicMock()
        mock_response.content = [mock_text]

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("anthropic.AsyncAnthropic", return_value=mock_client):
            summary, findings = await fallback_agent._synthesize(
                "covalent KRAS", sample_citations
            )

        assert "KRAS G12C summary." in summary
        assert findings == ["F1", "F2"]
        mock_client.messages.create.assert_awaited_once()

    async def test_synthesize_falls_back_on_anthropic_error(
        self, fallback_agent, sample_citations
    ):
        """If the anthropic call raises, fall back to deterministic synthesis."""
        with patch(
            "anthropic.AsyncAnthropic",
            side_effect=RuntimeError("missing api key"),
        ):
            summary, findings = await fallback_agent._synthesize(
                "covalent KRAS", sample_citations
            )

        assert "Found" in summary
        assert "covalent KRAS" in summary


# ---------------------------------------------------------------------------
# Full run() integration
# ---------------------------------------------------------------------------


class TestRun:
    """End-to-end tests for LiteratureRAGAgent.run()."""

    @pytest.fixture
    def fallback_agent(self):
        with patch.object(
            LiteratureRAGAgent,
            "_try_init_chroma",
            lambda self: setattr(self, "_chroma_available", False),
        ):
            return LiteratureRAGAgent()

    async def test_run_returns_literature_result(self, fallback_agent):
        query = LiteratureQuery(
            query="KRAS G12C covalent inhibitor",
            protein_name="KRAS",
            warhead_class="acrylamide",
            max_results=3,
        )

        # Force the deterministic fallback synthesis path so we don't depend on
        # an actual anthropic client at runtime
        with patch(
            "anthropic.AsyncAnthropic",
            side_effect=RuntimeError("no api key"),
        ):
            result = await fallback_agent.run(query)

        assert isinstance(result, LiteratureResult)
        assert result.query == query.query
        assert len(result.citations) > 0
        assert result.summary  # non-empty


# ---------------------------------------------------------------------------
# get_citation_by_pmid
# ---------------------------------------------------------------------------


class TestGetCitationByPmid:
    """Tests for the synchronous PMID lookup."""

    @pytest.fixture
    def fallback_agent(self):
        with patch.object(
            LiteratureRAGAgent,
            "_try_init_chroma",
            lambda self: setattr(self, "_chroma_available", False),
        ):
            return LiteratureRAGAgent()

    def test_lookup_known_pmid_returns_citation(self, fallback_agent):
        # Sotorasib paper from STARTER_CORPUS
        citation = fallback_agent.get_citation_by_pmid("31645765")
        assert citation is not None
        assert citation.relevance_score == 1.0
        assert "AMG 510" in citation.title or "KRAS" in citation.title

    def test_lookup_unknown_pmid_returns_none(self, fallback_agent):
        assert fallback_agent.get_citation_by_pmid("999999999") is None
