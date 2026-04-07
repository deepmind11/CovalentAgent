"""MCP-style tool definitions for literature search capabilities."""

from __future__ import annotations

TOOLS: list[dict] = [
    {
        "name": "search_literature",
        "description": (
            "Search the covalent drug design literature corpus using "
            "retrieval-augmented generation. Returns relevant citations "
            "with a synthesized summary and key findings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "protein_name": {
                    "type": "string",
                    "description": "Filter by protein name (optional)",
                },
                "warhead_class": {
                    "type": "string",
                    "description": "Filter by warhead class (optional)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of citations to return (1-20, default 5)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_citation",
        "description": (
            "Get full details for a specific paper by its PubMed ID (PMID). "
            "Returns title, authors, journal, year, and abstract."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pmid": {
                    "type": "string",
                    "description": "PubMed ID of the paper, e.g. 27309814",
                },
            },
            "required": ["pmid"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution dispatcher
# ---------------------------------------------------------------------------


async def execute_tool(tool_name: str, args: dict) -> dict:
    """Dispatch a tool call to the appropriate function.

    Args:
        tool_name: one of the tool names defined in TOOLS.
        args: dict of arguments matching the tool's input_schema.

    Returns:
        dict with the tool result.
    """
    if tool_name == "search_literature":
        return await _execute_search_literature(args)
    if tool_name == "get_citation":
        return _execute_get_citation(args)

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


async def _execute_search_literature(args: dict) -> dict:
    """Run the LiteratureRAGAgent with the given query."""
    from covalent_agent.agents.literature_rag import LiteratureRAGAgent
    from covalent_agent.schemas import LiteratureQuery

    input_model = LiteratureQuery(
        query=args["query"],
        protein_name=args.get("protein_name", ""),
        warhead_class=args.get("warhead_class", ""),
        max_results=args.get("max_results", 5),
    )

    agent = LiteratureRAGAgent()
    result = await agent.run(input_model)
    return result.model_dump()


def _execute_get_citation(args: dict) -> dict:
    """Look up a specific citation by PMID."""
    from covalent_agent.agents.literature_rag import LiteratureRAGAgent

    agent = LiteratureRAGAgent()
    citation = agent.get_citation_by_pmid(args["pmid"])

    if citation is not None:
        return {"found": True, **citation.model_dump()}

    return {
        "found": False,
        "pmid": args["pmid"],
        "message": "No paper with this PMID in the local corpus",
    }
