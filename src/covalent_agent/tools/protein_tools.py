"""MCP-style tool definitions for protein analysis capabilities."""

from __future__ import annotations

from covalent_agent.data.loaders import lookup_residue as _lookup_residue

TOOLS: list[dict] = [
    {
        "name": "analyze_target",
        "description": (
            "Analyze a protein target for covalent drug design opportunities. "
            "Evaluates ligandability, conservation, structural context, and "
            "known drugs for the specified protein and residue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protein_name": {
                    "type": "string",
                    "description": "Protein name, e.g. KRAS",
                },
                "residue": {
                    "type": "string",
                    "description": "Target residue, e.g. C12",
                },
                "indication": {
                    "type": "string",
                    "description": "Disease indication (optional)",
                },
            },
            "required": ["protein_name", "residue"],
        },
    },
    {
        "name": "lookup_residue",
        "description": (
            "Look up a protein/residue combination in the reactive residues "
            "database. Returns druggability data, known drugs, and notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protein_name": {
                    "type": "string",
                    "description": "Protein name, e.g. KRAS",
                },
                "residue": {
                    "type": "string",
                    "description": "Residue identifier, e.g. C12",
                },
            },
            "required": ["protein_name", "residue"],
        },
    },
    {
        "name": "fetch_protein_info",
        "description": (
            "Fetch protein information from UniProt by protein name. "
            "Returns UniProt ID, gene name, organism, function summary, "
            "and sequence length."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protein_name": {
                    "type": "string",
                    "description": "Protein name or gene symbol, e.g. KRAS",
                },
            },
            "required": ["protein_name"],
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
    if tool_name == "analyze_target":
        return await _execute_analyze_target(args)
    if tool_name == "lookup_residue":
        return _execute_lookup_residue(args)
    if tool_name == "fetch_protein_info":
        return await _execute_fetch_protein_info(args)

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


async def _execute_analyze_target(args: dict) -> dict:
    """Run the TargetAnalyst agent on the given protein/residue."""
    from covalent_agent.schemas import TargetAnalysisInput

    input_model = TargetAnalysisInput(
        protein_name=args["protein_name"],
        residue=args["residue"],
        indication=args.get("indication", ""),
    )

    # Lazy import to avoid circular dependency at module level
    try:
        from covalent_agent.agents.target_analyst import TargetAnalystAgent

        agent = TargetAnalystAgent()
        result = await agent.run(input_model)
        return result.model_dump()
    except ImportError:
        # Agent not yet implemented; return data from the residue database
        db_entry = _lookup_residue(args["protein_name"], args["residue"])
        if db_entry:
            return {
                "protein_name": db_entry["protein"],
                "uniprot_id": db_entry.get("uniprot", ""),
                "residue": args["residue"],
                "known_drugs": db_entry.get("approved_drugs", []),
                "indication": db_entry.get("indication", ""),
                "notes": db_entry.get("notes", ""),
                "status": "partial_from_database",
            }
        return {
            "protein_name": args["protein_name"],
            "residue": args["residue"],
            "status": "not_found",
            "error": "Protein/residue not in database and TargetAnalyst not available",
        }


def _execute_lookup_residue(args: dict) -> dict:
    """Look up a residue in the reactive residues database."""
    entry = _lookup_residue(args["protein_name"], args["residue"])
    if entry:
        return {"found": True, **entry}
    return {
        "found": False,
        "protein_name": args["protein_name"],
        "residue": args["residue"],
        "message": "No matching entry in the reactive residues database",
    }


async def _execute_fetch_protein_info(args: dict) -> dict:
    """Fetch protein info from UniProt REST API.

    Uses the UniProt search API to look up the protein by gene name,
    returning key metadata. Falls back to local database on network errors.
    """
    protein_name = args["protein_name"]
    url = (
        f"https://rest.uniprot.org/uniprotkb/search"
        f"?query=gene_exact:{protein_name}+AND+organism_id:9606"
        f"&fields=accession,gene_names,protein_name,length,organism_name"
        f"&format=json&size=1"
    )

    try:
        import httpx

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        results = data.get("results", [])
        if not results:
            return {
                "protein_name": protein_name,
                "found": False,
                "message": "No human protein found in UniProt for this name",
            }

        entry = results[0]
        return {
            "found": True,
            "uniprot_id": entry.get("primaryAccession", ""),
            "gene_names": entry.get("genes", [{}])[0].get("geneName", {}).get("value", ""),
            "protein_name": entry.get("proteinDescription", {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value", protein_name),
            "length": entry.get("sequence", {}).get("length", 0),
            "organism": entry.get("organism", {}).get("scientificName", ""),
        }

    except Exception as exc:
        # Fall back to local database
        from covalent_agent.data.loaders import lookup_protein

        db_entry = lookup_protein(protein_name)
        if db_entry:
            return {
                "found": True,
                "uniprot_id": db_entry.get("uniprot", ""),
                "protein_name": protein_name,
                "source": "local_database",
                "notes": db_entry.get("notes", ""),
            }
        return {
            "found": False,
            "protein_name": protein_name,
            "error": f"UniProt lookup failed and no local entry: {exc}",
        }
