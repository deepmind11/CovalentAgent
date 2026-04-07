"""MCP-style tool definitions for chemistry capabilities."""

from __future__ import annotations

TOOLS: list[dict] = [
    {
        "name": "select_warheads",
        "description": (
            "Select and rank warhead candidates for a given residue type. "
            "Returns scored recommendations based on reactivity, selectivity, "
            "and structural context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "residue_type": {
                    "type": "string",
                    "description": (
                        "Target residue type, e.g. cysteine, lysine, serine"
                    ),
                },
                "ligandability_score": {
                    "type": "number",
                    "description": "Ligandability score from target analysis (0-1)",
                },
                "structural_context": {
                    "type": "string",
                    "description": "Structural context around the target residue",
                },
                "protein_name": {
                    "type": "string",
                    "description": "Target protein name",
                },
            },
            "required": ["residue_type", "protein_name"],
        },
    },
    {
        "name": "predict_properties",
        "description": (
            "Predict drug-likeness, ADMET, and physicochemical properties "
            "for a list of candidate molecules specified by SMILES."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "smiles": {
                                "type": "string",
                                "description": "SMILES string of the molecule",
                            },
                            "name": {
                                "type": "string",
                                "description": "Name or identifier for the molecule",
                            },
                            "scaffold_type": {
                                "type": "string",
                                "description": "Scaffold classification",
                            },
                            "warhead_class": {
                                "type": "string",
                                "description": "Warhead class used in the molecule",
                            },
                        },
                        "required": ["smiles"],
                    },
                    "description": "List of candidate molecules to evaluate",
                },
            },
            "required": ["candidates"],
        },
    },
    {
        "name": "validate_smiles",
        "description": (
            "Validate whether a SMILES string represents a chemically valid "
            "molecule using RDKit. Returns validity status and basic info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string to validate",
                },
            },
            "required": ["smiles"],
        },
    },
    {
        "name": "compute_descriptors",
        "description": (
            "Compute molecular descriptors for a SMILES string including "
            "molecular weight, LogP, hydrogen bond donors/acceptors, TPSA, "
            "and rotatable bonds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": "SMILES string of the molecule",
                },
            },
            "required": ["smiles"],
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
    if tool_name == "select_warheads":
        return await _execute_select_warheads(args)
    if tool_name == "predict_properties":
        return await _execute_predict_properties(args)
    if tool_name == "validate_smiles":
        return _execute_validate_smiles(args)
    if tool_name == "compute_descriptors":
        return _execute_compute_descriptors(args)

    return {"error": f"Unknown tool: {tool_name}"}


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


async def _execute_select_warheads(args: dict) -> dict:
    """Run the WarheadSelector agent on the given residue type."""
    from covalent_agent.schemas import WarheadSelectionInput

    input_model = WarheadSelectionInput(
        residue_type=args["residue_type"],
        ligandability_score=args.get("ligandability_score", 0.5),
        structural_context=args.get("structural_context", ""),
        protein_name=args["protein_name"],
    )

    try:
        from covalent_agent.agents.warhead_selector import WarheadSelectorAgent

        agent = WarheadSelectorAgent()
        result = await agent.run(input_model)
        return result.model_dump()
    except ImportError:
        # Agent not yet implemented; return data from warhead library
        from covalent_agent.data.warhead_library import WarheadLibrary

        library = WarheadLibrary()
        warheads = library.get_warheads_for_residue(args["residue_type"])
        ligandability = args.get("ligandability_score", 0.5)

        recommendations = []
        for w in warheads:
            score = library.score_warhead_for_context(
                w, args["residue_type"], ligandability
            )
            recommendations.append(
                {
                    "warhead_class": w["name"],
                    "smarts": w["smarts"],
                    "reactivity": w["reactivity"],
                    "selectivity": w["selectivity"],
                    "score": score,
                    "rationale": w.get("notes", ""),
                    "examples": w.get("examples", []),
                    "mechanism": w.get("mechanism", ""),
                }
            )

        recommendations.sort(key=lambda r: r["score"], reverse=True)

        return {
            "target_residue": args["residue_type"],
            "recommendations": recommendations,
            "status": "from_library",
        }


async def _execute_predict_properties(args: dict) -> dict:
    """Run the PropertyPredictor agent on candidate molecules."""
    from covalent_agent.schemas import CandidateMolecule, PropertyPredictionInput

    candidates = [
        CandidateMolecule(
            smiles=c["smiles"],
            name=c.get("name", ""),
            scaffold_type=c.get("scaffold_type", ""),
            warhead_class=c.get("warhead_class", ""),
        )
        for c in args["candidates"]
    ]

    input_model = PropertyPredictionInput(candidates=candidates)

    try:
        from covalent_agent.agents.property_predictor import PropertyPredictorAgent

        agent = PropertyPredictorAgent()
        result = await agent.run(input_model)
        return result.model_dump()
    except ImportError:
        # Agent not yet available; compute basic RDKit descriptors as fallback
        predictions = []
        for cand in candidates:
            descriptors = _execute_compute_descriptors({"smiles": cand.smiles})
            if descriptors.get("valid"):
                predictions.append(
                    {
                        "smiles": cand.smiles,
                        "molecular_weight": descriptors.get("molecular_weight", 0.0),
                        "logp": descriptors.get("logp", 0.0),
                        "descriptors": descriptors,
                        "status": "basic_descriptors_only",
                    }
                )
            else:
                predictions.append(
                    {
                        "smiles": cand.smiles,
                        "error": descriptors.get("error", "Invalid SMILES"),
                        "status": "invalid",
                    }
                )
        return {"predictions": predictions, "status": "fallback_rdkit"}


def _execute_validate_smiles(args: dict) -> dict:
    """Validate a SMILES string using RDKit."""
    smiles = args["smiles"]

    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "error": "RDKit could not parse this SMILES string",
            }

        return {
            "smiles": smiles,
            "valid": True,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "molecular_formula": Chem.rdMolDescriptors.CalcMolFormula(mol),
        }

    except ImportError:
        return {
            "smiles": smiles,
            "valid": None,
            "error": "RDKit not installed; cannot validate SMILES",
        }


def _execute_compute_descriptors(args: dict) -> dict:
    """Compute molecular descriptors for a SMILES string using RDKit."""
    smiles = args["smiles"]

    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "smiles": smiles,
                "valid": False,
                "error": "RDKit could not parse this SMILES string",
            }

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        tpsa = rdMolDescriptors.CalcTPSA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        rings = rdMolDescriptors.CalcNumRings(mol)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

        # Lipinski's rule of five violations
        lipinski_violations = sum([
            mw > 500,
            logp > 5,
            hbd > 5,
            hba > 10,
        ])

        return {
            "smiles": smiles,
            "valid": True,
            "canonical_smiles": Chem.MolToSmiles(mol),
            "molecular_weight": round(mw, 2),
            "logp": round(logp, 2),
            "num_h_donors": hbd,
            "num_h_acceptors": hba,
            "tpsa": round(tpsa, 2),
            "num_rotatable_bonds": rotatable,
            "num_rings": rings,
            "num_aromatic_rings": aromatic_rings,
            "lipinski_violations": lipinski_violations,
            "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
        }

    except ImportError:
        return {
            "smiles": smiles,
            "valid": None,
            "error": "RDKit not installed; cannot compute descriptors",
        }
