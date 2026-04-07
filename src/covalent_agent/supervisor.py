"""Supervisor: LangGraph state machine orchestrating the CovalentAgent pipeline.

Runs each specialist agent in sequence, passing structured outputs forward
through a shared ``SupervisorState``. Any node failure short-circuits the
pipeline and surfaces the error via the ``error`` field.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from covalent_agent.agents.literature_rag import LiteratureRAGAgent
from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent
from covalent_agent.agents.property_predictor import PropertyPredictorAgent
from covalent_agent.agents.reporter import ReporterAgent
from covalent_agent.agents.target_analyst import TargetAnalystAgent
from covalent_agent.agents.warhead_selector import WarheadSelectorAgent
from covalent_agent.schemas import (
    FinalReport,
    LiteratureQuery,
    MoleculeDesignInput,
    PropertyPredictionInput,
    SupervisorState,
    TargetAnalysisInput,
    WarheadSelectionInput,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def analyze_target(state: SupervisorState) -> dict:
    """Run the TargetAnalyst agent on the initial target input."""
    try:
        target_input: TargetAnalysisInput = state["target_input"]
        agent = TargetAnalystAgent()
        result = await agent.run(target_input)
        logger.info(
            "Target analysis complete: %s (ligandability=%.2f)",
            result.protein_name,
            result.ligandability_score,
        )
        return {"target_analysis": result, "current_step": "warhead_selection"}
    except Exception as exc:
        logger.exception("Target analysis failed")
        return {"error": str(exc), "current_step": "target_analysis"}


async def select_warheads(state: SupervisorState) -> dict:
    """Run the WarheadSelector agent using target analysis results."""
    try:
        target_analysis = state["target_analysis"]
        input_data = WarheadSelectionInput(
            residue_type=target_analysis.residue_type,
            ligandability_score=target_analysis.ligandability_score,
            structural_context=target_analysis.structural_context,
            protein_name=target_analysis.protein_name,
        )
        agent = WarheadSelectorAgent()
        result = await agent.run(input_data)
        logger.info(
            "Warhead selection complete: %d recommendations",
            len(result.recommendations),
        )
        return {"warhead_selection": result, "current_step": "molecule_design"}
    except Exception as exc:
        logger.exception("Warhead selection failed")
        return {"error": str(exc), "current_step": "warhead_selection"}


async def design_molecules(state: SupervisorState) -> dict:
    """Run the MoleculeDesigner agent using warhead recommendations."""
    try:
        warhead_selection = state["warhead_selection"]
        target_analysis = state["target_analysis"]
        input_data = MoleculeDesignInput(
            warhead_recommendations=warhead_selection.recommendations,
            target_protein=target_analysis.protein_name,
            target_residue=f"{target_analysis.residue_type[0].upper()}{target_analysis.residue_position}",
        )
        agent = MoleculeDesignerAgent()
        result = await agent.run(input_data)
        logger.info(
            "Molecule design complete: %d candidates",
            len(result.candidates),
        )
        return {"molecule_design": result, "current_step": "property_prediction"}
    except Exception as exc:
        logger.exception("Molecule design failed")
        return {"error": str(exc), "current_step": "molecule_design"}


async def predict_properties(state: SupervisorState) -> dict:
    """Run the PropertyPredictor agent on designed candidates."""
    try:
        molecule_design = state["molecule_design"]
        input_data = PropertyPredictionInput(
            candidates=molecule_design.candidates,
        )
        agent = PropertyPredictorAgent()
        result = await agent.run(input_data)
        logger.info(
            "Property prediction complete: %d predictions",
            len(result.predictions),
        )
        return {"property_prediction": result, "current_step": "literature_search"}
    except Exception as exc:
        logger.exception("Property prediction failed")
        return {"error": str(exc), "current_step": "property_prediction"}


async def search_literature(state: SupervisorState) -> dict:
    """Run the LiteratureRAG agent combining protein, warhead, and indication context."""
    try:
        target_analysis = state["target_analysis"]
        warhead_selection = state["warhead_selection"]
        target_input: TargetAnalysisInput = state["target_input"]

        warhead_classes = [
            rec.warhead_class for rec in warhead_selection.recommendations
        ]
        warhead_str = ", ".join(warhead_classes[:3]) if warhead_classes else ""

        query_parts = [
            f"covalent inhibitors targeting {target_analysis.protein_name}",
            f"{target_analysis.residue_type} residue",
        ]
        if warhead_str:
            query_parts.append(f"warhead classes: {warhead_str}")
        if target_input.indication:
            query_parts.append(f"indication: {target_input.indication}")

        input_data = LiteratureQuery(
            query=" ".join(query_parts),
            protein_name=target_analysis.protein_name,
            warhead_class=warhead_classes[0] if warhead_classes else "",
        )
        agent = LiteratureRAGAgent()
        result = await agent.run(input_data)
        logger.info(
            "Literature search complete: %d citations",
            len(result.citations),
        )
        return {"literature": result, "current_step": "report_generation"}
    except Exception as exc:
        logger.exception("Literature search failed")
        return {"error": str(exc), "current_step": "literature_search"}


async def generate_report(state: SupervisorState) -> dict:
    """Run the Reporter agent to produce the final ranked report."""
    try:
        agent = ReporterAgent()
        result = await agent.run(
            target_analysis=state["target_analysis"],
            warhead_selection=state["warhead_selection"],
            molecule_design=state["molecule_design"],
            property_prediction=state["property_prediction"],
            literature=state["literature"],
            target_input=state["target_input"],
        )
        logger.info(
            "Report generation complete: %d ranked candidates",
            len(result.ranked_candidates),
        )
        return {"final_report": result, "current_step": "done"}
    except Exception as exc:
        logger.exception("Report generation failed")
        return {"error": str(exc), "current_step": "report_generation"}


# ---------------------------------------------------------------------------
# Conditional edge: check for errors after each node
# ---------------------------------------------------------------------------

def _check_error(state: SupervisorState) -> str:
    """Route to END if the state contains an error, otherwise continue."""
    if state.get("error"):
        return "end"
    return "continue"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class CovalentAgentPipeline:
    """LangGraph-based orchestrator for the full covalent drug design pipeline.

    Nodes execute in strict sequence::

        analyze_target -> select_warheads -> design_molecules
        -> predict_properties -> search_literature -> generate_report

    Each node catches exceptions and writes to ``state["error"]``.
    A conditional edge after every node short-circuits to END on failure.
    """

    def __init__(self) -> None:
        self.graph = self._build_graph()

    async def run(
        self,
        target: str,
        residue: str,
        indication: str = "",
    ) -> FinalReport:
        """Run the full pipeline and return a FinalReport.

        Args:
            target: Protein name (e.g. ``"KRAS"``).
            residue: Target residue (e.g. ``"C12"``).
            indication: Optional disease indication.

        Returns:
            A ``FinalReport`` with ranked candidate molecules.

        Raises:
            RuntimeError: If any pipeline step fails.
        """
        initial_state: SupervisorState = {
            "target_input": TargetAnalysisInput(
                protein_name=target,
                residue=residue,
                indication=indication,
            ),
            "target_analysis": None,
            "warhead_selection": None,
            "molecule_design": None,
            "property_prediction": None,
            "literature": None,
            "final_report": None,
            "current_step": "target_analysis",
            "error": None,
        }

        result = await self.graph.ainvoke(initial_state)

        if result.get("error"):
            raise RuntimeError(
                f"Pipeline failed at {result['current_step']}: {result['error']}"
            )

        return result["final_report"]

    @staticmethod
    def _build_graph() -> StateGraph:
        """Construct and compile the LangGraph state machine."""
        graph = StateGraph(SupervisorState)

        # -- Add nodes --
        graph.add_node("analyze_target", analyze_target)
        graph.add_node("select_warheads", select_warheads)
        graph.add_node("design_molecules", design_molecules)
        graph.add_node("predict_properties", predict_properties)
        graph.add_node("search_literature", search_literature)
        graph.add_node("generate_report", generate_report)

        # -- Entry point --
        graph.set_entry_point("analyze_target")

        # -- Conditional edges: bail to END on error, otherwise continue --
        graph.add_conditional_edges(
            "analyze_target",
            _check_error,
            {"continue": "select_warheads", "end": END},
        )
        graph.add_conditional_edges(
            "select_warheads",
            _check_error,
            {"continue": "design_molecules", "end": END},
        )
        graph.add_conditional_edges(
            "design_molecules",
            _check_error,
            {"continue": "predict_properties", "end": END},
        )
        graph.add_conditional_edges(
            "predict_properties",
            _check_error,
            {"continue": "search_literature", "end": END},
        )
        graph.add_conditional_edges(
            "search_literature",
            _check_error,
            {"continue": "generate_report", "end": END},
        )

        # -- Final node goes straight to END --
        graph.add_edge("generate_report", END)

        return graph.compile()
