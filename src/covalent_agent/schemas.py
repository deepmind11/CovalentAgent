"""Pydantic models for inter-agent communication in CovalentAgent.

Every agent takes a structured input model and returns a structured output model.
The SupervisorState ties them together for LangGraph orchestration.
"""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Target Analysis
# ---------------------------------------------------------------------------

class TargetAnalysisInput(BaseModel):
    """Input to the TargetAnalyst agent."""

    protein_name: str = Field(description="Protein name, e.g. KRAS")
    residue: str = Field(description="Target residue, e.g. C12")
    indication: str = Field(default="", description="Disease indication")


class TargetAnalysisResult(BaseModel):
    """Output from the TargetAnalyst agent."""

    protein_name: str
    uniprot_id: str
    residue_type: str
    residue_position: int
    ligandability_score: float = Field(ge=0.0, le=1.0)
    conservation_score: float = Field(ge=0.0, le=1.0)
    structural_context: str
    known_drugs: list[str] = Field(default_factory=list)
    esm_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


# ---------------------------------------------------------------------------
# Warhead Selection
# ---------------------------------------------------------------------------

class WarheadRecommendation(BaseModel):
    """A single warhead recommendation with scored rationale."""

    warhead_class: str
    smarts: str
    reactivity: str
    selectivity: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    examples: list[str] = Field(default_factory=list)
    mechanism: str


class WarheadSelectionInput(BaseModel):
    """Input to the WarheadSelector agent."""

    residue_type: str
    ligandability_score: float
    structural_context: str
    protein_name: str


class WarheadSelectionResult(BaseModel):
    """Output from the WarheadSelector agent."""

    target_residue: str
    recommendations: list[WarheadRecommendation]


# ---------------------------------------------------------------------------
# Molecule Design
# ---------------------------------------------------------------------------

class CandidateMolecule(BaseModel):
    """A generated candidate molecule with computed descriptors."""

    smiles: str
    name: str
    scaffold_type: str
    warhead_class: str
    molecular_weight: float = 0.0
    logp: float = 0.0
    num_h_donors: int = 0
    num_h_acceptors: int = 0
    num_rotatable_bonds: int = 0
    tpsa: float = 0.0


class MoleculeDesignInput(BaseModel):
    """Input to the MoleculeDesigner agent."""

    warhead_recommendations: list[WarheadRecommendation]
    target_protein: str
    target_residue: str
    num_candidates: int = Field(default=5, ge=1, le=20)


class MoleculeDesignResult(BaseModel):
    """Output from the MoleculeDesigner agent."""

    candidates: list[CandidateMolecule]
    design_rationale: str


# ---------------------------------------------------------------------------
# Property Prediction
# ---------------------------------------------------------------------------

class ADMETProfile(BaseModel):
    """Predicted ADMET (absorption, distribution, metabolism, excretion, toxicity)."""

    absorption_score: float = Field(default=0.5, ge=0.0, le=1.0)
    distribution_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metabolism_score: float = Field(default=0.5, ge=0.0, le=1.0)
    excretion_score: float = Field(default=0.5, ge=0.0, le=1.0)
    toxicity_risk: float = Field(default=0.3, ge=0.0, le=1.0)


class MoleculeProperties(BaseModel):
    """Full property profile for a candidate molecule."""

    smiles: str
    drug_likeness_score: float = Field(ge=0.0, le=1.0)
    qed_score: float = Field(ge=0.0, le=1.0)
    lipinski_violations: int = Field(ge=0)
    admet: ADMETProfile
    synthetic_accessibility: float = Field(ge=1.0, le=10.0)
    overall_score: float = Field(ge=0.0, le=1.0)


class PropertyPredictionInput(BaseModel):
    """Input to the PropertyPredictor agent."""

    candidates: list[CandidateMolecule]


class PropertyPredictionResult(BaseModel):
    """Output from the PropertyPredictor agent."""

    predictions: list[MoleculeProperties]


# ---------------------------------------------------------------------------
# Literature RAG
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A literature citation with relevance score."""

    title: str
    authors: list[str] = Field(default_factory=list)
    journal: str = ""
    year: int = 0
    pmid: str = ""
    doi: str = ""
    abstract: str = ""
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)


class LiteratureQuery(BaseModel):
    """Input to the LiteratureRAG agent."""

    query: str
    protein_name: str = ""
    warhead_class: str = ""
    max_results: int = Field(default=5, ge=1, le=20)


class LiteratureResult(BaseModel):
    """Output from the LiteratureRAG agent."""

    query: str
    citations: list[Citation]
    summary: str
    key_findings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------

class RankedCandidate(BaseModel):
    """A candidate molecule with composite score and full rationale."""

    rank: int
    smiles: str
    name: str
    composite_score: float = Field(ge=0.0, le=1.0)
    warhead_class: str
    drug_likeness: float
    qed_score: float
    admet_summary: str
    synthetic_accessibility: float
    literature_support: str
    rationale: str


class FinalReport(BaseModel):
    """The final output report ranking all candidate molecules."""

    target_protein: str
    target_residue: str
    indication: str
    ligandability_assessment: str
    num_candidates_generated: int
    num_candidates_passing: int
    ranked_candidates: list[RankedCandidate]
    methodology_summary: str
    citations: list[Citation]
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ---------------------------------------------------------------------------
# Supervisor State (LangGraph)
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    """State dictionary maintained by the LangGraph supervisor.

    Fields start as None and are populated as each agent completes its step.
    """

    target_input: TargetAnalysisInput
    target_analysis: TargetAnalysisResult | None
    warhead_selection: WarheadSelectionResult | None
    molecule_design: MoleculeDesignResult | None
    property_prediction: PropertyPredictionResult | None
    literature: LiteratureResult | None
    final_report: FinalReport | None
    current_step: str
    error: str | None
