"""ReporterAgent: generates a structured FinalReport ranking all candidate molecules.

Takes all intermediate results from the covalent drug design pipeline
(target analysis, warhead selection, molecule design, property prediction,
literature search) and produces a unified FinalReport with ranked candidates,
methodology summary, and citations.
"""

from __future__ import annotations

import logging

from covalent_agent.config import settings
from covalent_agent.schemas import (
    Citation,
    FinalReport,
    LiteratureResult,
    MoleculeDesignResult,
    MoleculeProperties,
    PropertyPredictionResult,
    RankedCandidate,
    TargetAnalysisInput,
    TargetAnalysisResult,
    WarheadSelectionResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Passing threshold for composite score
# ---------------------------------------------------------------------------
_PASSING_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _format_admet(admet_profile) -> str:
    """Format an ADMETProfile as a human-readable summary string."""
    return (
        f"Absorption: {admet_profile.absorption_score:.1f}, "
        f"Distribution: {admet_profile.distribution_score:.1f}, "
        f"Metabolism: {admet_profile.metabolism_score:.1f}, "
        f"Excretion: {admet_profile.excretion_score:.1f}, "
        f"Toxicity risk: {admet_profile.toxicity_risk:.1f}"
    )


def _find_literature_support(
    warhead_class: str,
    citations: list[Citation],
) -> str:
    """Find citations mentioning this warhead class and return a summary string.

    Searches title, abstract, and journal fields for the warhead class name.
    Returns a comma-separated list of matching citation titles, or a fallback
    message when no matches are found.
    """
    if not citations:
        return "No literature citations available."

    warhead_lower = warhead_class.lower()
    matching: list[str] = []

    for citation in citations:
        searchable = (
            f"{citation.title} {citation.abstract} {citation.journal}"
        ).lower()
        if warhead_lower in searchable:
            matching.append(citation.title)

    if not matching:
        return f"No specific literature found for {warhead_class} warhead class."

    return f"Supported by: {'; '.join(matching)}"


def _build_candidate_rationale(
    warhead_class: str,
    warhead_recommendations: list,
    properties: MoleculeProperties,
) -> str:
    """Combine warhead recommendation rationale with property assessment."""
    # Find the matching warhead recommendation rationale
    warhead_rationale = ""
    for rec in warhead_recommendations:
        if rec.warhead_class.lower() == warhead_class.lower():
            warhead_rationale = rec.rationale
            break

    # Build property assessment
    drug_likeness_label = (
        "excellent" if properties.drug_likeness_score >= 0.75
        else "good" if properties.drug_likeness_score >= 0.5
        else "moderate" if properties.drug_likeness_score >= 0.25
        else "poor"
    )

    sa_label = (
        "easy" if properties.synthetic_accessibility <= 3.0
        else "moderate" if properties.synthetic_accessibility <= 5.0
        else "difficult" if properties.synthetic_accessibility <= 7.0
        else "very difficult"
    )

    property_assessment = (
        f"Drug-likeness: {drug_likeness_label} "
        f"(score {properties.drug_likeness_score:.2f}). "
        f"QED: {properties.qed_score:.2f}. "
        f"Synthetic accessibility: {sa_label} "
        f"(SA score {properties.synthetic_accessibility:.1f}). "
        f"Toxicity risk: {properties.admet.toxicity_risk:.2f}."
    )

    if warhead_rationale:
        return f"{warhead_rationale} {property_assessment}"
    return property_assessment


def _build_methodology_template(
    protein: str,
    residue: str,
    score: float,
    n_warheads: int,
    top_warhead: str,
    n_candidates: int,
    n_passing: int,
    n_citations: int,
) -> str:
    """Deterministic fallback methodology summary when Claude API is unavailable."""
    return (
        f"This analysis employed a multi-agent pipeline for covalent drug "
        f"design targeting {protein} {residue}. "
        f"Target analysis scored ligandability at {score:.2f} using ESM-2 "
        f"protein language model embeddings. "
        f"{n_warheads} warhead classes were evaluated, with {top_warhead} "
        f"selected as the primary recommendation. "
        f"{n_candidates} candidate molecules were generated using "
        f"fragment-based design with warhead attachment, "
        f"and evaluated for drug-likeness (QED), ADMET properties, and "
        f"synthetic accessibility. "
        f"{n_passing} candidates passed the composite scoring threshold. "
        f"Literature search identified {n_citations} relevant publications "
        f"supporting the design rationale."
    )


# ---------------------------------------------------------------------------
# ReporterAgent
# ---------------------------------------------------------------------------


class ReporterAgent:
    """Generate a structured FinalReport ranking all candidate molecules.

    Standalone agent class: takes all intermediate pipeline results and
    returns a ``FinalReport``. No LangGraph dependency; can be tested in
    isolation.
    """

    async def run(
        self,
        target_analysis: TargetAnalysisResult,
        warhead_selection: WarheadSelectionResult,
        molecule_design: MoleculeDesignResult,
        property_prediction: PropertyPredictionResult,
        literature: LiteratureResult,
        target_input: TargetAnalysisInput,
    ) -> FinalReport:
        """Execute the reporting workflow.

        Steps:
            1. Merge molecule data with property predictions by SMILES.
            2. Rank candidates by composite score (descending).
            3. Build RankedCandidate entries with full rationale.
            4. Generate methodology summary (Claude or template fallback).
            5. Assemble and return the FinalReport.
        """
        # 1. Merge molecules with their property predictions
        properties_by_smiles: dict[str, MoleculeProperties] = {
            prop.smiles: prop for prop in property_prediction.predictions
        }

        merged: list[tuple] = []
        for candidate in molecule_design.candidates:
            props = properties_by_smiles.get(candidate.smiles)
            if props is None:
                logger.warning(
                    "No property prediction found for candidate '%s' "
                    "(SMILES: %s); skipping from ranking.",
                    candidate.name,
                    candidate.smiles,
                )
                continue
            merged.append((candidate, props))

        # 2. Sort by overall_score descending
        merged.sort(key=lambda pair: pair[1].overall_score, reverse=True)

        # 3. Build ranked candidates
        citations = literature.citations if literature.citations else []
        ranked_candidates: list[RankedCandidate] = []

        for rank_index, (candidate, props) in enumerate(merged, start=1):
            ranked_candidates.append(
                RankedCandidate(
                    rank=rank_index,
                    smiles=candidate.smiles,
                    name=candidate.name,
                    composite_score=props.overall_score,
                    warhead_class=candidate.warhead_class,
                    drug_likeness=props.drug_likeness_score,
                    qed_score=props.qed_score,
                    admet_summary=_format_admet(props.admet),
                    synthetic_accessibility=props.synthetic_accessibility,
                    literature_support=_find_literature_support(
                        candidate.warhead_class, citations,
                    ),
                    rationale=_build_candidate_rationale(
                        candidate.warhead_class,
                        warhead_selection.recommendations,
                        props,
                    ),
                )
            )

        # Counts
        num_generated = len(molecule_design.candidates)
        num_passing = sum(
            1 for _, props in merged
            if props.overall_score > _PASSING_THRESHOLD
        )

        # 4. Generate methodology summary
        top_warhead = (
            warhead_selection.recommendations[0].warhead_class
            if warhead_selection.recommendations
            else "N/A"
        )

        methodology_summary = await self._generate_methodology(
            protein=target_input.protein_name,
            residue=target_input.residue,
            indication=target_input.indication,
            ligandability_score=target_analysis.ligandability_score,
            n_warheads=len(warhead_selection.recommendations),
            top_warhead=top_warhead,
            n_candidates=num_generated,
            n_passing=num_passing,
            n_citations=len(citations),
        )

        # Ligandability assessment summary
        ligandability_assessment = (
            f"Ligandability score: {target_analysis.ligandability_score:.2f}. "
            f"{target_analysis.rationale}"
        )

        # 5. Assemble the final report
        return FinalReport(
            target_protein=target_input.protein_name,
            target_residue=target_input.residue,
            indication=target_input.indication,
            ligandability_assessment=ligandability_assessment,
            num_candidates_generated=num_generated,
            num_candidates_passing=num_passing,
            ranked_candidates=ranked_candidates,
            methodology_summary=methodology_summary,
            citations=citations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_methodology(
        self,
        protein: str,
        residue: str,
        indication: str,
        ligandability_score: float,
        n_warheads: int,
        top_warhead: str,
        n_candidates: int,
        n_passing: int,
        n_citations: int,
    ) -> str:
        """Generate a methodology summary using Claude, with template fallback."""
        if not settings.anthropic_api_key:
            return _build_methodology_template(
                protein=protein,
                residue=residue,
                score=ligandability_score,
                n_warheads=n_warheads,
                top_warhead=top_warhead,
                n_candidates=n_candidates,
                n_passing=n_passing,
                n_citations=n_citations,
            )

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

            prompt = (
                "You are a computational chemistry expert writing a methodology "
                "section for a covalent drug design report. Write a concise 2-3 "
                "paragraph summary covering the following pipeline steps. "
                "Do not use em dashes.\n\n"
                f"Target: {protein} {residue}"
                f"{f', indication: {indication}' if indication else ''}\n"
                f"Target analysis: ESM-2 protein language model embeddings "
                f"scored ligandability at {ligandability_score:.2f}\n"
                f"Warhead selection: {n_warheads} warhead classes evaluated, "
                f"top recommendation: {top_warhead}\n"
                f"Molecule design: {n_candidates} candidates generated via "
                f"fragment-based design with warhead attachment\n"
                f"Property prediction: QED drug-likeness, ADMET profiling, "
                f"synthetic accessibility scoring\n"
                f"Literature search: {n_citations} relevant publications found\n"
                f"Results: {n_passing} of {n_candidates} candidates passed "
                f"the composite scoring threshold (>{_PASSING_THRESHOLD})\n\n"
                "Be concise, scientifically accurate, and professional."
            )

            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        except Exception:
            logger.exception(
                "Claude methodology generation failed; using template fallback"
            )
            return _build_methodology_template(
                protein=protein,
                residue=residue,
                score=ligandability_score,
                n_warheads=n_warheads,
                top_warhead=top_warhead,
                n_candidates=n_candidates,
                n_passing=n_passing,
                n_citations=n_citations,
            )
