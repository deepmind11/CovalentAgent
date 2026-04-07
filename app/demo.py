"""CovalentAgent Streamlit Demo UI.

Step-by-step visualization of the multi-agent covalent drug design pipeline.
Each agent runs individually so progress can be displayed to the user.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import streamlit as st

from covalent_agent.agents.literature_rag import LiteratureRAGAgent
from covalent_agent.agents.molecule_designer import MoleculeDesignerAgent
from covalent_agent.agents.property_predictor import PropertyPredictorAgent
from covalent_agent.agents.reporter import ReporterAgent
from covalent_agent.agents.target_analyst import TargetAnalystAgent
from covalent_agent.agents.warhead_selector import WarheadSelectorAgent
from covalent_agent.schemas import (
    LiteratureQuery,
    MoleculeDesignInput,
    PropertyPredictionInput,
    TargetAnalysisInput,
    WarheadSelectionInput,
)

# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            return loop.run_until_complete(coro)
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CovalentAgent: AI-Powered Covalent Drug Design",
    page_icon="\U0001f9ea",  # test tube emoji
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("\U0001f9ea CovalentAgent")
    st.markdown("**AI-Powered Covalent Drug Design**")

    st.divider()

    protein_name = st.text_input("Protein name", value="KRAS")
    target_residue = st.text_input("Target residue", value="C12")
    indication = st.text_input("Indication", value="non-small cell lung cancer")

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    st.divider()

    st.info(
        "**What does CovalentAgent do?**\n\n"
        "CovalentAgent orchestrates six specialized AI agents to mirror the "
        "covalent drug discovery workflow:\n\n"
        "1. **Target Analysis** - Score residue ligandability with ESM-2\n"
        "2. **Warhead Selection** - Identify optimal electrophilic warheads\n"
        "3. **Molecule Design** - Generate candidates via fragment assembly\n"
        "4. **Property Prediction** - ADMET, QED, and drug-likeness scoring\n"
        "5. **Literature Search** - RAG over covalent drug design literature\n"
        "6. **Report Generation** - Ranked candidates with full rationale\n"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("CovalentAgent: AI-Powered Covalent Drug Design")
st.markdown(
    "Run the full multi-agent pipeline to identify and rank covalent drug "
    "candidates for your target protein."
)

if not run_button:
    st.markdown("---")
    st.markdown(
        "Configure your target protein in the sidebar and click "
        "**Run Analysis** to begin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

STEPS = [
    ("Analyzing target protein...", 0.0),
    ("Selecting warhead classes...", 0.17),
    ("Designing candidate molecules...", 0.33),
    ("Predicting molecular properties...", 0.50),
    ("Searching literature...", 0.67),
    ("Generating report...", 0.83),
]

progress_bar = st.progress(0.0, text="Starting pipeline...")

# -- Step 1: Target Analysis -------------------------------------------------

progress_bar.progress(STEPS[0][1], text=STEPS[0][0])

with st.status("Step 1/6: Target Analysis", expanded=True) as status_1:
    st.write(f"Analyzing **{protein_name}** residue **{target_residue}**...")

    target_input = TargetAnalysisInput(
        protein_name=protein_name,
        residue=target_residue,
        indication=indication,
    )

    try:
        target_agent = TargetAnalystAgent()
        target_result = run_async(target_agent.run(target_input))

        col1, col2, col3 = st.columns(3)
        col1.metric("Ligandability Score", f"{target_result.ligandability_score:.2f}")
        col2.metric("Conservation Score", f"{target_result.conservation_score:.2f}")
        col3.metric("ESM Confidence", f"{target_result.esm_confidence:.2f}")

        st.markdown(f"**Protein:** {target_result.protein_name} ({target_result.uniprot_id})")
        st.markdown(f"**Residue:** {target_result.residue_type} at position {target_result.residue_position}")

        if target_result.known_drugs:
            st.markdown(f"**Known drugs:** {', '.join(target_result.known_drugs)}")

        with st.expander("Rationale"):
            st.write(target_result.rationale)

        status_1.update(label="Step 1/6: Target Analysis (complete)", state="complete")

    except Exception as exc:
        st.error(f"Target analysis failed: {exc}")
        status_1.update(label="Step 1/6: Target Analysis (failed)", state="error")
        st.stop()

# -- Step 2: Warhead Selection ------------------------------------------------

progress_bar.progress(STEPS[1][1], text=STEPS[1][0])

with st.status("Step 2/6: Warhead Selection", expanded=True) as status_2:
    st.write("Evaluating warhead classes for this target context...")

    warhead_input = WarheadSelectionInput(
        residue_type=target_result.residue_type,
        ligandability_score=target_result.ligandability_score,
        structural_context=target_result.structural_context,
        protein_name=target_result.protein_name,
    )

    try:
        warhead_agent = WarheadSelectorAgent()
        warhead_result = run_async(warhead_agent.run(warhead_input))

        if warhead_result.recommendations:
            warhead_df = pd.DataFrame(
                [
                    {
                        "Warhead Class": rec.warhead_class,
                        "Score": f"{rec.score:.3f}",
                        "Reactivity": rec.reactivity,
                        "Selectivity": rec.selectivity,
                        "Mechanism": rec.mechanism,
                        "Examples": ", ".join(rec.examples) if rec.examples else "N/A",
                    }
                    for rec in warhead_result.recommendations
                ]
            )
            st.dataframe(warhead_df, use_container_width=True, hide_index=True)

            top_rec = warhead_result.recommendations[0]
            st.metric("Top Recommendation", top_rec.warhead_class, delta=f"Score: {top_rec.score:.3f}")

            with st.expander("Warhead Rationales"):
                for rec in warhead_result.recommendations:
                    st.markdown(f"**{rec.warhead_class}:** {rec.rationale}")
        else:
            st.warning("No compatible warhead classes found for this residue type.")

        status_2.update(label="Step 2/6: Warhead Selection (complete)", state="complete")

    except Exception as exc:
        st.error(f"Warhead selection failed: {exc}")
        status_2.update(label="Step 2/6: Warhead Selection (failed)", state="error")
        st.stop()

# -- Step 3: Molecule Design --------------------------------------------------

progress_bar.progress(STEPS[2][1], text=STEPS[2][0])

with st.status("Step 3/6: Molecule Design", expanded=True) as status_3:
    st.write("Generating candidate molecules via fragment-based assembly...")

    residue_letter = target_result.residue_type[0].upper() if target_result.residue_type else "C"
    molecule_input = MoleculeDesignInput(
        warhead_recommendations=warhead_result.recommendations,
        target_protein=target_result.protein_name,
        target_residue=f"{residue_letter}{target_result.residue_position}",
    )

    try:
        molecule_agent = MoleculeDesignerAgent()
        molecule_result = run_async(molecule_agent.run(molecule_input))

        st.metric("Candidates Generated", len(molecule_result.candidates))

        if molecule_result.candidates:
            mol_df = pd.DataFrame(
                [
                    {
                        "Name": c.name,
                        "SMILES": c.smiles,
                        "Scaffold": c.scaffold_type,
                        "Warhead": c.warhead_class,
                        "MW": f"{c.molecular_weight:.1f}" if c.molecular_weight > 0 else "N/A",
                        "LogP": f"{c.logp:.2f}" if c.molecular_weight > 0 else "N/A",
                        "HBD": c.num_h_donors,
                        "HBA": c.num_h_acceptors,
                    }
                    for c in molecule_result.candidates
                ]
            )
            st.dataframe(mol_df, use_container_width=True, hide_index=True)

        with st.expander("Design Rationale"):
            st.write(molecule_result.design_rationale)

        status_3.update(label="Step 3/6: Molecule Design (complete)", state="complete")

    except Exception as exc:
        st.error(f"Molecule design failed: {exc}")
        status_3.update(label="Step 3/6: Molecule Design (failed)", state="error")
        st.stop()

# -- Step 4: Property Prediction ----------------------------------------------

progress_bar.progress(STEPS[3][1], text=STEPS[3][0])

with st.status("Step 4/6: Property Prediction", expanded=True) as status_4:
    st.write("Scoring candidates for drug-likeness, ADMET, and synthetic accessibility...")

    property_input = PropertyPredictionInput(
        candidates=molecule_result.candidates,
    )

    try:
        property_agent = PropertyPredictorAgent()
        property_result = run_async(property_agent.run(property_input))

        if property_result.predictions:
            prop_df = pd.DataFrame(
                [
                    {
                        "SMILES": p.smiles[:50] + ("..." if len(p.smiles) > 50 else ""),
                        "QED": f"{p.qed_score:.3f}",
                        "Drug-Likeness": f"{p.drug_likeness_score:.3f}",
                        "Lipinski Violations": p.lipinski_violations,
                        "Toxicity Risk": f"{p.admet.toxicity_risk:.2f}",
                        "Synth. Accessibility": f"{p.synthetic_accessibility:.1f}",
                        "Overall Score": f"{p.overall_score:.3f}",
                    }
                    for p in property_result.predictions
                ]
            )
            st.dataframe(prop_df, use_container_width=True, hide_index=True)

            scores = [p.overall_score for p in property_result.predictions]
            col1, col2, col3 = st.columns(3)
            col1.metric("Predictions", len(property_result.predictions))
            col2.metric("Avg. Overall Score", f"{sum(scores) / len(scores):.3f}")
            col3.metric("Top Score", f"{max(scores):.3f}")

        status_4.update(label="Step 4/6: Property Prediction (complete)", state="complete")

    except Exception as exc:
        st.error(f"Property prediction failed: {exc}")
        status_4.update(label="Step 4/6: Property Prediction (failed)", state="error")
        st.stop()

# -- Step 5: Literature Search ------------------------------------------------

progress_bar.progress(STEPS[4][1], text=STEPS[4][0])

with st.status("Step 5/6: Literature Search", expanded=True) as status_5:
    st.write("Searching covalent drug design literature...")

    warhead_classes = [rec.warhead_class for rec in warhead_result.recommendations]
    warhead_str = ", ".join(warhead_classes[:3]) if warhead_classes else ""

    query_parts = [
        f"covalent inhibitors targeting {target_result.protein_name}",
        f"{target_result.residue_type} residue",
    ]
    if warhead_str:
        query_parts.append(f"warhead classes: {warhead_str}")
    if indication:
        query_parts.append(f"indication: {indication}")

    literature_input = LiteratureQuery(
        query=" ".join(query_parts),
        protein_name=target_result.protein_name,
        warhead_class=warhead_classes[0] if warhead_classes else "",
    )

    try:
        literature_agent = LiteratureRAGAgent()
        literature_result = run_async(literature_agent.run(literature_input))

        st.metric("Citations Found", len(literature_result.citations))

        if literature_result.summary:
            st.markdown(f"**Summary:** {literature_result.summary}")

        if literature_result.key_findings:
            st.markdown("**Key Findings:**")
            for finding in literature_result.key_findings:
                st.markdown(f"- {finding}")

        if literature_result.citations:
            with st.expander("Citations"):
                for cit in literature_result.citations:
                    authors_str = ", ".join(cit.authors) if cit.authors else "Unknown"
                    st.markdown(
                        f"- **{cit.title}**. {authors_str}. "
                        f"*{cit.journal}* ({cit.year}). "
                        f"PMID: {cit.pmid}. Relevance: {cit.relevance_score:.2f}"
                    )

        status_5.update(label="Step 5/6: Literature Search (complete)", state="complete")

    except Exception as exc:
        st.error(f"Literature search failed: {exc}")
        status_5.update(label="Step 5/6: Literature Search (failed)", state="error")
        st.stop()

# -- Step 6: Report Generation ------------------------------------------------

progress_bar.progress(STEPS[5][1], text=STEPS[5][0])

with st.status("Step 6/6: Report Generation", expanded=True) as status_6:
    st.write("Generating final ranked report...")

    try:
        reporter_agent = ReporterAgent()
        final_report = run_async(
            reporter_agent.run(
                target_analysis=target_result,
                warhead_selection=warhead_result,
                molecule_design=molecule_result,
                property_prediction=property_result,
                literature=literature_result,
                target_input=target_input,
            )
        )

        st.success(f"Report generated at {final_report.generated_at}")
        status_6.update(label="Step 6/6: Report Generation (complete)", state="complete")

    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
        status_6.update(label="Step 6/6: Report Generation (failed)", state="error")
        st.stop()

progress_bar.progress(1.0, text="Pipeline complete!")

# ---------------------------------------------------------------------------
# Final Report Display
# ---------------------------------------------------------------------------

st.markdown("---")
st.header("Final Report")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Target", f"{final_report.target_protein} {final_report.target_residue}")
col2.metric("Candidates Generated", final_report.num_candidates_generated)
col3.metric("Candidates Passing", final_report.num_candidates_passing)

if final_report.ranked_candidates:
    col4.metric("Top Score", f"{final_report.ranked_candidates[0].composite_score:.3f}")
else:
    col4.metric("Top Score", "N/A")

# Ranked candidates table
if final_report.ranked_candidates:
    st.subheader("Ranked Candidates")

    ranked_df = pd.DataFrame(
        [
            {
                "Rank": rc.rank,
                "Name": rc.name,
                "SMILES": rc.smiles,
                "Composite Score": rc.composite_score,
                "Warhead": rc.warhead_class,
                "Drug-Likeness": rc.drug_likeness,
                "QED": rc.qed_score,
                "Synth. Accessibility": rc.synthetic_accessibility,
            }
            for rc in final_report.ranked_candidates
        ]
    )

    st.dataframe(
        ranked_df.style.background_gradient(
            subset=["Composite Score"], cmap="RdYlGn"
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Detailed view for each candidate
    st.subheader("Candidate Details")
    for rc in final_report.ranked_candidates:
        with st.expander(f"#{rc.rank}: {rc.name} (score: {rc.composite_score:.3f})"):
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.markdown(f"**SMILES:** `{rc.smiles}`")
                st.markdown(f"**Warhead class:** {rc.warhead_class}")
                st.markdown(f"**Drug-likeness:** {rc.drug_likeness:.3f}")
                st.markdown(f"**QED score:** {rc.qed_score:.3f}")
            with detail_col2:
                st.markdown(f"**Synthetic accessibility:** {rc.synthetic_accessibility:.1f}")
                st.markdown(f"**ADMET summary:** {rc.admet_summary}")
                st.markdown(f"**Literature support:** {rc.literature_support}")
            st.markdown(f"**Rationale:** {rc.rationale}")
else:
    st.warning("No candidates passed the scoring threshold.")

# Methodology
with st.expander("Methodology"):
    st.markdown(final_report.methodology_summary)

# Ligandability assessment
with st.expander("Ligandability Assessment"):
    st.markdown(final_report.ligandability_assessment)

# Citations
if final_report.citations:
    with st.expander("Citations"):
        for cit in final_report.citations:
            authors_str = ", ".join(cit.authors) if cit.authors else "Unknown"
            st.markdown(
                f"- **{cit.title}**. {authors_str}. "
                f"*{cit.journal}* ({cit.year}). "
                f"PMID: {cit.pmid}"
            )

# Download button
st.divider()
report_json = final_report.model_dump_json(indent=2)
st.download_button(
    label="Download Report as JSON",
    data=report_json,
    file_name=f"covalent_agent_report_{protein_name}_{target_residue}.json",
    mime="application/json",
    use_container_width=True,
)
