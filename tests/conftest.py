"""Shared fixtures for CovalentAgent test suite.

All fixtures return realistic sample data that mirrors actual pipeline outputs.
External dependencies (Anthropic API, ESM-2, Chemprop, ChromaDB) are mocked
so tests run without API keys or heavy ML models.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from covalent_agent.schemas import (
    ADMETProfile,
    CandidateMolecule,
    Citation,
    LiteratureResult,
    MoleculeDesignResult,
    MoleculeProperties,
    PropertyPredictionResult,
    TargetAnalysisInput,
    TargetAnalysisResult,
    WarheadRecommendation,
    WarheadSelectionInput,
    WarheadSelectionResult,
)


# ---------------------------------------------------------------------------
# Anthropic mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic():
    """Mock Anthropic client that returns reasonable LLM responses.

    The mock simulates ``anthropic.AsyncAnthropic().messages.create()``
    returning a response object whose ``content[0].text`` is a string.
    """
    mock_text_block = MagicMock()
    mock_text_block.text = (
        "KRAS G12C is a validated covalent drug target. The mutant cysteine "
        "at position 12 is accessible to electrophilic warheads and has been "
        "successfully targeted by sotorasib and adagrasib."
    )

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]

    mock_client_instance = MagicMock()
    mock_client_instance.messages = MagicMock()
    mock_client_instance.messages.create = AsyncMock(return_value=mock_response)

    mock_client_class = MagicMock(return_value=mock_client_instance)

    return mock_client_class


# ---------------------------------------------------------------------------
# Target analysis fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_target_input():
    """Realistic TargetAnalysisInput for KRAS G12C."""
    return TargetAnalysisInput(
        protein_name="KRAS",
        residue="C12",
        indication="NSCLC",
    )


@pytest.fixture
def sample_target_input_unknown():
    """TargetAnalysisInput for a protein NOT in the known database."""
    return TargetAnalysisInput(
        protein_name="FAKEPROT",
        residue="C42",
        indication="",
    )


@pytest.fixture
def sample_target_result():
    """Realistic TargetAnalysisResult for KRAS G12C."""
    return TargetAnalysisResult(
        protein_name="KRAS",
        uniprot_id="P01116",
        residue_type="cysteine",
        residue_position=12,
        ligandability_score=0.85,
        conservation_score=0.7,
        structural_context="Mutant cysteine in switch II region",
        known_drugs=["sotorasib", "adagrasib"],
        esm_confidence=0.8,
        rationale="KRAS G12C is a validated covalent drug target",
    )


# ---------------------------------------------------------------------------
# Warhead selection fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_warhead_input():
    """Realistic WarheadSelectionInput derived from KRAS target analysis."""
    return WarheadSelectionInput(
        residue_type="cysteine",
        ligandability_score=0.85,
        structural_context="Mutant cysteine in switch II region",
        protein_name="KRAS",
    )


@pytest.fixture
def sample_warhead_recommendations():
    """List of realistic WarheadRecommendation objects for cysteine."""
    return [
        WarheadRecommendation(
            warhead_class="Acrylamide",
            smarts="[CH2]=[CH]-C(=O)-N",
            reactivity="moderate",
            selectivity="high",
            score=0.85,
            rationale=(
                "Acrylamide is the most validated warhead for cysteine targeting. "
                "Sotorasib and ibrutinib use this warhead class."
            ),
            examples=["osimertinib", "ibrutinib", "afatinib"],
            mechanism="Michael addition to cysteine thiol",
        ),
        WarheadRecommendation(
            warhead_class="Chloroacetamide",
            smarts="ClCC(=O)N",
            reactivity="high",
            selectivity="moderate",
            score=0.72,
            rationale="Chloroacetamide provides high reactivity via SN2 displacement.",
            examples=["ML162"],
            mechanism="SN2 displacement by cysteine thiol",
        ),
        WarheadRecommendation(
            warhead_class="Cyanoacrylamide",
            smarts="N#C/C=C/C(=O)N",
            reactivity="high",
            selectivity="high",
            score=0.68,
            rationale="Reversible covalent warhead reduces off-target toxicity.",
            examples=["PRN1371"],
            mechanism="Reversible covalent Michael addition",
        ),
    ]


@pytest.fixture
def sample_warhead_selection_result(sample_warhead_recommendations):
    """WarheadSelectionResult for cysteine targeting."""
    return WarheadSelectionResult(
        target_residue="cysteine",
        recommendations=sample_warhead_recommendations,
    )


# ---------------------------------------------------------------------------
# Molecule design fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_candidates():
    """List of realistic CandidateMolecule objects."""
    return [
        CandidateMolecule(
            smiles="c1cnc(NC(=O)C=C)nc1",
            name="KRAS_Acrylamide_pyrimidine_0",
            scaffold_type="pyrimidine",
            warhead_class="Acrylamide",
            molecular_weight=150.14,
            logp=0.5,
            num_h_donors=2,
            num_h_acceptors=4,
            num_rotatable_bonds=2,
            tpsa=80.0,
        ),
        CandidateMolecule(
            smiles="c1ccc2[nH]cnc2c1NC(=O)C=C",
            name="KRAS_Acrylamide_benzimidazole_1",
            scaffold_type="benzimidazole",
            warhead_class="Acrylamide",
            molecular_weight=213.24,
            logp=1.2,
            num_h_donors=2,
            num_h_acceptors=4,
            num_rotatable_bonds=2,
            tpsa=70.4,
        ),
        CandidateMolecule(
            smiles="Nc1ccncc1NC(=O)C=C",
            name="KRAS_Acrylamide_aminopyridine_2",
            scaffold_type="aminopyridine",
            warhead_class="Acrylamide",
            molecular_weight=163.18,
            logp=-0.1,
            num_h_donors=3,
            num_h_acceptors=4,
            num_rotatable_bonds=2,
            tpsa=90.3,
        ),
    ]


@pytest.fixture
def sample_molecule_design_result(sample_candidates):
    """MoleculeDesignResult with 3 candidates."""
    return MoleculeDesignResult(
        candidates=sample_candidates,
        design_rationale=(
            "Generated 3 candidate molecules using fragment-based assembly "
            "with pyrimidine, benzimidazole, and aminopyridine scaffolds."
        ),
    )


# ---------------------------------------------------------------------------
# Property prediction fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_molecule_properties():
    """List of realistic MoleculeProperties for the sample candidates."""
    return [
        MoleculeProperties(
            smiles="c1cnc(NC(=O)C=C)nc1",
            drug_likeness_score=0.75,
            qed_score=0.62,
            lipinski_violations=0,
            admet=ADMETProfile(
                absorption_score=0.85,
                distribution_score=0.70,
                metabolism_score=0.65,
                excretion_score=0.80,
                toxicity_risk=0.15,
            ),
            synthetic_accessibility=2.5,
            overall_score=0.72,
        ),
        MoleculeProperties(
            smiles="c1ccc2[nH]cnc2c1NC(=O)C=C",
            drug_likeness_score=0.80,
            qed_score=0.58,
            lipinski_violations=0,
            admet=ADMETProfile(
                absorption_score=0.80,
                distribution_score=0.65,
                metabolism_score=0.60,
                excretion_score=0.75,
                toxicity_risk=0.20,
            ),
            synthetic_accessibility=3.1,
            overall_score=0.68,
        ),
        MoleculeProperties(
            smiles="Nc1ccncc1NC(=O)C=C",
            drug_likeness_score=0.70,
            qed_score=0.55,
            lipinski_violations=0,
            admet=ADMETProfile(
                absorption_score=0.82,
                distribution_score=0.68,
                metabolism_score=0.62,
                excretion_score=0.78,
                toxicity_risk=0.18,
            ),
            synthetic_accessibility=2.0,
            overall_score=0.65,
        ),
    ]


@pytest.fixture
def sample_property_prediction_result(sample_molecule_properties):
    """PropertyPredictionResult with predictions for all sample candidates."""
    return PropertyPredictionResult(predictions=sample_molecule_properties)


# ---------------------------------------------------------------------------
# Literature fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_citations():
    """List of realistic Citation objects from the starter corpus."""
    return [
        Citation(
            title="K-Ras(G12C) inhibitors allosterically control GTP affinity and effector interactions",
            authors=["Ostrem JM", "Peters U", "Sos ML", "Wells JA", "Shokat KM"],
            journal="Nature",
            year=2013,
            pmid="24256730",
            abstract=(
                "Discovery of allosteric inhibitors targeting KRAS G12C mutant "
                "cysteine in the switch II pocket."
            ),
            relevance_score=0.92,
        ),
        Citation(
            title="The clinical KRAS(G12C) inhibitor AMG 510 drives anti-tumour immunity",
            authors=["Canon J", "Rex K", "Saiki AY", "et al."],
            journal="Nature",
            year=2019,
            pmid="31645765",
            abstract=(
                "Preclinical and clinical characterization of sotorasib (AMG 510), "
                "the first KRAS G12C inhibitor to enter clinical trials."
            ),
            relevance_score=0.88,
        ),
    ]


@pytest.fixture
def sample_literature_result(sample_citations):
    """LiteratureResult for a KRAS covalent inhibitor query."""
    return LiteratureResult(
        query="covalent inhibitors targeting KRAS cysteine residue",
        citations=sample_citations,
        summary=(
            "KRAS G12C has been validated as a covalent drug target with two "
            "approved inhibitors, sotorasib and adagrasib."
        ),
        key_findings=[
            "KRAS G12C can be allosterically inhibited via the switch II pocket.",
            "Sotorasib was the first KRAS G12C inhibitor in clinical trials.",
        ],
    )


# ---------------------------------------------------------------------------
# ESM wrapper mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_esm_wrapper():
    """Mock ESMWrapper that operates in fallback mode without torch."""
    with patch("covalent_agent.agents.target_analyst.ESMWrapper") as mock_cls:
        instance = MagicMock()
        instance.fallback_mode = True
        instance.score_residue_ligandability.return_value = 0.78
        instance.get_context_window.return_value = "VVGACGVGKSALTIQ"
        mock_cls.return_value = instance
        yield instance


# ---------------------------------------------------------------------------
# httpx mock (for UniProt requests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_httpx_uniprot():
    """Mock httpx.AsyncClient for UniProt API calls."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "results": [
            {
                "primaryAccession": "P01116",
                "sequence": {"value": "MTEYKLVVVGACGVGKSALTIQLIQNHFVDEYDPTIEDSY"},
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "GTPase KRas"},
                    },
                },
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response
