# CovalentAgent

**The first open-source multi-agent system for covalent drug design.**

CovalentAgent orchestrates specialized AI agents to mirror the covalent drug discovery workflow: from protein target identification through binding site scoring, warhead selection, molecule generation, and property prediction, all coordinated by a supervisor agent using MCP and tool calling.

## Why Covalent?

Most drug discovery AI tools focus on non-covalent binding. Covalent drugs, which form permanent bonds with their target proteins, represent a growing class of therapeutics (including blockbusters like osimertinib and sotorasib). Yet the open-source tooling for AI-powered covalent drug design is nearly nonexistent.

CovalentAgent fills this gap.

## Architecture

```
Supervisor Agent (LangGraph + MCP)
  |
  +-- TargetAnalyst       Protein -> reactive residue identification -> ligandability scoring
  |                       (ESM-2 protein language model)
  |
  +-- WarheadSelector     Residue type + pocket context -> optimal warhead class
  |                       (Reactivity models + rule-based selection)
  |
  +-- MoleculeDesigner    Scaffold generation + warhead attachment
  |                       (RDKit, molecular generation)
  |
  +-- PropertyPredictor   ADMET, selectivity, drug-likeness scoring
  |                       (Chemprop, RDKit descriptors)
  |
  +-- LiteratureRAG       Citation-backed answers from covalent drug design literature
  |                       (ChromaDB + PubMed/CovPDB corpus)
  |
  +-- Reporter            Ranked candidate report with rationale and citations
```

## Quick Start

```bash
# Clone
git clone https://github.com/deepmind11/CovalentAgent.git
cd CovalentAgent

# Install
pip install -e ".[dev]"

# Set API keys
cp .env.example .env
# Edit .env with your API keys

# Run the demo
streamlit run app/demo.py

# Or use the CLI
python -m covalent_agent --target KRAS --residue C12
```

## Example Workflow

```python
from covalent_agent import CovalentAgent

agent = CovalentAgent()

# Analyze a target protein for covalent drug design opportunities
result = agent.run(
    target="KRAS",
    residue="C12",
    indication="non-small cell lung cancer"
)

# Result includes:
# - Ligandability score for the target cysteine
# - Recommended warhead classes with rationale
# - Generated candidate molecules with warheads attached
# - ADMET and property predictions for each candidate
# - Literature references supporting design decisions
print(result.report())
```

## Agents in Detail

### TargetAnalyst
Takes a protein (UniProt ID or PDB structure) and identifies druggable reactive residues (cysteines, lysines, etc.). Uses ESM-2 protein language model embeddings to score residue ligandability based on structural context and evolutionary conservation.

### WarheadSelector
Given a target residue type and binding pocket context, recommends optimal covalent warhead classes (acrylamides, chloroacetamides, vinyl sulfonamides, etc.). Considers reactivity, selectivity, and precedent from known covalent drugs.

### MoleculeDesigner
Generates candidate molecules with appropriate warheads attached. Uses RDKit for molecular manipulation, fragment-based design, and scaffold enumeration. Outputs valid SMILES with drug-like properties.

### PropertyPredictor
Predicts ADMET properties (absorption, distribution, metabolism, excretion, toxicity), selectivity profiles, and drug-likeness scores using Chemprop message-passing neural networks and RDKit molecular descriptors.

### LiteratureRAG
Retrieves and synthesizes information from covalent drug design literature, CovPDB entries, and published SAR data. Provides citation-backed rationale for every design decision.

### Reporter
Generates a structured report ranking all candidate molecules by composite score, including predicted properties, warhead rationale, literature support, and recommended next steps.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Orchestration | LangGraph + MCP |
| Protein Language Model | ESM-2 (Facebook/Meta) |
| Molecular Property Prediction | Chemprop |
| Molecular Manipulation | RDKit |
| Vector Store (RAG) | ChromaDB |
| API Serving | FastAPI |
| Demo UI | Streamlit |
| Containerization | Docker |

## Data Sources

- **CovPDB**: Covalent protein-ligand complex database
- **CysDB**: Cysteine chemoproteomics annotations
- **ChEMBL**: Bioactivity data for covalent inhibitors
- **PDB**: Protein structures
- **PubMed**: Covalent drug design literature

## Project Structure

```
CovalentAgent/
├── README.md
├── pyproject.toml
├── .env.example
├── app/
│   └── demo.py                  # Streamlit demo UI
├── src/
│   └── covalent_agent/
│       ├── __init__.py
│       ├── supervisor.py        # Supervisor agent orchestration
│       ├── config.py            # Configuration and settings
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── target_analyst.py
│       │   ├── warhead_selector.py
│       │   ├── molecule_designer.py
│       │   ├── property_predictor.py
│       │   ├── literature_rag.py
│       │   └── reporter.py
│       ├── tools/               # MCP tool definitions
│       │   ├── __init__.py
│       │   ├── protein_tools.py
│       │   ├── chemistry_tools.py
│       │   └── literature_tools.py
│       ├── models/              # ML model wrappers
│       │   ├── __init__.py
│       │   ├── esm_wrapper.py
│       │   └── chemprop_wrapper.py
│       └── data/
│           ├── __init__.py
│           ├── loaders.py
│           └── warhead_library.py
├── tests/
│   ├── __init__.py
│   ├── test_target_analyst.py
│   ├── test_warhead_selector.py
│   ├── test_molecule_designer.py
│   ├── test_property_predictor.py
│   └── test_supervisor.py
├── data/
│   ├── warheads.json            # Warhead class definitions
│   ├── reactive_residues.json   # Reactive residue types and properties
│   └── README.md                # Data source documentation
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

## Contributing

Contributions welcome. Please open an issue first to discuss what you'd like to change.

## License

MIT

## References

- Backus et al. "Proteome-wide covalent ligand discovery in native biological systems." Nature (2016)
- Kuljanin et al. "Reimagining high-throughput profiling of reactive cysteines for cell-based screening of large electrophile libraries." Nature Biotechnology (2021)
- Lin et al. "Evolutionary-scale prediction of atomic-level protein structure with a language model." Science (2023)
- Yang et al. "Analyzing Learned Molecular Representations for Property Prediction." JCIM (2019)
