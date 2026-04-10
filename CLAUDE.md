# CovalentAgent

## What This Is

A multi-agent system for covalent drug design. Integrates multi-agent orchestration, MCP/tool calling, RAG pipelines, molecular property prediction, and model deployment.

## Architecture

Supervisor agent orchestrates 6 specialized agents via LangGraph + MCP:

1. **TargetAnalyst** - Protein -> reactive residue identification -> ligandability scoring (ESM-2)
2. **WarheadSelector** - Residue type + pocket -> optimal warhead class (rule-based + reactivity models)
3. **MoleculeDesigner** - Scaffold generation + warhead attachment (RDKit)
4. **PropertyPredictor** - ADMET, drug-likeness scoring (Chemprop + RDKit descriptors)
5. **LiteratureRAG** - RAG over covalent drug design papers (ChromaDB + PubMed)
6. **Reporter** - Structured report with ranked candidates, rationale, citations

## Tech Stack

- Python 3.11+, PyTorch, LangGraph, Anthropic SDK
- RDKit (molecular manipulation), Chemprop (property prediction)
- ESM-2 via HuggingFace (protein language model)
- ChromaDB (vector store), FastAPI (serving), Streamlit (demo UI)

## Build Priorities

1. Get a working end-to-end demo first (even if agents are simple)
2. Each agent should work independently and be testable in isolation
3. The supervisor orchestration is the star; individual agents can wrap existing libraries

## Git Config

```bash
git config user.email "harshitghosh@gmail.com"
git config user.name "Harshit Ghosh"
```

## Key Design Decisions

- Agents communicate through structured Pydantic models, not free text
- Each agent exposes its capabilities as MCP tools
- The supervisor uses a state machine (LangGraph) to decide which agent to call next
- RAG indexes CovPDB entries + PubMed abstracts on covalent drug design
- Warhead library is a curated JSON of known warhead classes with properties

## Testing

- Each agent has unit tests with mocked dependencies
- Integration test: end-to-end KRAS C12 -> ranked candidates
- `pytest` with `pytest-asyncio` for async agent tests
