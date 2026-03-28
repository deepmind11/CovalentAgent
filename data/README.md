# Data Sources

## Included Data

- `warheads.json`: Curated library of covalent warhead classes with SMARTS patterns, target residues, reactivity profiles, and approved drug examples.
- `reactive_residues.json`: Known druggable reactive residues from validated covalent drug targets.

## External Data (Not Included, Downloaded at Runtime)

- **CovPDB** (http://www.covalentindb.cn/): Covalent protein-ligand complex database
- **CysDB**: Cysteine chemoproteomics annotations
- **ChEMBL**: Bioactivity data filtered for covalent inhibitors
- **PDB**: Protein structures via RCSB API
- **PubMed**: Abstracts indexed for RAG via Entrez API
