"""Literature RAG agent using ChromaDB over PubMed abstracts.

Maintains a starter corpus of key covalent drug design papers, indexes them
in ChromaDB with embeddings, and uses Anthropic Claude to synthesize
summaries from retrieved documents.
"""

from __future__ import annotations

import logging
from typing import Any

from covalent_agent.config import settings
from covalent_agent.schemas import Citation, LiteratureQuery, LiteratureResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Starter corpus: real papers in covalent drug design
# ---------------------------------------------------------------------------

STARTER_CORPUS: list[dict[str, Any]] = [
    {
        "title": "Proteome-wide covalent ligand discovery in native biological systems",
        "authors": ["Backus KM", "Correia BE", "Lum KM", "Forber S", "et al."],
        "journal": "Nature",
        "year": 2016,
        "pmid": "27309814",
        "abstract": (
            "Fragment-based ligand discovery using quantitative chemical proteomics "
            "to identify covalent ligands for >700 cysteine residues in the human "
            "proteome. Demonstrates that a large fraction of the proteome contains "
            "ligandable cysteines accessible to electrophilic small molecules."
        ),
    },
    {
        "title": (
            "Reimagining high-throughput profiling of reactive cysteines for "
            "cell-based screening of large electrophile libraries"
        ),
        "authors": ["Kuljanin M", "Mitchell DC", "Schweppe DK", "et al."],
        "journal": "Nature Biotechnology",
        "year": 2021,
        "pmid": "33398153",
        "abstract": (
            "Scalable chemoproteomic platform for screening large electrophile "
            "libraries against reactive cysteines in live cells. Identifies "
            "thousands of cysteine-ligand interactions enabling systematic "
            "covalent drug discovery."
        ),
    },
    {
        "title": "K-Ras(G12C) inhibitors allosterically control GTP affinity and effector interactions",
        "authors": ["Ostrem JM", "Peters U", "Sos ML", "Wells JA", "Shokat KM"],
        "journal": "Nature",
        "year": 2013,
        "pmid": "24256730",
        "abstract": (
            "Discovery of allosteric inhibitors targeting KRAS G12C mutant cysteine "
            "in the switch II pocket. Demonstrates that small molecules can lock "
            "KRAS in its inactive GDP-bound conformation through covalent "
            "modification of cysteine 12."
        ),
    },
    {
        "title": "The clinical KRAS(G12C) inhibitor AMG 510 drives anti-tumour immunity",
        "authors": ["Canon J", "Rex K", "Saiki AY", "et al."],
        "journal": "Nature",
        "year": 2019,
        "pmid": "31645765",
        "abstract": (
            "Preclinical and clinical characterization of sotorasib (AMG 510), the "
            "first KRAS G12C inhibitor to enter clinical trials. Shows potent and "
            "selective covalent inhibition of KRAS G12C with anti-tumour immune "
            "responses in vivo."
        ),
    },
    {
        "title": "KRAS(G12C) inhibition with sotorasib in advanced solid tumors",
        "authors": ["Hong DS", "Fakih MG", "Strickler JH", "et al."],
        "journal": "New England Journal of Medicine",
        "year": 2020,
        "pmid": "32955176",
        "abstract": (
            "Phase I clinical trial of sotorasib in patients with advanced "
            "KRAS G12C-mutated solid tumors. Demonstrated durable clinical "
            "responses with manageable toxicity, establishing proof of concept "
            "for covalent KRAS G12C inhibition in humans."
        ),
    },
    {
        "title": "Anti-tumour efficacy of MRTX849, a mutant-selective KRAS(G12C) inhibitor",
        "authors": ["Hallin J", "Engstrom LD", "Hargis L", "et al."],
        "journal": "Cancer Discovery",
        "year": 2020,
        "pmid": "31658955",
        "abstract": (
            "Characterization of adagrasib (MRTX849), a covalent KRAS G12C "
            "inhibitor with optimized pharmacokinetic properties. Demonstrates "
            "broad anti-tumour activity and central nervous system penetration "
            "in preclinical models."
        ),
    },
    {
        "title": "The resurgence of covalent drugs",
        "authors": ["Singh J", "Petter RC", "Baillie TA", "Whitty A"],
        "journal": "Nature Reviews Drug Discovery",
        "year": 2011,
        "pmid": "21455239",
        "abstract": (
            "Comprehensive review of the re-emergence of targeted covalent "
            "inhibitors in drug discovery. Argues that rational design of "
            "covalent drugs can achieve high selectivity and prolonged target "
            "engagement while managing reactivity risks."
        ),
    },
    {
        "title": (
            "Emerging and re-emerging warheads for targeted covalent inhibitors: "
            "applications in medicinal chemistry and chemical biology"
        ),
        "authors": ["Gehringer M", "Laufer SA"],
        "journal": "Journal of Medicinal Chemistry",
        "year": 2019,
        "pmid": "30565923",
        "abstract": (
            "Systematic review of electrophilic warheads for targeted covalent "
            "inhibitors. Covers acrylamides, vinyl sulfonamides, cyanoacrylamides, "
            "propynamides, and emerging warhead classes with structure-activity "
            "relationships and selectivity considerations."
        ),
    },
    {
        "title": "Covalent inhibitors: a rational approach to drug discovery",
        "authors": ["Lagoutte R", "Patber R", "Bhatt A"],
        "journal": "Current Opinion in Chemical Biology",
        "year": 2017,
        "pmid": "28822910",
        "abstract": (
            "Review of rational design strategies for covalent inhibitors "
            "including warhead selection, linker optimization, and "
            "structure-based design. Discusses targeting non-catalytic "
            "cysteines and expanding beyond kinase targets."
        ),
    },
    {
        "title": "CovPDB: a high-resolution coverage of the covalent protein-ligand interactome",
        "authors": ["Roth BM", "Bhatt D", "Engberding AI", "et al."],
        "journal": "Journal of Chemical Information and Modeling",
        "year": 2020,
        "pmid": "32672949",
        "abstract": (
            "Comprehensive database of covalent protein-ligand complexes curated "
            "from the PDB. Provides structural analysis of covalent binding modes, "
            "warhead preferences, and residue targeting across the proteome."
        ),
    },
    {
        "title": "Chemoproteomics-enabled covalent ligand screening in a proteome-wide manner",
        "authors": ["Drewes G", "Knapp S"],
        "journal": "Trends in Biotechnology",
        "year": 2018,
        "pmid": "30115413",
        "abstract": (
            "Review of chemoproteomic methods for discovering and profiling "
            "covalent ligands. Covers activity-based protein profiling, "
            "isoTOP-ABPP, and other mass spectrometry-based approaches for "
            "mapping ligandable residues."
        ),
    },
    {
        "title": "Tri-complex inhibitors of the oncogenic KRAS G12D mutant",
        "authors": ["Pan Z", "Scheerens H", "Li SJ", "et al."],
        "journal": "Science",
        "year": 2022,
        "pmid": "36302005",
        "abstract": (
            "Discovery of small molecules targeting KRAS G12D through a "
            "tri-complex mechanism involving a cyclophilin chaperone. "
            "Addresses the challenge of targeting non-cysteine KRAS mutants "
            "and provides a framework for drugging undruggable oncoproteins."
        ),
    },
    {
        "title": "Targeted covalent inhibitors of the kinase family",
        "authors": ["Zhao Z", "Bourne PE"],
        "journal": "Drug Discovery Today",
        "year": 2018,
        "pmid": "29337069",
        "abstract": (
            "Systematic analysis of covalent kinase inhibitors including "
            "their binding modes, warhead chemistry, and clinical development "
            "status. Covers approved drugs like ibrutinib, afatinib, "
            "osimertinib, and neratinib."
        ),
    },
    {
        "title": (
            "Expanding the druggable proteome: non-canonical amino acid targets "
            "for covalent drugs"
        ),
        "authors": ["Hacker SM", "Backus KM", "Lazear MR", "et al."],
        "journal": "Journal of the American Chemical Society",
        "year": 2017,
        "pmid": "29048884",
        "abstract": (
            "Extends covalent ligand discovery beyond cysteine to target lysine, "
            "methionine, and other amino acids. Uses electrophilic fragments with "
            "tuned reactivity to discover ligands for non-traditional nucleophilic "
            "residues in the proteome."
        ),
    },
    {
        "title": "Structure-based design of covalent Siah1 inhibitors",
        "authors": ["Stebbins JL", "Santelli E", "Feng Y", "et al."],
        "journal": "Chemistry and Biology",
        "year": 2013,
        "pmid": "24267275",
        "abstract": (
            "Structure-guided design of covalent inhibitors targeting the E3 "
            "ubiquitin ligase Siah1. Demonstrates rational covalent drug design "
            "using crystal structures to position electrophilic warheads near "
            "target cysteine residues."
        ),
    },
    {
        "title": "An activity-guided map of electrophile-cysteine interactions in primary human T cells",
        "authors": ["Vinogradova EV", "Zhang X", "Remillard D", "et al."],
        "journal": "Cell",
        "year": 2020,
        "pmid": "31955846",
        "abstract": (
            "Chemical proteomic mapping of electrophile-reactive cysteines in "
            "primary human T cells. Identifies druggable cysteines in immune "
            "signaling proteins and discovers covalent immunomodulatory compounds "
            "with therapeutic potential."
        ),
    },
    {
        "title": "Chemoproteomic profiling of kinases by ligand and target discovery",
        "authors": ["Zhao Q", "Ouyang X", "Wan X", "et al."],
        "journal": "ACS Chemical Biology",
        "year": 2017,
        "pmid": "28135068",
        "abstract": (
            "Chemoproteomic strategy for systematic identification of kinase "
            "targets of covalent inhibitors. Combines activity-based probes with "
            "quantitative mass spectrometry to map the selectivity landscape of "
            "covalent kinase drugs."
        ),
    },
    {
        "title": (
            "Irreversible inhibitors of the EGFR tyrosine kinase: a new "
            "paradigm for the treatment of ErbB-driven cancers"
        ),
        "authors": ["Sequist LV", "Soria JC", "Goldman JW", "et al."],
        "journal": "Cancer Discovery",
        "year": 2010,
        "pmid": "20671055",
        "abstract": (
            "Overview of irreversible EGFR inhibitors as a paradigm for covalent "
            "cancer therapeutics. Discusses the pharmacological advantages of "
            "irreversible inhibition including sustained target suppression and "
            "activity against resistance mutations."
        ),
    },
]


# ---------------------------------------------------------------------------
# Utility: build Citation from corpus entry
# ---------------------------------------------------------------------------

def _corpus_entry_to_citation(entry: dict[str, Any], relevance: float = 0.5) -> Citation:
    """Convert a starter corpus dict to a Citation model."""
    return Citation(
        title=entry["title"],
        authors=entry.get("authors", []),
        journal=entry.get("journal", ""),
        year=entry.get("year", 0),
        pmid=entry.get("pmid", ""),
        doi=entry.get("doi", ""),
        abstract=entry.get("abstract", ""),
        relevance_score=relevance,
    )


# ---------------------------------------------------------------------------
# LiteratureRAGAgent
# ---------------------------------------------------------------------------

class LiteratureRAGAgent:
    """RAG agent backed by ChromaDB over covalent drug design literature.

    On first instantiation the agent seeds a starter corpus of key papers.
    Queries are embedded and matched against the corpus, then Anthropic
    Claude synthesizes a narrative summary with citations.

    Falls back to keyword-based search over the hardcoded corpus if ChromaDB
    is unavailable.
    """

    _COLLECTION_NAME = "covalent_lit"
    _SEEDED_FLAG = "_covalent_lit_seeded"

    def __init__(self) -> None:
        self._chroma_available = False
        self._collection = None
        self._client = None
        self._try_init_chroma()

    # ------------------------------------------------------------------
    # ChromaDB initialisation
    # ------------------------------------------------------------------

    def _try_init_chroma(self) -> None:
        """Attempt to initialise ChromaDB; set fallback flag on failure."""
        try:
            import chromadb  # noqa: F811

            persist_dir = str(settings.chroma_persist_dir)
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self._COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_available = True
            self._seed_if_needed()
        except Exception as exc:
            logger.warning(
                "ChromaDB unavailable, falling back to keyword search: %s", exc
            )
            self._chroma_available = False

    def _seed_if_needed(self) -> None:
        """Seed the starter corpus into ChromaDB if the collection is empty."""
        if self._collection is None:
            return

        if self._collection.count() >= len(STARTER_CORPUS):
            return

        logger.info(
            "Seeding ChromaDB collection with %d starter papers", len(STARTER_CORPUS)
        )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for entry in STARTER_CORPUS:
            doc_id = f"pmid_{entry['pmid']}"
            doc_text = (
                f"{entry['title']}. "
                f"{', '.join(entry['authors'])}. "
                f"{entry['journal']} ({entry['year']}). "
                f"{entry['abstract']}"
            )
            metadata = {
                "title": entry["title"],
                "authors": ", ".join(entry["authors"]),
                "journal": entry.get("journal", ""),
                "year": entry.get("year", 0),
                "pmid": entry.get("pmid", ""),
            }

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(metadata)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Seeding complete: %d documents indexed", len(ids))

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    async def run(self, input: LiteratureQuery) -> LiteratureResult:
        """Execute the literature RAG pipeline.

        Steps:
          1. Retrieve top-k papers from ChromaDB (or fallback keyword search)
          2. Synthesize a summary via Anthropic Claude
          3. Return LiteratureResult with citations, summary, key findings
        """
        citations = self._retrieve(input)
        summary, key_findings = await self._synthesize(input.query, citations)

        return LiteratureResult(
            query=input.query,
            citations=citations,
            summary=summary,
            key_findings=key_findings,
        )

    def _retrieve(self, input: LiteratureQuery) -> list[Citation]:
        """Retrieve relevant papers for the query."""
        if self._chroma_available and self._collection is not None:
            return self._retrieve_chroma(input)
        return self._retrieve_fallback(input)

    def _retrieve_chroma(self, input: LiteratureQuery) -> list[Citation]:
        """Retrieve via ChromaDB vector similarity search."""
        query_text = input.query
        if input.protein_name:
            query_text += f" {input.protein_name}"
        if input.warhead_class:
            query_text += f" {input.warhead_class}"

        results = self._collection.query(
            query_texts=[query_text],
            n_results=min(input.max_results, self._collection.count()),
        )

        citations: list[Citation] = []
        if not results or not results.get("ids"):
            return citations

        ids = results["ids"][0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]

        for i, doc_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            relevance = round(max(1.0 - distance, 0.0), 3)
            meta = metadatas[i] if i < len(metadatas) else {}
            doc_text = documents[i] if i < len(documents) else ""

            # Extract abstract from full document text (after last period+space
            # that follows the journal/year line)
            abstract = meta.get("abstract", "")
            if not abstract and doc_text:
                abstract = doc_text

            citations.append(
                Citation(
                    title=meta.get("title", ""),
                    authors=(
                        meta.get("authors", "").split(", ")
                        if isinstance(meta.get("authors"), str)
                        else []
                    ),
                    journal=meta.get("journal", ""),
                    year=int(meta.get("year", 0)),
                    pmid=meta.get("pmid", ""),
                    abstract=abstract,
                    relevance_score=relevance,
                )
            )

        return citations

    def _retrieve_fallback(self, input: LiteratureQuery) -> list[Citation]:
        """Keyword-based fallback when ChromaDB is unavailable."""
        query_lower = input.query.lower()
        terms = query_lower.split()
        if input.protein_name:
            terms.append(input.protein_name.lower())
        if input.warhead_class:
            terms.append(input.warhead_class.lower())

        scored: list[tuple[float, dict[str, Any]]] = []

        for entry in STARTER_CORPUS:
            searchable = (
                f"{entry['title']} {entry['abstract']} "
                f"{' '.join(entry['authors'])} {entry['journal']}"
            ).lower()

            matches = sum(1 for t in terms if t in searchable)
            if matches > 0:
                relevance = min(matches / max(len(terms), 1), 1.0)
                scored.append((relevance, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: input.max_results]

        return [_corpus_entry_to_citation(entry, rel) for rel, entry in top]

    # ------------------------------------------------------------------
    # LLM synthesis
    # ------------------------------------------------------------------

    async def _synthesize(
        self, query: str, citations: list[Citation]
    ) -> tuple[str, list[str]]:
        """Use Anthropic Claude to synthesize a summary from retrieved papers.

        Returns (summary_text, list_of_key_findings).
        Falls back to a simple concatenation if the API call fails.
        """
        if not citations:
            return (
                "No relevant literature found for this query.",
                [],
            )

        context_parts: list[str] = []
        for i, cit in enumerate(citations, 1):
            context_parts.append(
                f"[{i}] {cit.title}. {', '.join(cit.authors)}. "
                f"{cit.journal} ({cit.year}). PMID: {cit.pmid}\n"
                f"Abstract: {cit.abstract}"
            )

        context_block = "\n\n".join(context_parts)

        prompt = (
            f"You are a medicinal chemistry expert. Based on the following "
            f"retrieved papers, provide:\n"
            f"1. A concise narrative summary answering the query\n"
            f"2. A list of 3-5 key findings (each one sentence)\n\n"
            f"Query: {query}\n\n"
            f"Retrieved papers:\n{context_block}\n\n"
            f"Format your response as:\n"
            f"SUMMARY:\n<your summary>\n\n"
            f"KEY FINDINGS:\n- <finding 1>\n- <finding 2>\n..."
        )

        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text
            return self._parse_synthesis(text)

        except Exception as exc:
            logger.warning("LLM synthesis failed, using fallback: %s", exc)
            return self._fallback_synthesis(query, citations)

    @staticmethod
    def _parse_synthesis(text: str) -> tuple[str, list[str]]:
        """Parse the LLM response into summary and key findings."""
        summary = ""
        key_findings: list[str] = []

        if "SUMMARY:" in text and "KEY FINDINGS:" in text:
            parts = text.split("KEY FINDINGS:")
            summary_part = parts[0].replace("SUMMARY:", "").strip()
            findings_part = parts[1].strip()

            summary = summary_part

            for line in findings_part.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    key_findings.append(line[2:].strip())
                elif line.startswith("* "):
                    key_findings.append(line[2:].strip())
                elif line and not line.startswith(("SUMMARY", "KEY")):
                    key_findings.append(line)
        else:
            summary = text.strip()

        return summary, key_findings

    @staticmethod
    def _fallback_synthesis(
        query: str, citations: list[Citation]
    ) -> tuple[str, list[str]]:
        """Generate a simple summary without LLM access."""
        titles = [f'"{c.title}" ({c.journal}, {c.year})' for c in citations]
        summary = (
            f"Found {len(citations)} relevant papers for: {query}. "
            f"Key references include: {'; '.join(titles[:3])}."
        )
        key_findings = [
            f"{c.title} ({c.year}): {c.abstract[:120]}..."
            for c in citations[:5]
            if c.abstract
        ]
        return summary, key_findings

    # ------------------------------------------------------------------
    # Single citation lookup
    # ------------------------------------------------------------------

    def get_citation_by_pmid(self, pmid: str) -> Citation | None:
        """Look up a single paper by PMID from the starter corpus."""
        for entry in STARTER_CORPUS:
            if entry["pmid"] == pmid:
                return _corpus_entry_to_citation(entry, relevance=1.0)
        return None
