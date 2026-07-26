"""Stage 1 (Ingestion) of the Graph RAG pipeline: extract entity-relationship
triplets from a PDF with the LLM backbone, then persist them so retrieval
always runs against a stable, pre-built representation (thesis §3.3.1).

Storage backend: Neo4j when NEO4J_URI is configured (see
config.neo4j_enabled()), otherwise a NetworkX graph persisted as JSON at
data/graph_index/<doc_id>/graph.json.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import networkx as nx
from langchain_community.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_llm, get_neo4j_driver, neo4j_database, neo4j_enabled

logger = logging.getLogger(__name__)

GRAPH_ROOT = Path(__file__).resolve().parent.parent / "data" / "graph_index"

# Larger than the vector chunk size — entity extraction benefits from more
# surrounding context per LLM call, and fewer calls keeps ingestion cheap.
# Extraction is a coarser task than vector-search retrieval, so it tolerates
# much bigger chunks; this cuts the number of LLM round-trips substantially.
CHUNK_SIZE = 6000
CHUNK_OVERLAP = 400

# Chunk-extraction LLM calls are independent, so they're fired concurrently
# by default. That only helps against a backend that actually serves
# multiple requests in parallel (a hosted API, or an Ollama server tuned
# with OLLAMA_NUM_PARALLEL > 1). A default-configured local Ollama instance
# serves one request at a time, so extra worker threads just add context-
# switching overhead with no throughput gain — default to 1 there unless
# the user opts in.
_DEFAULT_EXTRACTION_WORKERS = "1" if os.getenv("OLLAMA_MODEL") else "4"
EXTRACTION_WORKERS = int(
    os.getenv("GRAPH_EXTRACTION_WORKERS", _DEFAULT_EXTRACTION_WORKERS)
)

# Shared with retrieval/graph_rag.py's query-entity extraction — same task
# (structured-JSON information extraction), so both use the same system
# message rather than maintaining two copies of the same instruction.
EXTRACTION_SYSTEM_PROMPT = "You are a precise information-extraction system that outputs only JSON."

EXTRACTION_PROMPT = (
    "Extract entity-relationship triplets from the text below for a knowledge "
    "graph. Return ONLY a JSON array of objects, each with keys \"subject\", "
    "\"relation\", \"object\" — no prose, no markdown fences. Keep entity names "
    "short and consistent (e.g. reuse the same surface form for the same "
    "entity across triplets). Extract only factual relationships stated in "
    "the text; skip anything not explicitly present. If there is nothing "
    "worth extracting, return [].\n\n"
    "Text:\n{text}"
)


def _graph_dir(doc_id: str) -> Path:
    return GRAPH_ROOT / doc_id


# Entities are stored as generic `Entity` nodes scoped by a `docId` property
# (so multiple documents can share one Neo4j instance/database without their
# graphs merging), rather than one Neo4j label per document. Relationships
# use a single `RELATION` type with the actual LLM-extracted relation text
# stored as a `type` property — Cypher relationship *types* must be static
# in a query (dynamic types need the APOC plugin, which isn't assumed to be
# installed), so a static type + dynamic property is the portable choice.
def _neo4j_write_triplets(doc_id: str, triplets: list[dict], failed_chunks: int) -> None:
    driver = get_neo4j_driver()
    with driver.session(database=neo4j_database()) as session:
        session.run(
            "CREATE CONSTRAINT entity_doc_name IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.docId, e.name) IS UNIQUE"
        )
        # Replace any previous subgraph for this doc_id so a rebuild doesn't
        # leave stale nodes/relationships behind.
        session.run("MATCH (n:Entity {docId: $doc_id}) DETACH DELETE n", doc_id=doc_id)
        session.run("MATCH (d:Document {docId: $doc_id}) DELETE d", doc_id=doc_id)
        session.run(
            "MERGE (d:Document {docId: $doc_id}) SET d.failedChunks = $failed_chunks",
            doc_id=doc_id,
            failed_chunks=failed_chunks,
        )
        for triplet in triplets:
            session.run(
                """
                MERGE (s:Entity {docId: $doc_id, name: $subject})
                MERGE (o:Entity {docId: $doc_id, name: $object})
                CREATE (s)-[r:RELATION {type: $relation, page: $page}]->(o)
                """,
                doc_id=doc_id,
                subject=triplet["subject"],
                object=triplet["object"],
                relation=triplet["relation"],
                page=triplet.get("page"),
            )


def _neo4j_stats(doc_id: str) -> "GraphIndexStats":
    driver = get_neo4j_driver()
    with driver.session(database=neo4j_database()) as session:
        triplets = session.run(
            "MATCH (:Entity {docId: $doc_id})-[r:RELATION]->(:Entity {docId: $doc_id}) "
            "RETURN count(r) AS c",
            doc_id=doc_id,
        ).single()["c"]
        doc = session.run(
            "MATCH (d:Document {docId: $doc_id}) RETURN d.failedChunks AS f",
            doc_id=doc_id,
        ).single()
    return GraphIndexStats(triplets=triplets, failed_chunks=(doc["f"] if doc else 0) or 0)


def _neo4j_index_exists(doc_id: str) -> bool:
    driver = get_neo4j_driver()
    with driver.session(database=neo4j_database()) as session:
        result = session.run(
            "MATCH (n:Entity {docId: $doc_id}) RETURN count(n) AS c LIMIT 1", doc_id=doc_id
        ).single()
    return bool(result and result["c"] > 0)


def graph_index_exists(doc_id: str) -> bool:
    if neo4j_enabled():
        return _neo4j_index_exists(doc_id)
    return (_graph_dir(doc_id) / "graph.json").exists()


def _parse_triplets(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    triplets = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        if subject and relation and obj:
            triplets.append({"subject": subject, "relation": relation, "object": obj})
    return triplets


def _chunk_key(chunk) -> str:
    """Stable identity for a chunk, used as the checkpoint key. Content-based
    (not index-based) so a resumed run stays correct even if upstream PDF
    parsing or splitting shifts chunk boundaries slightly between runs."""
    return hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()


def _load_checkpoint(checkpoint_file: Path) -> dict[str, list[dict]]:
    if not checkpoint_file.exists():
        return {}
    try:
        return json.loads(checkpoint_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# Entity labels whose lowercase form matches exactly, or whose character-level
# similarity is at or above this threshold, are merged into one canonical
# node. Deliberately higher than graph_rag.py's query-time match threshold
# (0.6) — a false merge here silently conflates two distinct real-world
# entities, which is worse than the fragmentation it's meant to fix, so this
# only catches near-identical surface forms (casing, punctuation, whitespace).
CANONICALIZATION_THRESHOLD = 0.92


def _canonicalize_triplets(triplets: list[dict]) -> list[dict]:
    """Merge near-duplicate entity labels into one canonical surface form per
    cluster. Each chunk is extracted independently, so the same real-world
    entity can come back as slightly different strings from different
    chunks (case, punctuation, minor wording) — left alone, that fragments
    the graph into disconnected nodes for what should be one entity, which
    breaks BFS traversal at query time. This is a cheap string-similarity
    pass (no extra LLM calls), so it doesn't add to ingestion latency.
    """
    labels: list[str] = []
    seen: set[str] = set()
    for t in triplets:
        for label in (t["subject"], t["object"]):
            if label not in seen:
                seen.add(label)
                labels.append(label)

    canonical_for: dict[str, str] = {}
    representatives: list[str] = []

    for label in labels:
        label_lower = label.lower()
        match = None
        for rep in representatives:
            rep_lower = rep.lower()
            if label_lower == rep_lower:
                match = rep
                break
            if SequenceMatcher(None, label_lower, rep_lower).ratio() >= CANONICALIZATION_THRESHOLD:
                match = rep
                break
        if match is None:
            representatives.append(label)
            canonical_for[label] = label
        else:
            canonical_for[label] = match

    return [
        dict(t, subject=canonical_for[t["subject"]], object=canonical_for[t["object"]])
        for t in triplets
    ]


@dataclass
class GraphIndexStats:
    triplets: int
    failed_chunks: int


def build_graph_index(
    pdf_path: str | Path, doc_id: str, pages: list[Document] | None = None
) -> GraphIndexStats:
    """Extract entity-relationship triplets chunk-by-chunk via the LLM, merge
    near-duplicate entity labels, and persist them (to Neo4j, or to
    data/graph_index/<doc_id>/graph.json — see config.neo4j_enabled()).
    Returns the number of triplets extracted and how many chunks failed
    extraction.

    `pages` lets a caller that already loaded the PDF (e.g. to also build the
    vector index) pass the pages in instead of parsing the file twice.

    Progress is checkpointed per-chunk to a sidecar file so an interrupted
    run (crash, rate limit, manual cancel) can resume instead of redoing
    every chunk — the dominant cost of indexing is LLM round-trip latency,
    and a multi-hour job losing all progress on a transient failure is
    expensive in practice. Chunks that fail extraction are deliberately left
    out of the checkpoint so a resumed run retries them instead of
    permanently accepting the failure as "done with zero triplets".
    """
    if pages is None:
        pages = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    if not chunks:
        raise ValueError("No extractable text found in PDF.")

    out_dir = _graph_dir(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = out_dir / "checkpoint.json"
    checkpoint = _load_checkpoint(checkpoint_file)

    llm = get_llm(temperature=0)

    def extract_chunk(chunk) -> tuple[str, int | None, list[dict], bool]:
        key = _chunk_key(chunk)
        page = chunk.metadata.get("page")
        if key in checkpoint:
            return key, page, checkpoint[key], False

        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=EXTRACTION_PROMPT.format(text=chunk.page_content)),
        ]
        try:
            response = llm.invoke(messages)
        except Exception:
            logger.warning(
                "graph_index: extraction failed for a chunk (page=%s) of doc_id=%s",
                page,
                doc_id,
                exc_info=True,
            )
            return key, page, [], True
        return key, page, _parse_triplets(response.content), False

    all_triplets: list[dict] = []
    failed_chunks = 0
    pending_chunks = [c for c in chunks if _chunk_key(c) not in checkpoint]
    if pending_chunks:
        logger.info(
            "graph_index: extracting %d/%d chunks for doc_id=%s (%d already checkpointed)",
            len(pending_chunks),
            len(chunks),
            doc_id,
            len(chunks) - len(pending_chunks),
        )

    # Persist the checkpoint every few chunks rather than after each one —
    # writing the whole checkpoint on every chunk is O(n^2) I/O for large
    # documents, since each write serializes everything seen so far.
    CHECKPOINT_INTERVAL = 5
    with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
        for i, (key, page, triplets, failed) in enumerate(pool.map(extract_chunk, chunks), 1):
            if failed:
                failed_chunks += 1
            else:
                checkpoint[key] = triplets
            if i % CHECKPOINT_INTERVAL == 0 or i == len(chunks):
                checkpoint_file.write_text(json.dumps(checkpoint), encoding="utf-8")
            for triplet in triplets:
                all_triplets.append(dict(triplet, page=page))

    all_triplets = _canonicalize_triplets(all_triplets)

    if neo4j_enabled():
        _neo4j_write_triplets(doc_id, all_triplets, failed_chunks)
    else:
        (out_dir / "graph.json").write_text(
            json.dumps({"triplets": all_triplets, "failed_chunks": failed_chunks}, indent=2),
            encoding="utf-8",
        )
    checkpoint_file.unlink(missing_ok=True)

    return GraphIndexStats(triplets=len(all_triplets), failed_chunks=failed_chunks)


def load_graph_stats(doc_id: str) -> GraphIndexStats:
    """Read triplet/failed-chunk counts for an already-built graph index,
    without loading it into a full NetworkX graph."""
    if neo4j_enabled():
        return _neo4j_stats(doc_id)
    graph_file = _graph_dir(doc_id) / "graph.json"
    data = json.loads(graph_file.read_text(encoding="utf-8"))
    return GraphIndexStats(
        triplets=len(data.get("triplets", [])),
        failed_chunks=data.get("failed_chunks", 0),
    )


def load_graph(doc_id: str) -> nx.MultiDiGraph:
    graph_file = _graph_dir(doc_id) / "graph.json"
    if not graph_file.exists():
        raise FileNotFoundError(f"No graph index found for doc_id={doc_id!r}")

    data = json.loads(graph_file.read_text(encoding="utf-8"))
    graph = nx.MultiDiGraph()
    for triplet in data.get("triplets", []):
        graph.add_node(triplet["subject"])
        graph.add_node(triplet["object"])
        graph.add_edge(
            triplet["subject"],
            triplet["object"],
            relation=triplet["relation"],
            page=triplet.get("page"),
        )
    return graph


def list_graph_docs() -> list[str]:
    if not GRAPH_ROOT.exists():
        return []
    return sorted(p.name for p in GRAPH_ROOT.iterdir() if (p / "graph.json").exists())
