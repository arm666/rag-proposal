"""Stage 1 (Ingestion) of the Graph RAG pipeline: extract entity-relationship
triplets from a PDF with the LLM backbone, build a NetworkX knowledge graph,
and persist it as JSON so retrieval always runs against a stable, pre-built
representation (thesis §3.3.1)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_llm

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


def graph_index_exists(doc_id: str) -> bool:
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


def build_graph_index(pdf_path: str | Path, doc_id: str) -> int:
    """Load a PDF, extract entity-relationship triplets chunk-by-chunk via the
    LLM, build a NetworkX directed multigraph, and persist it as
    data/graph_index/<doc_id>/graph.json. Returns the number of triplets extracted.

    Progress is checkpointed per-chunk to a sidecar file so an interrupted
    run (crash, rate limit, manual cancel) can resume instead of redoing
    every chunk — the dominant cost of indexing is LLM round-trip latency,
    and a multi-hour job losing all progress on a transient failure is
    expensive in practice.
    """
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

    def extract_chunk(chunk) -> tuple[str, int | None, list[dict]]:
        key = _chunk_key(chunk)
        page = chunk.metadata.get("page")
        if key in checkpoint:
            return key, page, checkpoint[key]

        messages = [
            SystemMessage(
                content="You are a precise information-extraction system that outputs only JSON."
            ),
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
            return key, page, []
        return key, page, _parse_triplets(response.content)

    graph = nx.MultiDiGraph()
    all_triplets: list[dict] = []
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
        for i, (key, page, triplets) in enumerate(pool.map(extract_chunk, chunks), 1):
            checkpoint[key] = triplets
            if i % CHECKPOINT_INTERVAL == 0 or i == len(chunks):
                checkpoint_file.write_text(json.dumps(checkpoint), encoding="utf-8")
            for triplet in triplets:
                triplet = dict(triplet, page=page)
                all_triplets.append(triplet)
                graph.add_node(triplet["subject"])
                graph.add_node(triplet["object"])
                graph.add_edge(
                    triplet["subject"],
                    triplet["object"],
                    relation=triplet["relation"],
                    page=page,
                )

    (out_dir / "graph.json").write_text(
        json.dumps({"triplets": all_triplets}, indent=2), encoding="utf-8"
    )
    checkpoint_file.unlink(missing_ok=True)

    return len(all_triplets)


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
