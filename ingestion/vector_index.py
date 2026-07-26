"""Stage 1 (Ingestion) of the Vector RAG pipeline: chunk a PDF, embed the
chunks, and persist them to a FAISS index so retrieval always runs against a
stable, pre-built representation (thesis §3.3.1)."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_embeddings

INDEX_ROOT = Path(__file__).resolve().parent.parent / "data" / "faiss_index"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _index_dir(doc_id: str) -> Path:
    return INDEX_ROOT / doc_id


def index_exists(doc_id: str) -> bool:
    return (_index_dir(doc_id) / "index.faiss").exists()


def build_index(pdf_path: str | Path, doc_id: str, pages: list[Document] | None = None) -> int:
    """Chunk a PDF page-aware, embed the chunks, and persist a FAISS index
    under data/faiss_index/<doc_id>/. Returns the number of chunks indexed.

    `pages` lets a caller that already loaded the PDF (e.g. to also build the
    graph index) pass the pages in instead of parsing the file twice.
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

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    # This langchain_community version always builds a plain IndexFlatL2
    # regardless of distance_strategy (only MAX_INNER_PRODUCT gets a
    # different index type), and _euclidean_relevance_score_fn is the one
    # explicitly derived for L2 distance over *normalized* vectors (see its
    # docstring) — so normalize_L2=True with the default EUCLIDEAN_DISTANCE
    # is the combination that actually keeps the relevance score calibrated,
    # regardless of whether the embedding model returns unit-normed vectors
    # (OpenAI's are; many local/HF models aren't). distance_strategy=COSINE
    # would swap in a relevance function that assumes the raw distance is
    # already a cosine distance, which it isn't here.
    store = FAISS.from_documents(chunks, get_embeddings(), normalize_L2=True)

    out_dir = _index_dir(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))

    return len(chunks)


def load_index(doc_id: str) -> FAISS:
    out_dir = _index_dir(doc_id)
    if not (out_dir / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found for doc_id={doc_id!r}")
    return FAISS.load_local(
        str(out_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )
