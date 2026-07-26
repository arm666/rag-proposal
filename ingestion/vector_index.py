"""Stage 1 (Ingestion) of the Vector RAG pipeline: chunk a PDF, embed the
chunks, and persist them to a FAISS index so retrieval always runs against a
stable, pre-built representation (thesis §3.3.1)."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
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


def build_index(pdf_path: str | Path, doc_id: str) -> int:
    """Load a PDF, chunk it page-aware, embed the chunks, and persist a FAISS
    index under data/faiss_index/<doc_id>/. Returns the number of chunks indexed.
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

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    store = FAISS.from_documents(chunks, get_embeddings())

    out_dir = _index_dir(doc_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out_dir))

    return len(chunks)


def load_index(doc_id: str) -> FAISS:
    out_dir = _index_dir(doc_id)
    if not (out_dir / "index.faiss").exists():
        raise FileNotFoundError(f"No FAISS index found for doc_id={doc_id!r}")
    return FAISS.load_local(
        str(out_dir), get_embeddings(), allow_dangerous_deserialization=True
    )
