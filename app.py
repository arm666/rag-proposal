"""Web backend for the Vector RAG / Graph RAG chat UI.

Exposes:
  POST   /api/upload  — upload a PDF, build/reuse its FAISS + graph indexes (Stage 1)
  POST   /api/chat     — ask a question against an ingested PDF (Stage 2+3),
                        against the vector pipeline, the graph pipeline, or both
  GET    /api/docs      — list already-ingested documents
  DELETE /api/docs      — clear all uploaded files and indexes from local storage
  GET    /            — chat UI (static/index.html)

All uploads and indexes are persisted under data/ on the local filesystem
(data/uploads, data/faiss_index, data/graph_index), keyed by a content hash
of the PDF. Because doc_id is derived from file content and existence is
checked on disk, re-uploading the same PDF after a server restart reuses the
already-built indexes instead of rebuilding them.

Document metadata (filename, chunk/triplet counts) is tracked in a local
SQLite registry (data/docs.db, see db.py) so GET /api/docs can tell a client
what's already indexed without it having to re-upload anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel

from config import get_neo4j_driver, neo4j_database, neo4j_enabled
from db import clear_documents, list_documents, upsert_document
from ingestion.graph_index import build_graph_index, graph_index_exists, load_graph_stats
from ingestion.vector_index import build_index, index_exists, load_index

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vector RAG / Graph RAG Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Mode = Literal["vector", "graph", "both"]


class ChatRequest(BaseModel):
    doc_id: str
    question: str
    mode: Mode = "vector"


class ChatSource(BaseModel):
    chunk_id: int
    page: int | None
    text: str
    score: float


class ChatTriplet(BaseModel):
    subject: str
    relation: str
    object: str
    page: int | None


class PipelineResult(BaseModel):
    answer: str
    sources: list[ChatSource] = []
    triplets: list[ChatTriplet] = []
    matched_entities: list[str] = []
    latency_seconds: float


class ChatResponse(BaseModel):
    mode: Mode
    vector: PipelineResult | None = None
    graph: PipelineResult | None = None


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    triplets: int
    failed_chunks: int = 0


@app.post("/api/upload", response_model=DocInfo)
async def upload_pdf(file: UploadFile = File(...)) -> DocInfo:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"

    if not pdf_path.exists():
        # UPLOAD_DIR is only created once at import time, so if data/ gets
        # deleted while the server keeps running, re-create it here rather
        # than letting write_bytes fail with an unhandled FileNotFoundError.
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(raw)

    need_vector = not index_exists(doc_id)
    need_graph = not graph_index_exists(doc_id)

    # Both build steps are blocking (PDF parsing, embedding, LLM calls), so
    # every awaited call below runs on a worker thread rather than the
    # asyncio event loop — otherwise one upload would freeze every other
    # request the server is handling. Parsing the PDF once and handing the
    # same pages to both indexers also avoids loading it twice.
    pages = None
    if need_vector or need_graph:
        pages = await asyncio.to_thread(lambda: PyPDFLoader(str(pdf_path)).load())

    async def build_vector() -> int:
        if not need_vector:
            return (await asyncio.to_thread(load_index, doc_id)).index.ntotal
        try:
            return await asyncio.to_thread(build_index, pdf_path, doc_id, pages)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build vector index: {exc}") from exc

    async def build_graph():
        if not need_graph:
            return await asyncio.to_thread(load_graph_stats, doc_id)
        try:
            return await asyncio.to_thread(build_graph_index, pdf_path, doc_id, pages)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build graph index: {exc}") from exc

    # Vector and graph indexing don't depend on each other's output, so run
    # them concurrently instead of one after the other.
    chunks, graph_stats = await asyncio.gather(build_vector(), build_graph())

    upsert_document(doc_id, file.filename, chunks, graph_stats.triplets, graph_stats.failed_chunks)
    return DocInfo(
        doc_id=doc_id,
        filename=file.filename,
        chunks=chunks,
        triplets=graph_stats.triplets,
        failed_chunks=graph_stats.failed_chunks,
    )


@app.get("/api/docs", response_model=list[DocInfo])
async def get_docs() -> list[DocInfo]:
    """List already-ingested documents, from the local doc registry — filtered
    to entries whose vector index is still present on disk."""
    return [DocInfo(**d) for d in list_documents() if index_exists(d["doc_id"])]


@app.delete("/api/docs")
async def clear_docs() -> dict[str, bool]:
    """Remove all uploaded PDFs, built indexes, and the doc registry from local storage."""
    for directory in (UPLOAD_DIR, BASE_DIR / "data" / "faiss_index", BASE_DIR / "data" / "graph_index"):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    if neo4j_enabled():
        driver = get_neo4j_driver()
        with driver.session(database=neo4j_database()) as session:
            await asyncio.to_thread(session.run, "MATCH (n) DETACH DELETE n")
    clear_documents()
    return {"cleared": True}


async def _run_vector_pipeline(doc_id: str, question: str) -> PipelineResult:
    from retrieval.vector_rag import answer_question as vector_answer

    try:
        result = await asyncio.to_thread(vector_answer, doc_id, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Vector RAG failed: {exc}") from exc
    return PipelineResult(
        answer=result.answer,
        sources=[ChatSource(**s.__dict__) for s in result.sources],
        latency_seconds=result.latency_seconds,
    )


async def _run_graph_pipeline(doc_id: str, question: str) -> PipelineResult:
    from retrieval.graph_rag import answer_question as graph_answer

    try:
        result = await asyncio.to_thread(graph_answer, doc_id, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Graph RAG failed: {exc}") from exc
    return PipelineResult(
        answer=result.answer,
        triplets=[ChatTriplet(**t.__dict__) for t in result.triplets],
        matched_entities=result.matched_entities,
        latency_seconds=result.latency_seconds,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not index_exists(req.doc_id):
        raise HTTPException(status_code=404, detail="Unknown doc_id — upload the PDF first.")
    if req.mode in ("graph", "both") and not graph_index_exists(req.doc_id):
        raise HTTPException(status_code=404, detail="No graph index for this document — re-upload it.")

    # Both pipelines are blocking (embedding/LLM calls), so they're offloaded
    # to worker threads rather than run directly on the event loop — and,
    # in "both" mode, run concurrently instead of one after the other since
    # they're independent of each other.
    coros = []
    if req.mode in ("vector", "both"):
        coros.append(_run_vector_pipeline(req.doc_id, req.question))
    if req.mode in ("graph", "both"):
        coros.append(_run_graph_pipeline(req.doc_id, req.question))

    results = await asyncio.gather(*coros)

    results_iter = iter(results)
    vector_result = next(results_iter) if req.mode in ("vector", "both") else None
    graph_result = next(results_iter) if req.mode in ("graph", "both") else None

    return ChatResponse(mode=req.mode, vector=vector_result, graph=graph_result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
