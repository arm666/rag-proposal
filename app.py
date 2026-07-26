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

import hashlib
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db import clear_documents, list_documents, upsert_document
from ingestion.graph_index import build_graph_index, graph_index_exists
from ingestion.vector_index import build_index, index_exists

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
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class ChatResponse(BaseModel):
    mode: Mode
    vector: PipelineResult | None = None
    graph: PipelineResult | None = None


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int
    triplets: int


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
        pdf_path.write_bytes(raw)

    if not index_exists(doc_id):
        try:
            chunks = build_index(pdf_path, doc_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build vector index: {exc}") from exc
    else:
        from ingestion.vector_index import load_index

        chunks = load_index(doc_id).index.ntotal

    if not graph_index_exists(doc_id):
        try:
            triplets = build_graph_index(pdf_path, doc_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to build graph index: {exc}") from exc
    else:
        import json

        graph_file = BASE_DIR / "data" / "graph_index" / doc_id / "graph.json"
        triplets = len(json.loads(graph_file.read_text(encoding="utf-8")).get("triplets", []))

    upsert_document(doc_id, file.filename, chunks, triplets)
    return DocInfo(doc_id=doc_id, filename=file.filename, chunks=chunks, triplets=triplets)


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
    clear_documents()
    return {"cleared": True}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not index_exists(req.doc_id):
        raise HTTPException(status_code=404, detail="Unknown doc_id — upload the PDF first.")

    vector_result = None
    graph_result = None

    if req.mode in ("vector", "both"):
        from retrieval.vector_rag import answer_question as vector_answer

        try:
            result = vector_answer(req.doc_id, req.question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Vector RAG failed: {exc}") from exc
        vector_result = PipelineResult(
            answer=result.answer,
            sources=[ChatSource(**s.__dict__) for s in result.sources],
            latency_seconds=result.latency_seconds,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
        )

    if req.mode in ("graph", "both"):
        if not graph_index_exists(req.doc_id):
            raise HTTPException(status_code=404, detail="No graph index for this document — re-upload it.")
        from retrieval.graph_rag import answer_question as graph_answer

        try:
            result = graph_answer(req.doc_id, req.question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Graph RAG failed: {exc}") from exc
        graph_result = PipelineResult(
            answer=result.answer,
            triplets=[ChatTriplet(**t.__dict__) for t in result.triplets],
            matched_entities=result.matched_entities,
            latency_seconds=result.latency_seconds,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
            cost_usd=result.cost_usd,
        )

    return ChatResponse(mode=req.mode, vector=vector_result, graph=graph_result)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
