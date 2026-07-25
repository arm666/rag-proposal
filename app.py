"""Web backend for the Vector RAG chat UI.

Exposes:
  POST /api/upload  — upload a PDF, ingest it into a FAISS index (Stage 1)
  POST /api/chat     — ask a question against an ingested PDF (Stage 2+3)
  GET  /api/docs      — list already-ingested documents
  GET  /            — chat UI (static/index.html)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ingestion.vector_index import build_index, index_exists, list_indexed_docs

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vector RAG Chat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    doc_id: str
    question: str


class ChatSource(BaseModel):
    chunk_id: int
    page: int | None
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    chunks: int


@app.post("/api/upload", response_model=DocInfo)
async def upload_pdf(file: UploadFile = File(...)) -> DocInfo:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    doc_id = hashlib.sha256(raw).hexdigest()[:16]
    pdf_path = UPLOAD_DIR / f"{doc_id}.pdf"

    if not index_exists(doc_id):
        pdf_path.write_bytes(raw)
        try:
            chunks = build_index(pdf_path, doc_id)
        except Exception as exc:
            pdf_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Failed to index PDF: {exc}") from exc
    else:
        # Already indexed (same content hash) — just report chunk count via a fresh load.
        from ingestion.vector_index import load_index

        chunks = load_index(doc_id).index.ntotal

    return DocInfo(doc_id=doc_id, filename=file.filename, chunks=chunks)


@app.get("/api/docs", response_model=list[str])
async def get_docs() -> list[str]:
    return list_indexed_docs()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not index_exists(req.doc_id):
        raise HTTPException(status_code=404, detail="Unknown doc_id — upload the PDF first.")

    from retrieval.vector_rag import answer_question

    try:
        result = answer_question(req.doc_id, req.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc

    return ChatResponse(
        answer=result.answer,
        sources=[ChatSource(**s.__dict__) for s in result.sources],
        latency_seconds=result.latency_seconds,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_tokens=result.total_tokens,
        cost_usd=result.cost_usd,
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
