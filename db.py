"""Local SQLite store for ingested-document metadata.

Filenames and chunk/triplet counts only ever existed in the /api/upload
response — nothing persisted them, so a page refresh (or server restart)
had no way to know what was already indexed. This gives GET /api/docs a
durable source of truth to read from instead of just listing doc_id
directories on disk.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data" / "docs.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            chunks INTEGER NOT NULL,
            triplets INTEGER NOT NULL,
            failed_chunks INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    try:
        conn.execute("ALTER TABLE documents ADD COLUMN failed_chunks INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists (pre-existing db from before this field was added)
    return conn


def upsert_document(
    doc_id: str, filename: str, chunks: int, triplets: int, failed_chunks: int = 0
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (doc_id, filename, chunks, triplets, failed_chunks)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                filename = excluded.filename,
                chunks = excluded.chunks,
                triplets = excluded.triplets,
                failed_chunks = excluded.failed_chunks
            """,
            (doc_id, filename, chunks, triplets, failed_chunks),
        )


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename, chunks, triplets, failed_chunks FROM documents ORDER BY uploaded_at ASC"
        ).fetchall()
    return [
        {"doc_id": r[0], "filename": r[1], "chunks": r[2], "triplets": r[3], "failed_chunks": r[4]}
        for r in rows
    ]


def clear_documents() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents")
