"""Postgres + pgvector storage for chunks. Raw SQL, no ORM.

One table holds each chunk's text, its metadata (for filtering and citing) and, from
the next PR on, its embedding. Filtering by status or language is then a plain WHERE.
"""

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import get_settings
from app.domain import Chunk


def connect() -> psycopg.Connection:
    conn = psycopg.connect(get_settings().database_url)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # lets psycopg send and receive `vector` values as lists/arrays
    return conn


def schema_sql(dimension: int) -> str:
    return f"""
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE IF NOT EXISTS chunks (
        id            text PRIMARY KEY,           -- e.g. 'dadm-en:6.3.7:0'
        doc_id        text NOT NULL,
        title         text NOT NULL,
        section       text,
        headings      text[] NOT NULL DEFAULT '{{}}',
        language      text NOT NULL CHECK (language IN ('en', 'fr')),
        status        text NOT NULL CHECK (status IN ('current', 'superseded')),
        access_level  text,
        position      integer NOT NULL,           -- reading order within the document
        text          text NOT NULL,
        -- Filled in by ingest once embedding is added. Every vector records the model
        -- that produced it: vectors from different models must never be compared.
        embedding     vector({dimension}),
        embed_model   text
    );

    CREATE INDEX IF NOT EXISTS chunks_doc_id ON chunks (doc_id);
    """


def init_schema(conn: psycopg.Connection) -> None:
    conn.execute(schema_sql(get_settings().embed_dimension))
    conn.commit()


def replace_document_chunks(
    conn: psycopg.Connection,
    doc_id: str,
    chunks: list[Chunk],
    embeddings: list[list[float]] | None = None,
    embed_model: str | None = None,
) -> None:
    """Swap in a document's chunks: delete the old ones, insert the new, in one transaction.

    Replacing (rather than updating row by row) means a different chunking strategy
    can't leave stale chunks behind, and a failure halfway leaves the old rows intact.
    `embeddings`, if given, must line up one-to-one with `chunks`.
    """
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    vectors = embeddings or [None] * len(chunks)
    model = embed_model if embeddings is not None else None

    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                    (id, doc_id, title, section, headings, language, status, access_level,
                     position, text, embedding, embed_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (c.id, c.doc_id, c.title, c.section, c.headings, c.language,
                     c.status, c.access_level, position, c.text,
                     None if v is None else Vector(v), model)
                    for position, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )


def other_embed_models(conn: psycopg.Connection, model: str) -> list[str]:
    """Embedding models other than `model` that already have vectors in the table."""
    rows = conn.execute(
        "SELECT DISTINCT embed_model FROM chunks WHERE embed_model IS NOT NULL AND embed_model <> %s",
        (model,),
    ).fetchall()
    return [r[0] for r in rows]


def drop_chunks_table(conn: psycopg.Connection) -> None:
    """For a full re-index, e.g. after changing the embedding model or dimension."""
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.commit()


def chunk_counts(conn: psycopg.Connection) -> list[tuple[str, int, int]]:
    """(doc_id, chunks, chunks with an embedding) for each document."""
    return conn.execute(
        """
        SELECT doc_id, count(*), count(embedding)
        FROM chunks GROUP BY doc_id ORDER BY doc_id
        """
    ).fetchall()
