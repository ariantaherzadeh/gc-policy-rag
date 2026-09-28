"""Database tests. They need Postgres running (docker compose up -d db) and are skipped otherwise.

Each test gets its own temporary Postgres schema (see the `conn` fixture in conftest.py),
so the real `chunks` table is untouched.
"""

import psycopg
import pytest

from app import db
from app.config import get_settings
from app.domain import Chunk


def make_chunk(id: str, doc_id: str = "doc-en", text: str = "text") -> Chunk:
    return Chunk(id=id, doc_id=doc_id, title="Doc", section="1.1", headings=["1. Intro"],
                 language="en", status="current", text=text)


def test_chunks_round_trip_with_their_metadata(conn):
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:1.1:0", text="Hello")])
    row = conn.execute("SELECT id, section, headings, language, status, position, text FROM chunks").fetchone()
    assert row == ("doc-en:1.1:0", "1.1", ["1. Intro"], "en", "current", 0, "Hello")


def test_replacing_a_document_removes_its_old_chunks(conn):
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:a"), make_chunk("doc-en:b")])
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:c")])
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks").fetchall()]
    assert ids == ["doc-en:c"]


def test_replacing_one_document_leaves_others_alone(conn):
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:a")])
    db.replace_document_chunks(conn, "doc-fr", [make_chunk("doc-fr:a", doc_id="doc-fr")])
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:b")])
    assert db.chunk_counts(conn) == [("doc-en", 1, 0), ("doc-fr", 1, 0)]


def test_embeddings_are_stored_with_their_model(conn):
    vector = [0.5] * get_settings().embed_dimension
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:a")], [vector], "embed-v4.0")
    stored, model = conn.execute("SELECT embedding, embed_model FROM chunks").fetchone()
    assert stored.to_list() == vector
    assert model == "embed-v4.0"
    assert db.chunk_counts(conn) == [("doc-en", 1, 1)]


def test_other_embedding_models_are_detected(conn):
    vector = [0.5] * get_settings().embed_dimension
    db.replace_document_chunks(conn, "doc-en", [make_chunk("doc-en:a")], [vector], "old-model")
    assert db.other_embed_models(conn, "embed-v4.0") == ["old-model"]
    assert db.other_embed_models(conn, "old-model") == []


def test_chunks_and_embeddings_must_line_up(conn):
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        db.replace_document_chunks(conn, "doc-en", [make_chunk("a"), make_chunk("b")], [[0.1]], "m")


def test_invalid_language_is_rejected_by_the_database(conn):
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(
            "INSERT INTO chunks (id, doc_id, title, language, status, position, text) "
            "VALUES ('x', 'd', 't', 'de', 'current', 0, 'text')"
        )
