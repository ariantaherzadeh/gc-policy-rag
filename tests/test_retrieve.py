"""Vector search tests against a real Postgres (skipped if it isn't running).

Uses hand-made vectors instead of real embeddings, so the expected ranking is known exactly.
"""

import math

import pytest

from app import db
from app.config import get_settings
from app.domain import Chunk
from app.retrieve import Filters, retrieve, search

DIM = get_settings().embed_dimension
MODEL = "test-model"


def direction(angle_degrees: float) -> list[float]:
    """A unit vector at `angle_degrees` from the x-axis, padded to the index dimension.

    Cosine similarity between two of these is just cos(angle between them), which
    makes expected scores easy to reason about.
    """
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)] + [0.0] * (DIM - 2)


def chunk(id: str, language="en", status="current", access_level=None) -> Chunk:
    return Chunk(id=id, doc_id=id.split(":")[0], title="Doc", section="1", language=language,
                 status=status, access_level=access_level, text=f"text of {id}")


def store(conn, doc_id, chunks, angles, model=MODEL):
    db.replace_document_chunks(conn, doc_id, chunks, [direction(a) for a in angles], model)


def test_results_are_ranked_by_cosine_similarity(conn):
    store(conn, "doc", [chunk("doc:far"), chunk("doc:near"), chunk("doc:middle")], [80, 10, 45])
    results = search(conn, direction(0), MODEL, top_k=3)
    assert [c.id for c in results] == ["doc:near", "doc:middle", "doc:far"]
    assert results[0].score == pytest.approx(math.cos(math.radians(10)), abs=1e-6)


def test_top_k_limits_the_results(conn):
    store(conn, "doc", [chunk(f"doc:{i}") for i in range(5)], [0, 10, 20, 30, 40])
    assert len(search(conn, direction(0), MODEL, top_k=2)) == 2


def test_superseded_chunks_are_excluded_by_default(conn):
    store(conn, "new", [chunk("new:a")], [30])
    store(conn, "old", [chunk("old:a", status="superseded")], [0])  # closer, but superseded
    assert [c.id for c in search(conn, direction(0), MODEL, top_k=5)] == ["new:a"]

    everything = search(conn, direction(0), MODEL, top_k=5, filters=Filters(statuses=("current", "superseded")))
    assert [c.id for c in everything] == ["old:a", "new:a"]


def test_language_filter(conn):
    store(conn, "en", [chunk("en:a")], [0])
    store(conn, "fr", [chunk("fr:a", language="fr")], [5])
    assert [c.id for c in search(conn, direction(0), MODEL, 5, Filters(language="fr"))] == ["fr:a"]


def test_access_filter_keeps_open_chunks(conn):
    store(conn, "doc", [chunk("doc:open"), chunk("doc:secret", access_level="protected-b")], [0, 5])
    visible = search(conn, direction(0), MODEL, 5, Filters(access_levels=("unclassified",)))
    assert [c.id for c in visible] == ["doc:open"]


def test_vectors_from_another_model_are_never_compared(conn):
    store(conn, "doc", [chunk("doc:a")], [0], model="some-other-model")
    assert search(conn, direction(0), MODEL, top_k=5) == []


def test_retrieve_embeds_the_question_as_a_query(conn):
    store(conn, "doc", [chunk("doc:a")], [0], model=get_settings().embed_model)

    class RecordingEmbedder:
        def embed(self, texts, kind):
            self.kind = kind
            return [direction(0) for _ in texts]

    embedder = RecordingEmbedder()
    results = retrieve("a question?", embedder, conn, top_k=5)
    assert embedder.kind == "query"
    assert [c.id for c in results] == ["doc:a"]
