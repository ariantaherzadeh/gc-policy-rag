"""Embedding tests. All offline: a fake embedder or a fake Cohere client, never the real API."""

from types import SimpleNamespace

from app.providers.cache import CachedEmbedder
from app.providers.cohere import CohereEmbedder
from tests.fakes import FakeEmbedder


class CountingEmbedder(FakeEmbedder):
    def __init__(self):
        self.received: list[list[str]] = []

    def embed(self, texts, kind):
        self.received.append(list(texts))
        return super().embed(texts, kind)


def cached(tmp_path, inner=None):
    return CachedEmbedder(inner or CountingEmbedder(), tmp_path, model="fake", dimension=26)


def test_second_run_is_served_from_the_cache(tmp_path):
    embedder = cached(tmp_path)
    first = embedder.embed(["alpha", "beta"], "doc")
    second = embedder.embed(["alpha", "beta"], "doc")
    assert first == second
    assert embedder.inner.received == [["alpha", "beta"]]  # only one call to the inner embedder


def test_only_new_texts_are_sent(tmp_path):
    embedder = cached(tmp_path)
    embedder.embed(["alpha"], "doc")
    embedder.embed(["alpha", "gamma"], "doc")
    assert embedder.inner.received == [["alpha"], ["gamma"]]


def test_duplicate_texts_are_sent_once(tmp_path):
    embedder = cached(tmp_path)
    vectors = embedder.embed(["same", "same"], "doc")
    assert embedder.inner.received == [["same"]]
    assert vectors[0] == vectors[1]


def test_doc_and_query_embeddings_are_cached_separately(tmp_path):
    embedder = cached(tmp_path)
    embedder.embed(["alpha"], "doc")
    assert embedder.uncached(["alpha"], "query") == ["alpha"]


def test_cache_survives_a_new_embedder_instance(tmp_path):
    cached(tmp_path).embed(["alpha"], "doc")
    fresh = cached(tmp_path)
    fresh.embed(["alpha"], "doc")
    assert fresh.inner.received == []


def test_different_models_never_share_cache_entries(tmp_path):
    CachedEmbedder(CountingEmbedder(), tmp_path, model="model-a", dimension=26).embed(["alpha"], "doc")
    other = CachedEmbedder(CountingEmbedder(), tmp_path, model="model-b", dimension=26)
    assert other.uncached(["alpha"], "doc") == ["alpha"]


class FakeCohereClient:
    """Records what CohereEmbedder sends, and returns one dummy vector per text."""

    def __init__(self):
        self.requests = []

    def embed(self, **kwargs):
        self.requests.append(kwargs)
        vectors = [[0.0, 1.0] for _ in kwargs["texts"]]
        return SimpleNamespace(embeddings=SimpleNamespace(float_=vectors))


def test_cohere_embedder_batches_at_96_texts_per_call():
    client = FakeCohereClient()
    embedder = CohereEmbedder(client, model="embed-v4.0", dimension=1536)
    vectors = embedder.embed([f"text {i}" for i in range(200)], "doc")

    assert [len(r["texts"]) for r in client.requests] == [96, 96, 8]
    assert embedder.calls == 3
    assert len(vectors) == 200


def test_cohere_embedder_sends_the_right_parameters():
    client = FakeCohereClient()
    CohereEmbedder(client, model="embed-v4.0", dimension=1536).embed(["a question?"], "query")
    request = client.requests[0]
    assert request["model"] == "embed-v4.0"
    assert request["input_type"] == "search_query"
    assert request["output_dimension"] == 1536
    assert request["embedding_types"] == ["float"]
