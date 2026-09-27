from app.config import Settings
from app.domain import Chunk
from app.providers.base import Embedder, Generator, Reranker
from tests.fakes import FakeEmbedder, FakeGenerator, FakeReranker


def make_chunk(id: str, text: str) -> Chunk:
    return Chunk(id=id, doc_id="doc", title="Doc", language="en", status="current", text=text)


def test_fakes_satisfy_the_provider_interfaces():
    # Protocols are structural: no inheritance, just matching methods.
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeReranker(), Reranker)
    assert isinstance(FakeGenerator(), Generator)


def test_reranker_orders_by_relevance_and_trims_to_top_n():
    chunks = [
        make_chunk("a", "security screening for contractors"),
        make_chunk("b", "impact assessment before deploying an automated system"),
        make_chunk("c", "automated decision system impact assessment requirements"),
    ]
    top = FakeReranker().rerank("automated system impact assessment", chunks, top_n=2)
    assert [c.id for c in top] == ["b", "c"]
    assert all(c.score is not None for c in top)


def test_generator_abstains_without_chunks():
    answer = FakeGenerator().answer("anything?", chunks=[], history=[])
    assert answer.text == "I don't know."
    assert answer.citations == []


def test_api_key_is_masked(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "not-a-real-key")
    settings = Settings(_env_file=None)
    assert "not-a-real-key" not in repr(settings)
    assert settings.cohere_api_key.get_secret_value() == "not-a-real-key"


def test_settings_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("RERANK_TOP_N", "5")
    settings = Settings(_env_file=None)
    assert settings.rerank_enabled is False
    assert settings.rerank_top_n == 5
