"""Follow-up rewriting and abstention. Offline: fake providers, real Postgres (skipped if not running)."""

from types import SimpleNamespace as NS

import pytest

from app import db
from app.chat import NOT_FOUND, answer_question, is_abstention
from app.config import get_settings
from app.domain import Answer, Chunk, Citation, Turn
from app.providers.cohere import CohereRewriter
from tests.fakes import FakeGenerator, FakeReranker, FakeRewriter


@pytest.fixture
def stored(conn):
    """One stored chunk, and an embedder that records what it was asked to embed."""
    dim = get_settings().embed_dimension
    chunk = Chunk(id="d:6.3.7:0", doc_id="d", title="Directive", section="6.3.7", language="en",
                  status="current", text="peer review by qualified experts")
    db.replace_document_chunks(conn, "d", [chunk], [[1.0] + [0.0] * (dim - 1)], get_settings().embed_model)

    class RecordingEmbedder:
        def __init__(self):
            self.queries = []

        def embed(self, texts, kind):
            self.queries.extend(texts)
            return [[1.0] + [0.0] * (dim - 1) for _ in texts]

    return conn, RecordingEmbedder()


class RecordingGenerator(FakeGenerator):
    def __init__(self):
        self.questions = []

    def answer(self, question, chunks, history):
        self.questions.append(question)
        return super().answer(question, chunks, history)


def ask(stored, question, history=(), **overrides):
    conn, embedder = stored
    parts = dict(embedder=embedder, reranker=FakeReranker(), generator=RecordingGenerator(), rewriter=FakeRewriter())
    parts.update(overrides)
    result = answer_question(question, list(history), conn, **parts)
    return result, parts


def test_first_question_is_searched_as_typed(stored):
    result, parts = ask(stored, "Is peer review required?")
    assert parts["rewriter"].calls == 0
    assert result.search_query == "Is peer review required?"
    assert parts["embedder"].queries == ["Is peer review required?"]


def test_follow_up_is_rewritten_before_retrieval(stored):
    history = [Turn(role="user", content="peer review"), Turn(role="assistant", content="Yes, at Level II.")]
    result, parts = ask(stored, "And Level I?", history)
    assert parts["rewriter"].calls == 1
    assert result.search_query == "And Level I? (about: peer review)"
    assert parts["embedder"].queries == ["And Level I? (about: peer review)"]  # search used the rewrite
    assert parts["generator"].questions == ["And Level I?"]  # Command sees the original question


def test_a_cited_answer_is_not_an_abstention(stored):
    result, _ = ask(stored, "peer review?")
    assert result.abstained is False
    assert result.abstain_reason is None


def test_low_rerank_score_abstains_without_calling_chat(stored, monkeypatch):
    monkeypatch.setattr(get_settings(), "min_rerank_score", 5.0)  # FakeReranker scores are word overlaps
    result, parts = ask(stored, "peer review?")
    assert result.abstained is True
    assert result.answer.text == NOT_FOUND
    assert result.abstain_reason.startswith("best rerank score")
    assert parts["generator"].questions == []  # no Chat call spent


def test_threshold_is_ignored_when_rerank_is_off(stored, monkeypatch):
    monkeypatch.setattr(get_settings(), "min_rerank_score", 5.0)
    result, parts = ask(stored, "peer review?", rerank=False)
    assert result.abstained is False
    assert parts["generator"].questions == ["peer review?"]


class FakeEmbedder1536:
    def embed(self, texts, kind):
        return [[1.0] + [0.0] * (get_settings().embed_dimension - 1) for _ in texts]


def test_no_retrieved_passages_abstains_without_calling_chat(conn):
    result = answer_question("anything?", [], conn, embedder=FakeEmbedder1536(), reranker=FakeReranker(),
                             generator=RecordingGenerator(), rewriter=FakeRewriter())
    assert result.abstained is True
    assert result.abstain_reason == "no passages retrieved"


def test_uncited_answer_counts_as_abstention():
    assert is_abstention(Answer(text=NOT_FOUND)) is True
    assert is_abstention(Answer(text="Yes.", citations=[Citation(chunk_id="a", start=0, end=3, text="Yes")])) is False


class FakeChatClient:
    def __init__(self, reply):
        self.reply = reply
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        return NS(message=NS(content=[NS(type="text", text=self.reply)], citations=None))


def test_cohere_rewriter_sends_the_conversation_and_returns_the_rewrite():
    client = FakeChatClient("  Does peer review apply to Level I systems?  ")
    history = [Turn(role="user", content="Is peer review required?"), Turn(role="assistant", content="At Level II.")]
    rewritten = CohereRewriter(client, "command-a-03-2025").rewrite("And Level I?", history)

    assert rewritten == "Does peer review apply to Level I systems?"
    prompt = client.requests[0]["messages"][-1]["content"]
    assert "user: Is peer review required?" in prompt
    assert "Latest question: And Level I?" in prompt


def test_cohere_rewriter_falls_back_to_the_question_on_an_empty_reply():
    rewriter = CohereRewriter(FakeChatClient(""), "m")
    assert rewriter.rewrite("And Level I?", [Turn(role="user", content="x")]) == "And Level I?"
