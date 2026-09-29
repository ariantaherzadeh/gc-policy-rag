"""Rerank, generation and the full pipeline. All offline: fake Cohere clients and fake providers."""

from types import SimpleNamespace as NS

from app import db
from app.chat import answer_question, render
from app.config import get_settings
from app.domain import Answer, Chunk, Citation, Turn
from app.providers.cohere import SYSTEM_PROMPT, CohereGenerator, CohereReranker, to_answer
from tests.fakes import FakeGenerator, FakeReranker


def chunk(id: str, text: str = "text") -> Chunk:
    return Chunk(id=id, doc_id="doc", title="Directive", section=id.split(":")[1], headings=["6. Requirements"],
                 language="en", status="current", text=text)


# --- Rerank -----------------------------------------------------------------

class FakeRerankClient:
    def __init__(self, results):
        self.results = results
        self.requests = []

    def rerank(self, **kwargs):
        self.requests.append(kwargs)
        return NS(results=[NS(index=i, relevance_score=s) for i, s in self.results])


def test_reranker_maps_results_back_to_chunks_with_scores():
    chunks = [chunk("d:1"), chunk("d:2"), chunk("d:3")]
    client = FakeRerankClient(results=[(2, 0.9), (0, 0.4)])
    ranked = CohereReranker(client, "rerank-v4.0-pro").rerank("q", chunks, top_n=2)
    assert [(c.id, c.score) for c in ranked] == [("d:3", 0.9), ("d:1", 0.4)]
    assert client.requests[0]["top_n"] == 2
    assert client.requests[0]["documents"] == ["text", "text", "text"]


def test_reranker_makes_no_call_for_empty_input():
    client = FakeRerankClient(results=[])
    assert CohereReranker(client, "m").rerank("q", [], top_n=8) == []
    assert client.requests == []


# --- Generation ---------------------------------------------------------------

def chat_response(text, citations):
    return NS(message=NS(content=[NS(type="text", text=text)], citations=citations))


def test_citations_are_normalized_one_per_source():
    response = chat_response(
        "Peer review is required.",
        [NS(start=0, end=11, text="Peer review", sources=[NS(id="d:6.3.7:0"), NS(id="d:Appendix-C:2")])],
    )
    answer = to_answer(response)
    assert answer.text == "Peer review is required."
    assert answer.citations == [
        Citation(chunk_id="d:6.3.7:0", start=0, end=11, text="Peer review"),
        Citation(chunk_id="d:Appendix-C:2", start=0, end=11, text="Peer review"),
    ]


def test_answer_without_citations():
    answer = to_answer(chat_response("I could not find this in the policy documents.", None))
    assert answer.citations == []


class FakeChatClient:
    def __init__(self):
        self.requests = []

    def chat(self, **kwargs):
        self.requests.append(kwargs)
        return chat_response("ok", [])


def test_generator_sends_system_prompt_history_and_documents_with_our_ids():
    client = FakeChatClient()
    history = [Turn(role="user", content="first?"), Turn(role="assistant", content="first answer")]
    CohereGenerator(client, "command-a-03-2025").answer("second?", [chunk("d:6.1.1:0", "Complete an AIA.")], history)

    request = client.requests[0]
    assert [m["role"] for m in request["messages"]] == ["system", "user", "assistant", "user"]
    assert request["messages"][0]["content"] == SYSTEM_PROMPT
    assert request["messages"][-1]["content"] == "second?"
    [document] = request["documents"]
    assert document["id"] == "d:6.1.1:0"
    assert document["data"]["snippet"] == "Complete an AIA."
    assert document["data"]["section"] == "6.1.1"


# --- Rendering ----------------------------------------------------------------

def test_render_numbers_sources_in_order_of_first_citation():
    answer = Answer(
        text="Do an AIA. Get peer review.",
        citations=[
            Citation(chunk_id="b", start=11, end=26, text="Get peer review"),
            Citation(chunk_id="a", start=0, end=9, text="Do an AIA"),
        ],
    )
    text, ids = render(answer)
    assert text == "Do an AIA[1]. Get peer review[2]."
    assert ids == ["a", "b"]


def test_render_groups_sources_on_the_same_span():
    answer = Answer(text="Peer review.", citations=[
        Citation(chunk_id="a", start=0, end=11, text="Peer review"),
        Citation(chunk_id="b", start=0, end=11, text="Peer review"),
    ])
    assert render(answer)[0] == "Peer review[1,2]."


# --- Pipeline -----------------------------------------------------------------

def test_pipeline_with_and_without_rerank(conn):
    dim = get_settings().embed_dimension
    texts = ["notice to clients", "peer review of the system", "security of the data", "training employees"]
    chunks = [chunk(f"d:{i}", text) for i, text in enumerate(texts)]
    # Vector search order will be d:0, d:1, d:2, d:3 (decreasing similarity to the query vector).
    vectors = [[1.0, i * 0.5] + [0.0] * (dim - 2) for i in range(len(chunks))]
    db.replace_document_chunks(conn, "doc", chunks, vectors, get_settings().embed_model)

    class QueryEmbedder:
        def embed(self, texts, kind):
            return [[1.0, 0.0] + [0.0] * (dim - 2) for _ in texts]

    common = dict(history=[], conn=conn, embedder=QueryEmbedder(), reranker=FakeReranker(), generator=FakeGenerator())

    with_rerank = answer_question("peer review", rerank=True, **common)
    assert with_rerank.context[0].id == "d:1"  # FakeReranker promotes the chunk sharing query words
    assert with_rerank.answer.citations[0].chunk_id == "d:1"
    assert [c.id for c in with_rerank.retrieved] == ["d:0", "d:1", "d:2", "d:3"]

    without = answer_question("peer review", rerank=False, **common)
    assert without.context[0].id == "d:0"  # vector search order, untouched
    assert without.rerank_used is False
