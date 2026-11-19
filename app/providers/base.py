"""The three model-backed stages of the pipeline, as interfaces.

The rest of the app depends on these Protocols, not on Cohere's SDK. Cohere is the
default implementation (added in later PRs); a self-hosted model in an air-gapped
environment, or a fake in tests, can be swapped in without touching retrieval or the API.

A Protocol is structural typing: any class with matching method signatures counts,
no inheritance needed.
"""

from typing import Literal, Protocol, runtime_checkable

from app.domain import Answer, Chunk, Turn


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]:
        """Turn texts into vectors, one per text, in the same order.

        `kind` matters: documents and queries are embedded differently
        (Cohere's `search_document` vs `search_query`) so short questions
        land near the longer passages that answer them.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        """Return the `top_n` most relevant chunks, best first, with `score` set."""
        ...


@runtime_checkable
class Rewriter(Protocol):
    def rewrite(self, question: str, history: list[Turn]) -> str:
        """Turn a follow-up ("does that apply to Level I?") into a standalone search query.

        Separate from Generator so it can use a smaller, cheaper model in production.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    def answer(self, question: str, chunks: list[Chunk], history: list[Turn]) -> Answer:
        """Answer using only `chunks`, citing them; abstain if they don't contain the answer."""
        ...
