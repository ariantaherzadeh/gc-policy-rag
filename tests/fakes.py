"""Fake providers for tests: deterministic, offline, and free (no API calls)."""

from typing import Literal

from app.domain import Answer, Chunk, Citation, Turn


class FakeEmbedder:
    """Embeds text as letter counts. Crude, but texts sharing words get similar vectors."""

    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        return [[float(t.lower().count(c)) for c in alphabet] for t in texts]


class FakeReranker:
    """Scores chunks by how many query words they contain."""

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        words = set(query.lower().split())
        scored = [
            c.model_copy(update={"score": float(len(words & set(c.text.lower().split())))})
            for c in chunks
        ]
        return sorted(scored, key=lambda c: c.score or 0.0, reverse=True)[:top_n]


class FakeGenerator:
    """Answers with the first chunk's text and cites all of it; abstains with no chunks."""

    def answer(self, question: str, chunks: list[Chunk], history: list[Turn]) -> Answer:
        if not chunks:
            return Answer(text="I don't know.")
        text = chunks[0].text
        return Answer(
            text=text,
            citations=[Citation(chunk_id=chunks[0].id, start=0, end=len(text), text=text)],
        )


class FakeRewriter:
    """Appends the last user turn's text, so tests can see the rewrite was used."""

    def __init__(self):
        self.calls = 0

    def rewrite(self, question: str, history: list[Turn]) -> str:
        self.calls += 1
        last_user = next(t.content for t in reversed(history) if t.role == "user")
        return f"{question} (about: {last_user})"
