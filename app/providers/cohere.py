"""Cohere implementations of the provider interfaces.

API parameters checked against docs.cohere.com (reference/embed, reference/rerank,
docs/rag-citations) and the installed SDK (cohere 7.x, `ClientV2`). The SDK itself
retries rate-limited requests (HTTP 429) twice, waiting as long as the server's
Retry-After header asks.
"""

from typing import Literal

import cohere

from app.config import Settings
from app.domain import Answer, Chunk, Citation, Turn

# From the Embed API reference: "Maximum number of texts per call is 96."
EMBED_BATCH_SIZE = 96

INPUT_TYPES = {"doc": "search_document", "query": "search_query"}


def make_client(settings: Settings) -> cohere.ClientV2:
    if settings.cohere_api_key is None:
        raise RuntimeError("COHERE_API_KEY is not set. Add it to .env.")
    return cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())


class CohereEmbedder:
    def __init__(self, client: cohere.ClientV2, model: str, dimension: int):
        self.client = client
        self.model = model
        self.dimension = dimension
        self.calls = 0  # API calls made, for tracking the monthly budget

    def embed(self, texts: list[str], kind: Literal["doc", "query"]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            response = self.client.embed(
                model=self.model,
                texts=texts[start : start + EMBED_BATCH_SIZE],
                input_type=INPUT_TYPES[kind],
                embedding_types=["float"],
                output_dimension=self.dimension,
            )
            self.calls += 1
            vectors.extend(response.embeddings.float_)
        return vectors


class CohereReranker:
    def __init__(self, client: cohere.ClientV2, model: str):
        self.client = client
        self.model = model
        self.calls = 0

    def rerank(self, query: str, chunks: list[Chunk], top_n: int) -> list[Chunk]:
        if not chunks:
            return []
        response = self.client.rerank(
            model=self.model,
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=top_n,
        )
        self.calls += 1
        # Results come back best first; `index` points into the list we sent.
        return [
            chunks[result.index].model_copy(update={"score": result.relevance_score})
            for result in response.results
        ]


SYSTEM_PROMPT = """\
You answer questions about Government of Canada Treasury Board policy documents.
Use only the provided documents, and cite them. Do not use outside knowledge.
If the documents do not contain the answer, say that you could not find it in the
policy documents, instead of guessing.
Answer in the same language as the question."""


class CohereGenerator:
    def __init__(self, client: cohere.ClientV2, model: str):
        self.client = client
        self.model = model
        self.calls = 0

    def answer(self, question: str, chunks: list[Chunk], history: list[Turn]) -> Answer:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [{"role": turn.role, "content": turn.content} for turn in history]
        messages.append({"role": "user", "content": question})

        response = self.client.chat(
            model=self.model,
            messages=messages,
            # Each passage carries our chunk id, so citations point straight back to it.
            documents=[
                {
                    "id": chunk.id,
                    "data": {
                        "title": chunk.title,
                        "section": chunk.section or "",
                        "headings": " > ".join(chunk.headings),
                        "snippet": chunk.text,
                    },
                }
                for chunk in chunks
            ],
            temperature=0,  # as repeatable as possible, so eval runs are comparable
        )
        self.calls += 1
        return to_answer(response)


def to_answer(response) -> Answer:
    """Convert Cohere's chat response into our provider-neutral Answer.

    Cohere's citation is a span of the answer plus the source documents behind it.
    We emit one Citation per (span, source) pair, so a span backed by two chunks
    becomes two Citations with the same start/end.
    """
    message = response.message
    text = "".join(item.text for item in message.content or [] if getattr(item, "text", None))
    citations = [
        Citation(chunk_id=source.id, start=c.start, end=c.end, text=c.text)
        for c in message.citations or []
        for source in c.sources or []
        if getattr(source, "id", None) and c.start is not None and c.end is not None
    ]
    return Answer(text=text, citations=citations)
