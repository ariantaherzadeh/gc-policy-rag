"""The data that flows through the pipeline.

These types are ours, not any provider's. Cohere's SDK returns its own objects;
each provider converts them into these, so the rest of the app (retrieval, API, UI,
evals) never depends on one vendor's format.
"""

from typing import Literal

from pydantic import BaseModel

Language = Literal["en", "fr"]
Status = Literal["current", "superseded"]


class Document(BaseModel):
    """One entry in corpus/manifest.yaml: a source document and its metadata."""

    id: str
    title: str
    url: str
    language: Language
    published: str | None = None  # ISO date, e.g. "2025-06-24"
    status: Status
    format: Literal["html", "pdf"]
    access_level: str | None = None


class Block(BaseModel):
    """One structural unit of a parsed document, in reading order.

    Parsing produces blocks; chunking (a separate step) decides how to group
    or split them into the chunks that get embedded.
    """

    kind: Literal["clause", "paragraph", "definition", "table_row", "list"]
    section: str | None = None  # clause number ("6.1.1") or appendix ("Appendix C")
    headings: list[str] = []  # heading path, e.g. ["6. Requirements", "6.3 Quality assurance", "Peer review"]
    text: str


class Chunk(BaseModel):
    """One retrievable passage, plus the metadata needed to filter and cite it."""

    id: str  # stable across re-ingests, e.g. "dadm-en:6.1.1:0"
    doc_id: str  # matches `id` in corpus/manifest.yaml
    title: str  # document title, shown next to citations
    section: str | None = None  # e.g. "6.1.1"; evals match expected sources on this
    headings: list[str] = []  # heading path, shown with citations
    language: Language
    status: Status
    access_level: str | None = None
    text: str

    # Set by retrieval (cosine similarity) and then by rerank (relevance score).
    # None means the chunk hasn't been scored yet.
    score: float | None = None


class Citation(BaseModel):
    """A span of the answer text backed by a specific chunk.

    `start`/`end` are character offsets into `Answer.text`; `text` is the cited span.
    Normalized from the provider's format so the UI only ever sees this shape.
    """

    chunk_id: str
    start: int
    end: int
    text: str


class Answer(BaseModel):
    text: str
    citations: list[Citation] = []


class Turn(BaseModel):
    """One message in the conversation history."""

    role: Literal["user", "assistant"]
    content: str
