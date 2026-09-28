"""Vector search: find the chunks whose meaning is closest to a question.

    uv run python -m app.retrieve "What must happen before an automated decision system goes live?"
    uv run python -m app.retrieve "..." --lang fr --include-superseded --top-k 10

The question is embedded (1 API call, or 0 if that exact question is cached), then
Postgres ranks every chunk by cosine distance to it, after applying metadata filters.
"""

import argparse
from dataclasses import dataclass

import psycopg
from pgvector import Vector

from app import db
from app.config import get_settings
from app.domain import Chunk, Language, Status
from app.providers.base import Embedder


@dataclass(frozen=True)
class Filters:
    """Metadata filters, applied in SQL before ranking.

    The default only returns current documents: a superseded policy can't be
    retrieved, so it can't be cited.
    """

    statuses: tuple[Status, ...] = ("current",)
    language: Language | None = None  # None = any language
    access_levels: tuple[str, ...] | None = None  # None = no access filtering


def search(
    conn: psycopg.Connection,
    query_vector: list[float],
    embed_model: str,
    top_k: int,
    filters: Filters = Filters(),
) -> list[Chunk]:
    """Return the `top_k` chunks closest to `query_vector`, best first, with `score` set.

    `<=>` is pgvector's cosine distance (0 = same direction, 2 = opposite), so
    `1 - distance` is cosine similarity: higher is more similar.
    """
    conditions = ["embedding IS NOT NULL", "embed_model = %(model)s", "status = ANY(%(statuses)s)"]
    params: dict = {
        "vector": Vector(query_vector),
        "model": embed_model,
        "statuses": list(filters.statuses),
        "top_k": top_k,
    }
    if filters.language is not None:
        conditions.append("language = %(language)s")
        params["language"] = filters.language
    if filters.access_levels is not None:
        # Chunks without an access level are treated as open to everyone.
        conditions.append("(access_level IS NULL OR access_level = ANY(%(access)s))")
        params["access"] = list(filters.access_levels)

    rows = conn.execute(
        f"""
        SELECT id, doc_id, title, section, headings, language, status, access_level, text,
               1 - (embedding <=> %(vector)s) AS score
        FROM chunks
        WHERE {" AND ".join(conditions)}
        ORDER BY embedding <=> %(vector)s
        LIMIT %(top_k)s
        """,
        params,
    ).fetchall()

    columns = ["id", "doc_id", "title", "section", "headings", "language", "status", "access_level", "text", "score"]
    return [Chunk(**dict(zip(columns, row, strict=True))) for row in rows]


def retrieve(
    question: str,
    embedder: Embedder,
    conn: psycopg.Connection,
    top_k: int | None = None,
    filters: Filters = Filters(),
) -> list[Chunk]:
    """Embed the question and run the vector search."""
    settings = get_settings()
    [query_vector] = embedder.embed([question], "query")
    return search(conn, query_vector, settings.embed_model, top_k or settings.retrieve_top_k, filters)


if __name__ == "__main__":
    from app.providers import get_embedder

    parser = argparse.ArgumentParser(description="Vector search over the stored chunks.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=10, help="results to show (default 10)")
    parser.add_argument("--lang", choices=["en", "fr"], help="only chunks in this language")
    parser.add_argument("--include-superseded", action="store_true", help="also search superseded documents")
    args = parser.parse_args()

    filters = Filters(
        statuses=("current", "superseded") if args.include_superseded else ("current",),
        language=args.lang,
    )
    embedder = get_embedder()
    with db.connect() as conn:
        results = retrieve(args.question, embedder, conn, args.top_k, filters)

    for rank, chunk in enumerate(results, 1):
        print(f"{rank:2}. {chunk.score:.3f}  {chunk.id:26} {chunk.text.replace(chr(10), ' ')[:70]}")
    print(f"\nquery embedding: {'1 API call' if embedder.misses else 'cached (0 API calls)'}")
