"""The question-answering pipeline: retrieve → rerank → generate.

    uv run python -m app.chat "What must happen before an automated decision system goes live?"
    uv run python -m app.chat "..." --no-rerank      # skip rerank: top 8 straight from vector search
    uv run python -m app.chat "..." --show-context   # also list the passages sent to Command

Cost per question: 1 embed call (0 if the question is cached) + 1 rerank call
(0 with --no-rerank) + 1 chat call.
"""

import argparse
from dataclasses import dataclass

import psycopg

from app import db
from app.config import get_settings
from app.domain import Answer, Chunk, Turn
from app.providers.base import Embedder, Generator, Reranker
from app.retrieve import Filters, retrieve


@dataclass
class ChatResult:
    answer: Answer
    context: list[Chunk]  # the passages Command saw (after rerank), best first
    retrieved: list[Chunk]  # everything vector search returned, for debugging and evals
    rerank_used: bool = True


def answer_question(
    question: str,
    history: list[Turn],
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker,
    generator: Generator,
    rerank: bool | None = None,
    filters: Filters = Filters(),
) -> ChatResult:
    settings = get_settings()
    rerank = settings.rerank_enabled if rerank is None else rerank

    retrieved = retrieve(question, embedder, conn, settings.retrieve_top_k, filters)

    if rerank:
        context = reranker.rerank(question, retrieved, settings.rerank_top_n)
    else:
        # The ablation baseline: trust vector search's own order.
        context = retrieved[: settings.rerank_top_n]

    answer = generator.answer(question, context, history)
    return ChatResult(answer=answer, context=context, retrieved=retrieved, rerank_used=rerank)


def render(answer: Answer) -> tuple[str, list[str]]:
    """Answer text with [n] markers after each cited span, plus the chunk ids in order of n."""
    numbers: dict[str, int] = {}
    for citation in sorted(answer.citations, key=lambda c: c.start):
        numbers.setdefault(citation.chunk_id, len(numbers) + 1)

    markers: dict[int, set[int]] = {}
    for citation in answer.citations:
        markers.setdefault(citation.end, set()).add(numbers[citation.chunk_id])

    text = answer.text
    for end in sorted(markers, reverse=True):  # insert from the back so earlier offsets stay valid
        label = "[" + ",".join(str(n) for n in sorted(markers[end])) + "]"
        text = text[:end] + label + text[end:]
    return text, list(numbers)


if __name__ == "__main__":
    from app.providers import get_embedder, get_generator, get_reranker

    parser = argparse.ArgumentParser(description="Ask a question, get a cited answer.")
    parser.add_argument("question")
    parser.add_argument("--no-rerank", action="store_true", help="skip rerank (ablation baseline)")
    parser.add_argument("--show-context", action="store_true", help="list the passages sent to Command")
    args = parser.parse_args()

    embedder, reranker, generator = get_embedder(), get_reranker(), get_generator()
    with db.connect() as conn:
        result = answer_question(
            args.question, [], conn, embedder, reranker, generator,
            rerank=False if args.no_rerank else None,
        )

    text, cited_ids = render(result.answer)
    by_id = {c.id: c for c in result.context}
    print(text, "\n")
    for n, chunk_id in enumerate(cited_ids, 1):
        chunk = by_id[chunk_id]
        print(f"[{n}] {chunk.title}, section {chunk.section}  ({chunk_id})")

    if args.show_context:
        label = "rerank score" if result.rerank_used else "vector score"
        print(f"\nPassages sent to Command ({label}):")
        for chunk in result.context:
            print(f"  {chunk.score:.3f}  {chunk.id:26} {chunk.text.replace(chr(10), ' ')[:60]}")

    calls = 1 if embedder.misses else 0
    print(f"\nAPI calls: embed {calls}, rerank {reranker.calls}, chat {generator.calls} "
          f"= {calls + reranker.calls + generator.calls}")
