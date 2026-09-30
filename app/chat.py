"""The question-answering pipeline: rewrite → retrieve → rerank → generate.

    uv run python -m app.chat "What must happen before an automated decision system goes live?"
    uv run python -m app.chat "..." --no-rerank      # skip rerank: top 8 straight from vector search
    uv run python -m app.chat "..." --show-context   # also list the passages sent to Command
    uv run python -m app.chat                        # interactive: a conversation with follow-ups

Cost per question: 1 rewrite call (follow-ups only) + 1 embed call (0 if cached)
+ 1 rerank call (0 with --no-rerank) + 1 chat call (0 if it abstains early).
"""

import argparse
from dataclasses import dataclass

import psycopg

from app import db
from app.config import get_settings
from app.domain import Answer, Chunk, Turn
from app.providers.base import Embedder, Generator, Reranker, Rewriter
from app.retrieve import Filters, retrieve

NOT_FOUND = "I could not find this in the policy documents."


@dataclass
class ChatResult:
    answer: Answer
    search_query: str  # what was actually searched: the question, or its rewrite
    context: list[Chunk]  # the passages Command saw (after rerank), best first
    retrieved: list[Chunk]  # everything vector search returned, for debugging and evals
    rerank_used: bool
    abstained: bool
    abstain_reason: str | None = None


def answer_question(
    question: str,
    history: list[Turn],
    conn: psycopg.Connection,
    embedder: Embedder,
    reranker: Reranker,
    generator: Generator,
    rewriter: Rewriter,
    rerank: bool | None = None,
    filters: Filters = Filters(),
) -> ChatResult:
    settings = get_settings()
    rerank = settings.rerank_enabled if rerank is None else rerank

    # 1. Rewrite follow-ups into a standalone query. The first question has nothing to resolve.
    search_query = rewriter.rewrite(question, history) if history else question

    # 2. Retrieve, and 3. rerank, using the standalone query.
    retrieved = retrieve(search_query, embedder, conn, settings.retrieve_top_k, filters)
    if rerank:
        context = reranker.rerank(search_query, retrieved, settings.rerank_top_n)
    else:
        context = retrieved[: settings.rerank_top_n]  # the ablation baseline: vector search's own order

    def result(answer: Answer, abstained: bool, reason: str | None = None) -> ChatResult:
        return ChatResult(answer, search_query, context, retrieved, rerank, abstained, reason)

    # Early abstention: skip the Chat call when there's clearly nothing to answer from.
    if not context:
        return result(Answer(text=NOT_FOUND), abstained=True, reason="no passages retrieved")
    threshold = settings.min_rerank_score
    if rerank and threshold is not None and (context[0].score or 0.0) < threshold:
        return result(Answer(text=NOT_FOUND), abstained=True,
                      reason=f"best rerank score {context[0].score:.3f} < {threshold}")

    # 4. Generate. Command sees the original question and the conversation, not the rewrite.
    answer = generator.answer(question, context, history)
    if is_abstention(answer):
        return result(answer, abstained=True, reason="answer cites no passages")
    return result(answer, abstained=False)


def is_abstention(answer: Answer) -> bool:
    """An answer that cites nothing is treated as "I don't know".

    TODO(owner): a heuristic. It misses an answer that cites a passage while saying the
    answer isn't there, and flags a genuine answer the model forgot to cite. Eval runs
    will show how often either happens.
    """
    return not answer.citations


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


def print_result(question: str, result: ChatResult, show_context: bool) -> None:
    if result.search_query != question:
        print(f"(searched for: {result.search_query})\n")

    text, cited_ids = render(result.answer)
    by_id = {c.id: c for c in result.context}
    print(text, "\n")
    for n, chunk_id in enumerate(cited_ids, 1):
        chunk = by_id[chunk_id]
        print(f"[{n}] {chunk.title}, section {chunk.section}  ({chunk_id})")
    if result.abstained:
        print(f"(abstained: {result.abstain_reason})")

    if show_context:
        label = "rerank score" if result.rerank_used else "vector score"
        print(f"\nPassages ({label}):")
        for chunk in result.context:
            print(f"  {chunk.score:.3f}  {chunk.id:26} {chunk.text.replace(chr(10), ' ')[:60]}")


if __name__ == "__main__":
    from app.providers import get_embedder, get_generator, get_reranker, get_rewriter

    parser = argparse.ArgumentParser(description="Ask a question, get a cited answer.")
    parser.add_argument("question", nargs="?", help="omit for an interactive conversation")
    parser.add_argument("--no-rerank", action="store_true", help="skip rerank (ablation baseline)")
    parser.add_argument("--show-context", action="store_true", help="list the passages sent to Command")
    args = parser.parse_args()

    embedder, reranker, generator, rewriter = get_embedder(), get_reranker(), get_generator(), get_rewriter()
    history: list[Turn] = []

    def ask(question: str) -> None:
        result = answer_question(question, history, conn, embedder, reranker, generator, rewriter,
                                 rerank=False if args.no_rerank else None)
        print_result(question, result, args.show_context)
        history.extend([Turn(role="user", content=question), Turn(role="assistant", content=result.answer.text)])
        calls = {"rewrite": rewriter.calls, "embed": embedder.inner.calls,
                 "rerank": reranker.calls, "chat": generator.calls}
        print(f"\nAPI calls so far: {', '.join(f'{k} {v}' for k, v in calls.items())} = {sum(calls.values())}")

    with db.connect() as conn:
        if args.question:
            ask(args.question)
        else:
            print("Ask about the policy documents. Empty line to quit.")
            while question := input("\n> ").strip():
                ask(question)
