"""Parse, chunk, embed and store documents in Postgres.

    uv run python -m app.ingest --estimate    # show how many API calls it would take, call nothing
    uv run python -m app.ingest               # every document in the manifest
    uv run python -m app.ingest dadm-en       # just one
    uv run python -m app.ingest --no-embed    # store chunks without embeddings (no API calls)
    uv run python -m app.ingest --reindex     # drop everything and rebuild (e.g. after a model change)

Needs the database running (docker compose up -d db) and documents fetched
(python -m app.fetch). Embeddings are cached on disk, so unchanged text costs nothing.
"""

import argparse
import math

from app import db
from app.chunk import get_chunker
from app.config import get_settings
from app.corpus import get_document, load_manifest, raw_path
from app.domain import Block, Chunk, Document
from app.parse import parse_tbs_html
from app.providers import get_embedder
from app.providers.cohere import EMBED_BATCH_SIZE


def parse_document(doc: Document) -> list[Block]:
    path = raw_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m app.fetch` first.")
    if doc.format == "html":
        return parse_tbs_html(path.read_text(encoding="utf-8"))
    raise NotImplementedError(f"{doc.id}: {doc.format} parsing arrives with the full corpus")


def ingest(doc_ids: list[str] | None, embed: bool, estimate_only: bool, reindex: bool) -> None:
    settings = get_settings()
    if reindex and doc_ids:
        raise SystemExit("--reindex rebuilds the whole index; run it without document ids.")
    docs = [get_document(d) for d in doc_ids] if doc_ids else load_manifest()
    chunker = get_chunker()

    chunks_by_doc: dict[str, list[Chunk]] = {doc.id: chunker.chunk(doc, parse_document(doc)) for doc in docs}
    all_chunks = [c for chunks in chunks_by_doc.values() for c in chunks]
    print(f"chunker: {settings.chunker}, {len(all_chunks)} chunks from {len(docs)} document(s)")

    embedder = get_embedder() if embed else None
    if embedder is not None:
        texts = [c.text for c in all_chunks]
        new = embedder.uncached(texts, "doc")
        calls = math.ceil(len(new) / EMBED_BATCH_SIZE)
        print(f"embed:   {settings.embed_model} ({settings.embed_dimension} dims), "
              f"{len(texts) - len(new)} cached, {len(new)} new -> {calls} API call(s)")
    if estimate_only:
        return

    with db.connect() as conn:
        if reindex:
            db.drop_chunks_table(conn)
        db.init_schema(conn)

        if embedder is not None:
            if others := db.other_embed_models(conn, settings.embed_model):
                raise SystemExit(
                    f"The index holds vectors from {others}, not {settings.embed_model}. "
                    "Vectors from different models can't be compared: run with --reindex."
                )
            vectors = embedder.embed([c.text for c in all_chunks], "doc")
            print(f"embedded: {embedder.inner.calls} API call(s) made")

        position = 0
        for doc in docs:
            chunks = chunks_by_doc[doc.id]
            doc_vectors = vectors[position : position + len(chunks)] if embedder else None
            position += len(chunks)
            db.replace_document_chunks(conn, doc.id, chunks, doc_vectors, settings.embed_model)

        print("\ndoc_id            chunks  embedded")
        for doc_id, total, embedded in db.chunk_counts(conn):
            print(f"{doc_id:16} {total:7} {embedded:9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse, chunk, embed and store documents.")
    parser.add_argument("doc_ids", nargs="*", help="document ids (default: all in the manifest)")
    parser.add_argument("--estimate", action="store_true", help="print the API cost and stop")
    parser.add_argument("--no-embed", action="store_true", help="store chunks without embeddings")
    parser.add_argument("--reindex", action="store_true", help="drop the table and rebuild it")
    args = parser.parse_args()
    ingest(args.doc_ids or None, embed=not args.no_embed, estimate_only=args.estimate, reindex=args.reindex)
