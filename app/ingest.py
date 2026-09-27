"""Parse, chunk and store documents in Postgres.

    uv run python -m app.ingest               # every document in the manifest
    uv run python -m app.ingest dadm-en       # just one

Needs the database running (docker compose up -d db) and documents fetched
(python -m app.fetch). Embedding is added in the next PR; for now no API calls.
"""

import argparse

from app import db
from app.chunk import get_chunker
from app.config import get_settings
from app.corpus import get_document, load_manifest, raw_path
from app.domain import Block, Document
from app.parse import parse_tbs_html


def parse_document(doc: Document) -> list[Block]:
    path = raw_path(doc)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m app.fetch` first.")
    if doc.format == "html":
        return parse_tbs_html(path.read_text(encoding="utf-8"))
    raise NotImplementedError(f"{doc.id}: {doc.format} parsing arrives with the full corpus")


def ingest(doc_ids: list[str] | None = None) -> None:
    docs = [get_document(d) for d in doc_ids] if doc_ids else load_manifest()
    chunker = get_chunker()
    print(f"chunker: {get_settings().chunker}")

    with db.connect() as conn:
        db.init_schema(conn)
        for doc in docs:
            blocks = parse_document(doc)
            chunks = chunker.chunk(doc, blocks)
            db.replace_document_chunks(conn, doc.id, chunks)
            print(f"{doc.id}: {len(blocks)} blocks -> {len(chunks)} chunks stored")

        print("\ndoc_id            chunks  embedded")
        for doc_id, total, embedded in db.chunk_counts(conn):
            print(f"{doc_id:16} {total:7} {embedded:9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse, chunk and store documents.")
    parser.add_argument("doc_ids", nargs="*", help="document ids (default: all in the manifest)")
    ingest(parser.parse_args().doc_ids or None)
