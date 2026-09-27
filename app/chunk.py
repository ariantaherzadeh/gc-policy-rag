"""Chunking: turning a document's parsed blocks into the passages that get embedded.

This is the owner's judgment call (see docs/spec.md). This module provides the
plumbing: a Chunker interface, a registry so the strategy is picked by name in config
(CHUNKER=block), and one deliberately simple default. Real strategies get added here.

    uv run python -m app.chunk dadm-en          # preview chunks with the configured chunker
"""

import argparse
from collections import Counter
from typing import Protocol

from app.config import get_settings
from app.corpus import get_document, raw_path
from app.domain import Block, Chunk, Document


class Chunker(Protocol):
    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]: ...


class BlockChunker:
    """Default: one chunk per parsed block, text unchanged.

    Deliberately naive, so there's a baseline to measure real strategies against.
    Known weaknesses, all left for the owner to decide on:

    - TODO(owner): tiny blocks (e.g. "6.2.2 Providing notices prominently and in plain
      language.") may be too short to retrieve well. Merge neighbours?
    - TODO(owner): clauses in section 6 have no subject without the lead-in paragraph
      ("The assistant deputy minister ... is responsible for:"). Carry it into each chunk?
    - TODO(owner): should the heading path or document title be part of the embedded text?
    - TODO(owner): long blocks (big table rows) are never split. Is there a size limit?
    """

    def chunk(self, doc: Document, blocks: list[Block]) -> list[Chunk]:
        return [
            Chunk(
                id=chunk_id,
                doc_id=doc.id,
                title=doc.title,
                section=block.section,
                headings=block.headings,
                language=doc.language,
                status=doc.status,
                access_level=doc.access_level,
                text=block.text,
            )
            for chunk_id, block in zip(chunk_ids(doc, blocks), blocks, strict=True)
        ]


def chunk_ids(doc: Document, blocks: list[Block]) -> list[str]:
    """Stable, readable ids: '<doc>:<section>:<n>', where n counts repeats of a section.

    e.g. 'dadm-en:6.3.7:0', 'dadm-en:Appendix A:3'. The same input always produces the
    same ids, so re-ingesting updates rows instead of duplicating them, and eval runs
    can refer to chunks by id.
    """
    seen: Counter[str] = Counter()
    ids = []
    for block in blocks:
        section = block.section or "none"
        ids.append(f"{doc.id}:{section}:{seen[section]}")
        seen[section] += 1
    return ids


# Add new strategies here; select one with CHUNKER=<name> in .env.
CHUNKERS: dict[str, type[Chunker]] = {
    "block": BlockChunker,
}


def get_chunker(name: str | None = None) -> Chunker:
    name = name or get_settings().chunker
    try:
        return CHUNKERS[name]()
    except KeyError:
        raise ValueError(f"Unknown chunker {name!r}. Options: {', '.join(CHUNKERS)}") from None


if __name__ == "__main__":
    from app.parse import parse_tbs_html

    parser = argparse.ArgumentParser(description="Preview the chunks for one document.")
    parser.add_argument("doc_id")
    parser.add_argument("--chunker", help="override the configured chunker")
    args = parser.parse_args()

    doc = get_document(args.doc_id)
    chunks = get_chunker(args.chunker).chunk(doc, parse_tbs_html(raw_path(doc).read_text(encoding="utf-8")))

    for c in chunks:
        print(f"{c.id:28} {len(c.text):5} chars  {c.text.replace(chr(10), ' ')[:60]}")

    lengths = sorted(len(c.text) for c in chunks)
    print(f"\n{len(chunks)} chunks, length in chars: min {lengths[0]}, "
          f"median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
