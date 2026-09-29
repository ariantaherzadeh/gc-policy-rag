import pytest

from app.chunk import BlockChunker, chunk_ids, get_chunker
from app.domain import Block, Document

DOC = Document(
    id="sample-en", title="Sample Directive", url="https://example.org", language="en",
    status="current", format="html",
)
BLOCKS = [
    Block(kind="clause", section="6.3.7", headings=["6. Requirements", "6.3 Quality assurance"], text="Peer review."),
    Block(kind="definition", section="Appendix A", headings=["Appendix A"], text="production: In use."),
    Block(kind="definition", section="Appendix A", headings=["Appendix A"], text="test environment: Not in use."),
]


def test_block_chunker_makes_one_chunk_per_block_with_document_metadata():
    chunks = BlockChunker().chunk(DOC, BLOCKS)
    assert [c.text for c in chunks] == [b.text for b in BLOCKS]
    first = chunks[0]
    assert (first.doc_id, first.title, first.language, first.status) == ("sample-en", "Sample Directive", "en", "current")
    assert first.section == "6.3.7"
    assert first.headings == ["6. Requirements", "6.3 Quality assurance"]


def test_chunk_ids_are_unique_and_count_repeated_sections():
    assert chunk_ids(DOC, BLOCKS) == ["sample-en:6.3.7:0", "sample-en:Appendix-A:0", "sample-en:Appendix-A:1"]


def test_chunk_ids_are_stable_across_runs():
    assert chunk_ids(DOC, BLOCKS) == chunk_ids(DOC, list(BLOCKS))


def test_unknown_chunker_name_is_a_clear_error():
    with pytest.raises(ValueError, match="Unknown chunker"):
        get_chunker("does-not-exist")


def test_chunk_ids_contain_no_whitespace():
    # Cohere's Chat API rejects document ids with whitespace ("Appendix A" sections).
    assert not any(" " in chunk_id for chunk_id in chunk_ids(DOC, BLOCKS))
