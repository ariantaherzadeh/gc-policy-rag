from pathlib import Path

import pytest

from app.corpus import load_manifest
from app.parse import parse_tbs_html, section_of

SAMPLE = (Path(__file__).parent / "fixtures" / "tbs_sample.html").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def blocks():
    return parse_tbs_html(SAMPLE)


def by_section(blocks, section):
    return [b for b in blocks if b.section == section]


def test_blocks_come_out_in_reading_order(blocks):
    assert [b.section for b in blocks] == ["1.1", "1.2", "1.2.1", "6", "6.3.7", "Appendix A", "Appendix C"]


def test_page_chrome_is_ignored(blocks):
    all_text = " ".join(b.text for b in blocks)
    assert "Site menu" not in all_text
    assert "Footer" not in all_text


def test_clause_number_is_stripped_from_text(blocks):
    assert by_section(blocks, "1.1")[0].text == "This sample takes effect on April 1, 2030."


def test_nested_clause_becomes_its_own_block(blocks):
    parent = by_section(blocks, "1.2")[0]
    child = by_section(blocks, "1.2.1")[0]
    assert "one year" not in parent.text
    assert child.text == "Existing systems have one year to comply."


def test_heading_path_follows_h3_inside_a_wrapper_and_h4(blocks):
    clause = by_section(blocks, "6.3.7")[0]
    assert clause.headings == ["6. Requirements", "6.3 Quality assurance", "Peer review"]


def test_plain_lists_inside_a_clause_keep_one_item_per_line(blocks):
    assert by_section(blocks, "6.3.7")[0].text == "Consulting qualified experts:\n- Faculty members\n- Researchers"


def test_lead_in_paragraph_is_kept(blocks):
    paragraph = by_section(blocks, "6")[0]
    assert paragraph.kind == "paragraph"
    assert paragraph.text == "The responsible official is accountable for:"


def test_definition(blocks):
    definition = by_section(blocks, "Appendix A")[0]
    assert definition.kind == "definition"
    assert definition.text == "production: When a system is in use."


def test_table_row_labels_each_cell_with_its_column(blocks):
    row = by_section(blocks, "Appendix C")[0]
    assert row.kind == "table_row"
    assert row.text == "Requirement: Peer review\nLevel I: None\nLevel II: Consult one expert"


def test_non_policy_page_is_rejected():
    with pytest.raises(ValueError):
        parse_tbs_html("<html><body><p>Not a policy</p></body></html>")


@pytest.mark.parametrize(
    ("heading", "expected"),
    [
        ("6. Requirements", "6"),
        ("6.3 Quality assurance", "6.3"),
        ("Appendix C - Impact Level Requirements", "Appendix C"),
        ("Peer review", None),
    ],
)
def test_section_of(heading, expected):
    assert section_of(heading) == expected


def test_manifest_loads():
    docs = load_manifest()
    assert docs, "manifest should list at least one document"
    assert len({d.id for d in docs}) == len(docs), "document ids must be unique"
