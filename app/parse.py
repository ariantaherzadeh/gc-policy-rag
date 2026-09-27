"""Parse a TBS policy page (HTML) into ordered blocks.

    uv run python -m app.parse dadm-en            # summary of the blocks
    uv run python -m app.parse dadm-en --full     # print every block

TBS policy pages share one structure, which this parser relies on:

    #ps-doc
      details.pol-sec                  one per chapter or appendix
        summary > h2#cha6              "6. Requirements"  /  h2#appC "Appendix C - ..."
        div.pol-content
          h3 / h4                      sub-headings ("6.3 Quality assurance", "Peer review")
          ul.pol-cla > li.pol-cla      numbered clauses, which can nest:
            span.pol-cla-id            "6.3.4"
            ul.pol-cla > li.pol-cla    "6.3.4.1"
          p, dl, table, ul             lead-in text, definitions, tables, plain lists

Everything outside #ps-doc (site menus, sidebars, footer) is ignored.
"""

import argparse
import copy
import re

from bs4 import BeautifulSoup, Tag

from app.corpus import get_document, raw_path
from app.domain import Block

SECTION_NUMBER = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
APPENDIX = re.compile(r"^(Appendix [A-Z])\b")


def parse_tbs_html(html: str) -> list[Block]:
    soup = BeautifulSoup(html, "lxml")
    doc = soup.find(id="ps-doc")
    if doc is None:
        raise ValueError("Not a TBS policy page: no #ps-doc element")

    blocks: list[Block] = []
    for chapter in doc.select("details.pol-sec"):
        heading = clean(chapter.find("summary").get_text(" "))
        content = chapter.find("div", class_="pol-content")
        if content is not None:
            _walk(content, [heading], section_of(heading), blocks)
    return blocks


def _walk(
    node: Tag, headings: list[str], section: str | None, out: list[Block]
) -> tuple[list[str], str | None]:
    """Visit `node`'s children in reading order, emitting blocks.

    `headings` is the path of headings above the current position; h3/h4 update it
    for the elements that follow them. The updated state is returned because a
    heading can sit inside a wrapper (e.g. <header><h3>) while the content it
    applies to comes after the wrapper.
    """
    for child in node.find_all(recursive=False):
        name, classes = child.name, child.get("class") or []

        if name == "h3":
            text = clean(child.get_text(" "))
            headings = [headings[0], text]
            section = section_of(text) or section
        elif name == "h4":
            headings = headings[:2] + [clean(child.get_text(" "))]
        elif name == "ul" and "pol-cla" in classes:
            for li in child.find_all("li", class_="pol-cla", recursive=False):
                _clause(li, headings, out)
        elif name == "p":
            out.append(Block(kind="paragraph", section=section, headings=headings, text=text_of(child)))
        elif name == "dl":
            for dt in child.find_all("dt"):
                dd = dt.find_next_sibling("dd")
                definition = text_of(dd) if dd else ""
                out.append(Block(kind="definition", section=section, headings=headings,
                                 text=f"{text_of(dt)}: {definition}"))
        elif name == "table":
            out.extend(_table_rows(child, headings, section))
        elif name in ("ul", "ol"):
            out.append(Block(kind="list", section=section, headings=headings, text=text_of(child)))
        else:
            # Wrapper elements (section, div, header...): look inside.
            headings, section = _walk(child, headings, section, out)
    return headings, section


def _clause(li: Tag, headings: list[str], out: list[Block]) -> None:
    """Emit one numbered clause, then its sub-clauses as separate blocks."""
    number = clean(li.find("span", class_="pol-cla-id").get_text())

    own = copy.copy(li)
    own.find("span", class_="pol-cla-id").decompose()
    for nested in own.find_all("ul", class_="pol-cla"):
        nested.decompose()  # sub-clauses become their own blocks below
    out.append(Block(kind="clause", section=number, headings=headings, text=text_of(own)))

    for nested_list in li.find_all("ul", class_="pol-cla", recursive=False):
        for sub in nested_list.find_all("li", class_="pol-cla", recursive=False):
            _clause(sub, headings, out)


def _table_rows(table: Tag, headings: list[str], section: str | None) -> list[Block]:
    """One block per body row, each cell labelled with its column header.

    A row read on its own ("Level III: ...") still makes sense, which matters
    once it's retrieved without the rest of the table.
    """
    rows = table.find_all("tr")
    if not rows:
        return []
    header = [text_of(cell) for cell in rows[0].find_all(["th", "td"])]

    blocks = []
    for row in rows[1:]:
        cells = [text_of(cell) for cell in row.find_all(["th", "td"])]
        lines = [f"{label}: {value}" if label else value
                 for label, value in zip(header, cells, strict=False) if value]
        if lines:
            blocks.append(Block(kind="table_row", section=section, headings=headings, text="\n".join(lines)))
    return blocks


def section_of(heading: str) -> str | None:
    """'6.3 Quality assurance' -> '6.3', 'Appendix C - ...' -> 'Appendix C'."""
    if match := SECTION_NUMBER.match(heading) or APPENDIX.match(heading):
        return match.group(1)
    return None


def text_of(element: Tag) -> str:
    """Readable text, keeping list items on their own lines."""
    element = copy.copy(element)
    for li in element.find_all("li"):
        li.insert(0, "\n- ")
        li.append("\n")
    return clean(element.get_text())


def clean(text: str) -> str:
    """Collapse runs of spaces within lines and drop blank lines."""
    lines = (re.sub(r"\s+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a downloaded document into blocks.")
    parser.add_argument("doc_id")
    parser.add_argument("--full", action="store_true", help="print every block in full")
    args = parser.parse_args()

    doc = get_document(args.doc_id)
    blocks = parse_tbs_html(raw_path(doc).read_text(encoding="utf-8"))

    for block in blocks:
        where = " > ".join(block.headings)
        if args.full:
            print(f"[{block.kind} {block.section}] {where}\n{block.text}\n")
        else:
            preview = block.text.replace("\n", " ")[:70]
            print(f"{block.kind:10} {block.section or '-':11} {preview}")

    kinds = {k: sum(b.kind == k for b in blocks) for k in dict.fromkeys(b.kind for b in blocks)}
    print(f"\n{len(blocks)} blocks: {kinds}")
