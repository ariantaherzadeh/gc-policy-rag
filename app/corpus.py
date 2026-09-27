"""Reading the corpus manifest and locating downloaded files."""

from pathlib import Path

import yaml

from app.domain import Document

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "corpus" / "manifest.yaml"
RAW_DIR = ROOT / "data" / "raw"


def load_manifest(path: Path = MANIFEST_PATH) -> list[Document]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Document(**entry) for entry in data["documents"]]


def get_document(doc_id: str) -> Document:
    for doc in load_manifest():
        if doc.id == doc_id:
            return doc
    raise KeyError(f"No document with id {doc_id!r} in {MANIFEST_PATH.name}")


def raw_path(doc: Document) -> Path:
    """Where a document's downloaded file lives, e.g. data/raw/dadm-en.html."""
    return RAW_DIR / f"{doc.id}.{doc.format}"
