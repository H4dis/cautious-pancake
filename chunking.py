from __future__ import annotations

import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion import Source

_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def content_fingerprint(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()[:32]


def build_splitter(
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def sanitise_metadata(metadata: dict) -> dict:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def format_citation(title: str, section_label: str) -> str:
    title = (title or "source").strip()
    section_label = (section_label or "").strip()
    if not section_label or section_label == title:
        return title
    return f"{title} — {section_label}"


def chunk_source(source: Source, splitter: RecursiveCharacterTextSplitter | None = None) -> list[Document]:
    splitter = splitter or build_splitter()
    documents: list[Document] = []

    for section in source.sections:
        pieces = [p for p in splitter.split_text(section.text) if p.strip()]
        for position, piece in enumerate(pieces):
            metadata = sanitise_metadata(
                {
                    "source_id": source.fingerprint,
                    "kind": source.kind,
                    "title": source.display_name,
                    "locator": source.locator,
                    "section": section.label,
                    "section_ordinal": section.ordinal,
                    "chunk_index": position,
                    "chunk_total": len(pieces),
                    "chunk_id": content_fingerprint(
                        f"{source.fingerprint}|{section.ordinal}|{position}|{piece}"
                    ),
                    "citation": format_citation(source.display_name, section.label),
                }
            )
            documents.append(Document(page_content=piece, metadata=metadata))

    return documents


def chunk_ids(documents: list[Document]) -> list[str]:
    return [str(doc.metadata["chunk_id"]) for doc in documents]