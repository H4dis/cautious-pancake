from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import (
    MAX_PDF_MB,
    MAX_CHARS_PER_SOURCE,
    ArzensError,
    ConfigError
)


class IngestionError(ArzensError):
    default_message = "That source could not be ingested."


class PDFParseError(IngestionError):
    default_message = "That PDF could not be read."


class EmptyContentError(IngestionError):
    default_message = "That source contained no readable text."


@dataclass(frozen=True)
class Section:
    label: str
    text: str
    ordinal: int = 0


@dataclass
class Source:
    kind: str
    title: str
    locator: str
    sections: list[Section] = field(default_factory=list)
    fingerprint: str = ""

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)

    @property
    def display_name(self) -> str:
        return self.title or self.locator


_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKLINES_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def truncate_sections(sections: list[Section], max_chars: int) -> tuple[list[Section], bool]:
    kept: list[Section] = []
    running = 0
    for section in sections:
        if running + len(section.text) > max_chars:
            return kept or sections[:1], True
        kept.append(section)
        running += len(section.text)
    return kept, False


def _open_pdf_reader(data: bytes, filename: str) -> PdfReader:
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise PDFParseError(
            f"`{filename}` is not a readable PDF.",
            hint="The file may be corrupt or only partially downloaded.",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise PDFParseError(
            f"`{filename}` could not be opened.", cause=exc
        ) from exc

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise PDFParseError(
                    f"`{filename}` is password-protected.",
                    hint="Remove the password and upload it again.",
                )
        except PDFParseError:
            raise
        except Exception as exc:
            raise PDFParseError(
                f"`{filename}` is encrypted and could not be decrypted.",
                hint="Remove the password and upload it again.",
                cause=exc,
            ) from exc

    return reader


def ingest_pdf(data: bytes, filename: str) -> Source:
    size_mb = len(data) / (1024 * 1024)
    if size_mb > MAX_PDF_MB:
        raise PDFParseError(
            f"`{filename}` is {size_mb:.1f} MB, over the {MAX_PDF_MB} MB limit.",
            hint="Split the document.",
        )
    if not data:
        raise PDFParseError(f"`{filename}` is empty.")

    reader = _open_pdf_reader(data, filename)

    if len(reader.pages) == 0:
        raise EmptyContentError(f"`{filename}` has no pages.")

    sections: list[Section] = []
    empty_pages = 0

    for index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            empty_pages += 1
            continue

        text = normalise_text(raw)
        if not text:
            empty_pages += 1
            continue

        sections.append(Section(label=f"page {index}", text=text, ordinal=index))

    if not sections:
        raise EmptyContentError(
            f"`{filename}` contains no extractable text.",
            hint="This is almost certainly a scanned document.",
        )

    sections, truncated = truncate_sections(sections, MAX_CHARS_PER_SOURCE)
    if truncated:
        sections.append(
            Section(
                label="truncated",
                text=f"[This document exceeded the {MAX_CHARS_PER_SOURCE:,}-character limit]",
                ordinal=len(sections) + 1,
            )
        )

    import hashlib
    fingerprint = hashlib.sha256(data).hexdigest()[:32]

    return Source(
        kind="pdf",
        title=filename,
        locator=filename,
        sections=sections,
        fingerprint=fingerprint,
    )