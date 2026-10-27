from __future__ import annotations

import re
from urllib.parse import urlparse
import requests

from config import MAX_URLS_PER_INGEST, MAX_CHARS_PER_SOURCE, WEB_TIMEOUT_SECONDS, ArzensError
from ingestion import Source, Section, normalise_text, truncate_sections, ingest_pdf
from chunking import build_splitter, chunk_source
from vector_store import add_documents, Manifest, SourceRecord
import time


class WebFetchError(ArzensError):
    default_message = "That website could not be read."


JINA_READER_PREFIX = "https://r.jina.ai/"
_USER_AGENT = "Mozilla/5.0 (compatible; ArzensKnowledgeAssistant/1.0)"

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript|svg|template)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_CHROME_RE = re.compile(r"<(nav|header|footer|aside|form)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_END_RE = re.compile(r"</(p|div|section|article|li|tr|h[1-6]|blockquote|pre)\s*>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_H1_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", re.IGNORECASE | re.DOTALL)

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$", re.MULTILINE)


def normalise_url(raw: str) -> str:
    url = raw.strip().strip("<>\"'")
    if not url:
        raise WebFetchError("An empty URL was provided.")
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url
    parsed = urlparse(url)
    if not parsed.netloc or "." not in parsed.netloc:
        raise WebFetchError(f"`{raw}` does not look like a valid web address.")
    return url


def split_markdown_sections(markdown: str, fallback_label: str) -> list[Section]:
    text = normalise_text(markdown)
    if not text:
        return []

    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [Section(label=fallback_label, text=text, ordinal=0)]

    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section(label=fallback_label, text=preamble, ordinal=0))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        sections.append(
            Section(label=heading[:120] or fallback_label, text=f"{heading}\n\n{body}", ordinal=len(sections)))

    return sections or [Section(label=fallback_label, text=text, ordinal=0)]


def _fetch(url: str, *, headers: dict[str, str], label: str) -> requests.Response:
    try:
        response = requests.get(url, headers=headers, timeout=WEB_TIMEOUT_SECONDS)
    except requests.Timeout as exc:
        raise WebFetchError(f"{label} timed out after {WEB_TIMEOUT_SECONDS}s.", cause=exc) from exc
    except requests.RequestException as exc:
        raise WebFetchError(f"{label} could not be reached.", cause=exc) from exc
    if response.status_code >= 400:
        raise WebFetchError(f"{label} returned HTTP {response.status_code}.")
    return response


def fetch_via_jina(url: str) -> str:
    response = _fetch(JINA_READER_PREFIX + url, headers={"Accept": "text/plain", "User-Agent": _USER_AGENT},
                      label="The reader service")
    text = normalise_text(response.text)
    if len(text) < 40:
        raise WebFetchError("The reader service returned an empty page.")
    return text


def fetch_direct(url: str) -> tuple[str, str]:
    response = _fetch(url,
                      headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9"},
                      label="The website")
    content_type = response.headers.get("content-type", "").lower()
    raw = response.text

    if "html" in content_type or raw.lstrip()[:200].lower().startswith(("<!doctype", "<html")):
        import html as html_module
        text = _COMMENT_RE.sub(" ", raw)
        text = _SCRIPT_STYLE_RE.sub(" ", text)
        text = _CHROME_RE.sub(" ", text)
        text = _BR_RE.sub("\n", text)
        text = _BLOCK_END_RE.sub("\n", text)
        text = _TAG_RE.sub(" ", text)
        text = html_module.unescape(text)
        return normalise_text(text), url
    return normalise_text(raw), url


def ingest_url(raw_url: str) -> Source:
    url = normalise_url(raw_url)
    host = urlparse(url).netloc
    title = url
    reader_failure: WebFetchError | None = None

    try:
        text = fetch_via_jina(url)
        first_line, _, remainder = text.partition("\n")
        if first_line.lower().startswith("title:"):
            title = first_line.split(":", 1)[1].strip() or url
            text = remainder.strip()
        else:
            title = host
    except WebFetchError as exc:
        reader_failure = exc
        text, title = fetch_direct(url)

    if not text or len(text) < 40:
        raise WebFetchError(f"`{url}` contained no readable text.", cause=reader_failure)

    sections = split_markdown_sections(text, fallback_label=host or "page")
    sections, truncated = truncate_sections(sections, MAX_CHARS_PER_SOURCE)

    import hashlib
    fingerprint = hashlib.sha256((url + "\n" + text).encode("utf-8")).hexdigest()[:32]

    return Source(
        kind="web",
        title=(title or host)[:160],
        locator=url,
        sections=sections,
        fingerprint=fingerprint,
    )


def _store_source(source: Source, vectorstore, manifest: Manifest, splitter, force: bool, report: dict) -> None:
    if not force and manifest.has(source.fingerprint):
        report["skipped"].append(source.display_name)
        return

    documents = chunk_source(source, splitter)
    if not documents:
        report["failed"].append((source.display_name, ArzensError("No text could be extracted.")))
        return

    add_documents(vectorstore, documents)
    manifest.add(
        SourceRecord(
            source_id=source.fingerprint,
            kind=source.kind,
            title=source.display_name,
            locator=source.locator,
            chunk_count=len(documents),
            char_count=source.char_count,
            ingested_at=time.time(),
        )
    )
    report["added"].append((source.display_name, len(documents)))


def ingest_sources(*, urls: list[str], pdfs: list, vectorstore, manifest: Manifest, force: bool,
                   progress_bar=None) -> dict:
    report = {"added": [], "skipped": [], "failed": []}
    splitter = build_splitter()
    total = len(urls) + len(pdfs)
    if total == 0:
        return report

    done = 0

    def advance(label: str):
        nonlocal done
        done += 1
        if progress_bar:
            progress_bar.progress(min(done / total, 1.0), text=label)

    for raw_url in urls:
        try:
            if progress_bar: progress_bar.progress(done / total, text=f"Reading {raw_url}…")
            source = ingest_url(raw_url)
            _store_source(source, vectorstore, manifest, splitter, force, report)
        except ArzensError as exc:
            report["failed"].append((raw_url, exc))
        except Exception as exc:
            report["failed"].append((raw_url, ArzensError(f"`{raw_url}` failed unexpectedly.", cause=exc)))
        advance(f"Processed {raw_url}")

    for upload in pdfs:
        name = getattr(upload, "name", "document.pdf")
        try:
            if progress_bar: progress_bar.progress(done / total, text=f"Reading {name}…")
            source = ingest_pdf(upload.getvalue(), name)
            _store_source(source, vectorstore, manifest, splitter, force, report)
        except ArzensError as exc:
            report["failed"].append((name, exc))
        except Exception as exc:
            report["failed"].append((name, ArzensError(f"`{name}` failed unexpectedly.", cause=exc)))
        advance(f"Processed {name}")

    return report