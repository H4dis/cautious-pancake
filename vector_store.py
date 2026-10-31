from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
import streamlit as st

from config import CHROMA_DIR, MANIFEST_DIR, RETRIEVAL_K, RERANK_TOP_N, ArzensError

_client = None

class VectorStoreError(ArzensError):
    default_message = "The knowledge base could not be opened."

@st.cache_resource(show_spinner=False)
def _get_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

from config import (
    CHROMA_DIR,
    RETRIEVAL_K,
    RELEVANCE_FLOOR,
    MANIFEST_DIR,
    ArzensError
)
from chunking import chunk_ids

_client = None


class VectorStoreError(ArzensError):
    default_message = "The knowledge base could not be opened."


def collection_name(workspace_id: str) -> str:
    return f"ws_{workspace_id}"


def manifest_path(workspace_id: str) -> Path:
    return MANIFEST_DIR / f"{workspace_id}.json"


@dataclass
class SourceRecord:
    source_id: str
    kind: str
    title: str
    locator: str
    chunk_count: int
    char_count: int
    ingested_at: float

    @property
    def ingested_label(self) -> str:
        return time.strftime("%d %b %H:%M", time.localtime(self.ingested_at))


class Manifest:
    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id
        self.path = manifest_path(workspace_id)
        self.records: dict[str, SourceRecord] = {}
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.records = {}
            return
        self.records = {
            key: SourceRecord(**value)
            for key, value in payload.get("sources", {}).items()
            if isinstance(value, dict)
        }

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "workspace_id": self.workspace_id,
                "updated_at": time.time(),
                "sources": {key: asdict(rec) for key, rec in self.records.items()},
            }
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            raise VectorStoreError("The knowledge base index could not be saved.", cause=exc) from exc

    def has(self, source_id: str) -> bool:
        return source_id in self.records

    def add(self, record: SourceRecord) -> None:
        self.records[record.source_id] = record
        self.save()

    def remove(self, source_id: str) -> None:
        self.records.pop(source_id, None)
        self.save()

    def touch(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self.save()
            else:
                self.path.touch()
        except OSError:
            pass

    def summary(self) -> tuple[int, int]:
        return len(self.records), sum(r.chunk_count for r in self.records.values())


def get_client():
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
        )
    except Exception as exc:
        raise VectorStoreError("The local vector database could not be opened.", cause=exc) from exc
    return _client


def get_vectorstore(workspace_id: str, embeddings):
    from langchain_chroma import Chroma
    try:
        return Chroma(
            client=get_client(),
            collection_name=collection_name(workspace_id),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            create_collection_if_not_exists=True,
        )
    except Exception as exc:
        raise VectorStoreError("This session's knowledge base could not be opened.", cause=exc) from exc


def add_documents(vectorstore, documents: list[Document]) -> int:
    if not documents:
        return 0
    try:
        vectorstore.add_documents(documents=documents, ids=chunk_ids(documents))
    except Exception as exc:
        raise VectorStoreError("The passages could not be written to the knowledge base.", cause=exc) from exc
    return len(documents)


def delete_source(vectorstore, source_id: str) -> None:
    try:
        vectorstore.delete(where={"source_id": source_id})
    except Exception as exc:
        raise VectorStoreError("That source could not be removed.", cause=exc) from exc


def drop_workspace(workspace_id: str) -> None:
    try:
        get_client().delete_collection(collection_name(workspace_id))
    except Exception:
        pass
    try:
        manifest_path(workspace_id).unlink(missing_ok=True)
    except OSError:
        pass
def get_vectorstore(workspace_id: str, embeddings):
    from langchain_chroma import Chroma
    try:
        return Chroma(
            client=get_client(),
            collection_name=collection_name(workspace_id),
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            create_collection_if_not_exists=True,
        )
    except Exception as exc:
        raise VectorStoreError("This session's knowledge base could not be opened.", cause=exc) from exc

def retrieve(vectorstore, query: str, *, k: int = RETRIEVAL_K, floor: float = 0.25) -> list[tuple[Document, float]]:
    if not query.strip():
        return []
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        raise VectorStoreError("The knowledge base could not be searched.", cause=exc) from exc


    fixed_results = []
    for doc, score in scored:
        if "source" not in doc.metadata:
            doc.metadata["source"] = doc.metadata.get("title", "Unknown")
        if "page" not in doc.metadata:
            doc.metadata["page"] = doc.metadata.get("section", "N/A")
        fixed_results.append((doc, score))

    filtered = [(doc, score) for doc, score in fixed_results if score is None or score >= floor]
    if not filtered:
        return []


    reranker = _get_reranker()
    pairs = [[query, doc.page_content] for doc, _ in filtered]
    scores = reranker.predict(pairs)

    ranked = [(doc, float(score)) for (doc, _), score in zip(filtered, scores)]
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked[:RERANK_TOP_N]


def retrieve(vectorstore, query: str, *, k: int = RETRIEVAL_K, floor: float = RELEVANCE_FLOOR) -> list[
    tuple[Document, float]]:
    if not query.strip():
        return []
    try:
        scored = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    except Exception as exc:
        raise VectorStoreError("The knowledge base could not be searched.", cause=exc) from exc

    fixed_results = []
    for doc, score in scored:
        if "source" not in doc.metadata:
            doc.metadata["source"] = doc.metadata.get("title", "Unknown")
        if "page" not in doc.metadata:
            doc.metadata["page"] = doc.metadata.get("section", "N/A")
        fixed_results.append((doc, score))

    filtered = [(doc, score) for doc, score in fixed_results if score is None or score >= floor]

    if not filtered:
        return []

    from reranker import rerank_documents
    from config import RERANK_TOP_N

    return rerank_documents(query, filtered, top_n=RERANK_TOP_N)