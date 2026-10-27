from __future__ import annotations

import hashlib
import math
import struct
import time
from pathlib import Path

import requests
from langchain_core.embeddings import Embeddings

from config import (
    CACHE_DIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    ArzensError
)

DOCUMENT_TASK = "RETRIEVAL_DOCUMENT"
QUERY_TASK = "RETRIEVAL_QUERY"
_BATCH_SIZE = 32
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 1.5


class EmbeddingError(ArzensError):
    default_message = "The documents could not be embedded."


class VectorDiskCache:
    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(text: str, *, model: str, dimensions: int, task_type: str) -> str:
        digest = hashlib.sha256(
            f"{model}|{dimensions}|{task_type}|{text}".encode("utf-8")
        ).hexdigest()
        return digest

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.vec"

    def get(self, key: str) -> list[float] | None:
        path = self._path(key)
        try:
            raw = path.read_bytes()
        except (FileNotFoundError, OSError):
            self.misses += 1
            return None
        if not raw or len(raw) % 4 != 0:
            self.misses += 1
            return None
        self.hits += 1
        return list(struct.unpack(f"<{len(raw) // 4}f", raw))

    def put(self, key: str, vector: list[float]) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(struct.pack(f"<{len(vector)}f", *vector))
            tmp.replace(path)
        except OSError:
            pass


def _normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429", "rate limit", "resource_exhausted", "quota",
            "500", "502", "503", "504", "unavailable", "internal",
            "deadline", "timeout", "connection",
        )
    )


def _embedding_error_message(exc: Exception | None) -> str:
    text = str(exc or "").lower()
    if "api key" in text or "api_key" in text or "401" in text or "permission" in text:
        return "The API key was rejected while embedding."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "The embedding quota was exhausted."
    return "The documents could not be embedded."


def _embedding_error_hint(exc: Exception | None) -> str:
    text = str(exc or "").lower()
    if "api key" in text or "401" in text or "permission" in text:
        return "Check API_NINJAS_KEY in your Streamlit secrets or environment."
    if "429" in text or "quota" in text:
        return "Wait a minute and try again, or ingest fewer documents at once."
    return "Try again in a moment. Progress so far has been cached."


class APINinjasEmbeddings(Embeddings):
    def __init__(
        self,
        api_key: str,
        *,
        model: str = EMBEDDING_MODEL,
        dimensions: int = EMBEDDING_DIMENSIONS,
        cache: VectorDiskCache | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.cache = cache or VectorDiskCache()
        self.api_key = api_key
        self.url = "https://api.api-ninjas.com/v1/embeddings"
        self.headers = {"X-Api-Key": api_key}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_many(texts, task_type=DOCUMENT_TASK)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_many([text], task_type=QUERY_TASK)[0]

    def _embed_many(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []

        keys = [
            VectorDiskCache.make_key(
                text, model=self.model, dimensions=self.dimensions, task_type=task_type
            )
            for text in texts
        ]

        results: list[list[float] | None] = [self.cache.get(key) for key in keys]
        pending = [i for i, vector in enumerate(results) if vector is None]

        for start in range(0, len(pending), _BATCH_SIZE):
            batch_indices = pending[start : start + _BATCH_SIZE]
            batch_texts = [texts[i] for i in batch_indices]
            vectors = self._call_api(batch_texts, task_type=task_type)
            for index, vector in zip(batch_indices, vectors):
                normalised = _normalise(vector)
                results[index] = normalised
                self.cache.put(keys[index], normalised)

        missing = [i for i, vector in enumerate(results) if vector is None]
        if missing:
            raise EmbeddingError(
                f"{len(missing)} passage(s) could not be embedded.",
                hint="Try ingesting again — partial progress was cached, so the "
                "retry will only redo what failed.",
            )
        return [vector for vector in results if vector is not None]

    def _call_api(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        last_error: Exception | None = None
        vectors: list[list[float]] = []

        for text in texts:
            for attempt in range(_MAX_RETRIES):
                try:
                    payload = {"text": text[:5000]}
                    response = requests.post(
                        self.url, headers=self.headers, json=payload, timeout=30
                    )
                    response.raise_for_status()
                    vectors.append(response.json()["embeddings"])
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if not _is_transient(exc) or attempt == _MAX_RETRIES - 1:
                        break
                    time.sleep(_RETRY_BASE_SECONDS * (2**attempt))

            if last_error is not None:
                raise EmbeddingError(
                    _embedding_error_message(last_error),
                    hint=_embedding_error_hint(last_error),
                    cause=last_error,
                )

        return vectors