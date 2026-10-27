
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st






CHAT_MODEL = "gpt-4o-mini"

#
EMBEDDING_MODEL = "api-ninjas-embeddings"
EMBEDDING_DIMENSIONS = 768


CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


RETRIEVAL_K = 6
RELEVANCE_FLOOR = 0.30


HISTORY_TURNS = 12


MAX_PDF_MB = 25
MAX_PDFS_PER_UPLOAD = 10
MAX_URLS_PER_INGEST = 10
MAX_CHARS_PER_SOURCE = 600_000
WEB_TIMEOUT_SECONDS = 45

DATA_DIR = Path(os.environ.get("ARZENS_DATA_DIR", ".arzens_data"))
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "embedding_cache"
MANIFEST_DIR = DATA_DIR / "manifests"

APP_TITLE = "VerilyAI"
WORKSPACE_TTL_HOURS = 48


class ArzensError(Exception):
    """Base class. Anything the UI catches is one of these."""

    default_message = "Something went wrong."

    def __init__(
        self,
        user_message: str | None = None,
        *,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.user_message = user_message or self.default_message
        self.hint = hint
        self.cause = cause
        super().__init__(self.user_message)

    def display(self) -> str:
        return f"{self.user_message}\n\n{self.hint}" if self.hint else self.user_message

    def technical_detail(self) -> str:
        if self.cause is None:
            return "No additional detail."
        return f"{type(self.cause).__name__}: {self.cause}"


class ConfigError(ArzensError):
    default_message = "The app is not configured correctly."





#class ConfigError(Exception):
    """Base class. Anything the UI catches is one of these."""

   # default_message = "Something went wrong."

    def __init__(
        self,
        user_message: str | None = None,
        *,
        hint: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.user_message = user_message or self.default_message
        self.hint = hint
        self.cause = cause
        super().__init__(self.user_message)

    def display(self) -> str:
        return f"{self.user_message}\n\n{self.hint}" if self.hint else self.user_message






@dataclass(frozen=True)
class Settings:
    api_ninjas_key: str
    gapgpt_api_key: str
    chat_model: str = CHAT_MODEL
    embedding_model: str = EMBEDDING_MODEL
    data_dir: Path = field(default=DATA_DIR)


def _from_streamlit_secrets(name: str) -> str | None:

    try:
        value = st.secrets.get(name)  # type: ignore[union-attr]
    except Exception:
        return None
    return str(value) if value else None


def resolve_api_ninjas_key() -> str:
    for getter in (
        lambda: _from_streamlit_secrets("API_NINJAS_KEY"),
        lambda: os.environ.get("API_NINJAS_KEY"),
    ):
        value = getter()
        if value and value.strip():
            return value.strip()
    raise ConfigError(
        "No API Ninjas key found.",
        hint=(
            "Add API_NINJAS_KEY to your .env or .streamlit/secrets.toml.\n"
            'Example: API_NINJAS_KEY = "your-key-here"'
        ),
    )


def resolve_gapgpt_api_key() -> str:
    for getter in (
        lambda: _from_streamlit_secrets("GAPGPT_API_KEY"),
        lambda: os.environ.get("GAPGPT_API_KEY"),
    ):
        value = getter()
        if value and value.strip():
            return value.strip()
    raise ConfigError(
        "No GapGPT key found.",
        hint=(
            "Add GAPGPT_API_KEY to your .env or .streamlit/secrets.toml.\n"
            'Example: GAPGPT_API_KEY = "your-key-here"'
        ),
    )


def get_settings() -> Settings:
    return Settings(
        api_ninjas_key=resolve_api_ninjas_key(),
        gapgpt_api_key=resolve_gapgpt_api_key(),
    )




def ensure_directories() -> None:
    for path in (DATA_DIR, CHROMA_DIR, CACHE_DIR, MANIFEST_DIR):
        path.mkdir(parents=True, exist_ok=True)