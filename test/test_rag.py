import sys
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking import build_splitter
from ingestion import Section, Source
from chunking import chunk_source
from prompts import _ANSWER_PROMPT


def test_chunking_logic():


    long_text = "This is a sentence. " * 100
    splitter = build_splitter(chunk_size=100, chunk_overlap=20)

    chunks = splitter.split_text(long_text)

    assert len(chunks) > 1, "متن طولانی باید به بیش از یک تکه تقسیم شود"
    assert all(len(c) <= 100 for c in chunks), "هیچ تکه‌ای نباید از chunk_size بزرگ‌تر باشد"


def test_source_creation_and_chunking():

    section = Section(label="page 1", text="Hello world. This is a test.", ordinal=1)
    source = Source(kind="pdf", title="test.pdf", locator="test.pdf", sections=[section], fingerprint="123")

    documents = chunk_source(source)

    assert len(documents) >= 1, "باید حداقل یک Document ساخته شود"
    doc = documents[0]
    assert doc.metadata["title"] == "test.pdf"
    assert doc.metadata["section"] == "page 1"
    assert "source_id" in doc.metadata
    assert "chunk_id" in doc.metadata


def test_prompt_formatting():

    app_name = "TestApp"
    context = "This is a sample context."

    formatted_prompt = _ANSWER_PROMPT.format(app_name=app_name, context=context)

    assert app_name in formatted_prompt
    assert context in formatted_prompt
    assert "retrieval-grounded" in formatted_prompt


if __name__ == "__main__":
    test_chunking_logic()
    test_source_creation_and_chunking()
    test_prompt_formatting()
    print("✅ All tests passed successfully!")