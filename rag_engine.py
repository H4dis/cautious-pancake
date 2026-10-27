from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (
    CHAT_MODEL,
    HISTORY_TURNS,
    ArzensError
)
from prompts import _CONDENSE_SYSTEM, build_system_prompt

_MAX_CONDENSE_CHARS = 4000


class GenerationError(ArzensError):
    default_message = "The assistant could not produce an answer."


@dataclass
class Turn:
    question: str
    answer: str
    citations: list[str] | None = None


@dataclass
class Passage:
    number: int
    document: Document
    score: float

    @property
    def citation(self) -> str:
        return str(self.document.metadata.get("citation", "source"))

    @property
    def locator(self) -> str:
        return str(self.document.metadata.get("locator", ""))

    @property
    def kind(self) -> str:
        return str(self.document.metadata.get("kind", ""))

    @property
    def text(self) -> str:
        return self.document.page_content


def build_llm(api_key: str, *, model: str = CHAT_MODEL, temperature: float = 0.2):
    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.gapgpt.app/v1",
            temperature=temperature,
            max_tokens=2048,
        )
    except Exception as exc:
        raise GenerationError(
            "The language model could not be initialised.",
            hint="Check that your GapGPT API key is valid.",
            cause=exc,
        ) from exc


def condense_question(llm, question: str, history: Sequence[Turn]) -> str:
    if not history or not question.strip():
        return question.strip()

    transcript_lines: list[str] = []
    for turn in list(history)[-HISTORY_TURNS:]:
        transcript_lines.append(f"User: {turn.question}")
        transcript_lines.append(f"Assistant: {turn.answer}")
    transcript = "\n".join(transcript_lines)[-8000:]

    messages = [
        SystemMessage(content=_CONDENSE_SYSTEM),
        HumanMessage(
            content=f"Conversation so far:\n{transcript}\n\n"
            f"Latest message: {question}\n\nStandalone query:"
        ),
    ]

    try:
        response = llm.invoke(messages)
        rewritten = (getattr(response, "content", "") or "").strip()
    except Exception:
        return question.strip()

    if not rewritten or len(rewritten) > 500:
        return question.strip()
    return rewritten


def to_passages(scored: Sequence[tuple[Document, float]]) -> list[Passage]:
    return [
        Passage(number=i, document=doc, score=float(score or 0.0))
        for i, (doc, score) in enumerate(scored, start=1)
    ]


def format_context(passages: Sequence[Passage]) -> str:
    if not passages:
        return "(no relevant passages were found in the knowledge base)"
    return "\n\n".join(
        f'[source="{p.citation}", page="{p.document.metadata.get("section_ordinal", "N/A")}", relevance={p.score:.2f}]\n{p.text}'
        for p in passages
    )


def summarize_document(llm, vectorstore, filename: str) -> Iterator[str]:
    try:
        results = vectorstore.get(where={"title": filename})
        if not results or not results["documents"]:
            yield "no content found for summerizing ."
            return


        full_text = "\n\n".join(results["documents"])


        messages = [
            SystemMessage(
                content="You are a helpful assistant. Summarize the following document concisely in 3-5 bullet points. Focus on the main topics and key findings."),
            HumanMessage(content=f"Document text:\n{full_text[:30000]}")  # محدودیت کاراکتر برای جلوگیری از خطای API
        ]


        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text

    except Exception as exc:
        yield f"error fail summerizing : {str(exc)}"

def describe_library(source_titles: Sequence[str], limit: int = 8) -> str:
    titles = [t for t in source_titles if t]
    if not titles:
        return "(the knowledge base is empty)"
    shown = ", ".join(titles[:limit])
    if len(titles) > limit:
        shown += f", and {len(titles) - limit} more"
    return shown


def build_messages(
    *,
    system_prompt: str,
    history: Sequence[Turn],
    question: str,
    passages: Sequence[Passage],
) -> list:
    messages: list = [SystemMessage(content=system_prompt)]

    for turn in list(history)[-HISTORY_TURNS:]:
        messages.append(HumanMessage(content=turn.question))
        messages.append(AIMessage(content=turn.answer))

    messages.append(
        HumanMessage(
            content=f"CONTEXT:\n{format_context(passages)}\n\n"
            f"QUESTION: {question}"
        )
    )
    return messages


def stream_answer(llm, messages: list) -> Iterator[str]:
    started = False
    try:
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None)
            if text is None:
                continue
            if isinstance(text, list):
                text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in text
                )
            if text:
                started = True
                yield text
    except Exception as exc:
        if started:
            yield f"\n\n---\n*The answer was cut short: {_generation_reason(exc)}*"
            return
        raise GenerationError(
            _generation_message(exc),
            hint=_generation_hint(exc),
            cause=exc,
        ) from exc

    if not started:
        yield (
            "The model returned an empty response. This usually means a safety "
            "filter blocked the output. Try rephrasing the question."
        )


def _generation_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "429" in text or "quota" in text:
        return "the API rate limit was hit"
    if "timeout" in text or "deadline" in text:
        return "the request timed out"
    return "the connection to the model dropped"


def _generation_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "401" in text or "permission" in text:
        return "The GapGPT API key was rejected."
    if "429" in text or "quota" in text or "resource_exhausted" in text:
        return "The GapGPT rate limit or quota was reached."
    if "404" in text or "not found" in text:
        return "The configured model is not available to this API key."
    if "safety" in text or "blocked" in text:
        return "The response was blocked by a safety filter."
    return "The assistant could not generate an answer."


def _generation_hint(exc: Exception) -> str:
    text = str(exc).lower()
    if "api key" in text or "401" in text:
        return "Check GAPGPT_API_KEY in your Streamlit secrets or environment."
    if "429" in text or "quota" in text:
        return "Wait a moment and ask again."
    if "404" in text:
        return f"Confirm `{CHAT_MODEL}` is enabled for your project in GapGPT."
    return "Try asking again. If it keeps failing, check your network connection."