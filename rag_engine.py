from __future__ import annotations
from typing import Iterator
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from config import CHAT_MODEL, ArzensError
import prompts


class GenerationError(ArzensError):
    default_message = "The assistant could not produce an answer."


def build_llm(api_key: str, *, model: str = CHAT_MODEL, temperature: float = 0.2):
    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=2048,
        )
    except Exception as exc:
        raise GenerationError("The language model could not be initialised.", cause=exc) from exc


class RAGAgent:
    def __init__(self, llm, vectorstore):
        self._llm = llm
        self._vectorstore = vectorstore
        self._sessions: dict[str, BaseChatMessageHistory] = {}

        self._chain = RunnableWithMessageHistory(
            self._build_rag_chain(),
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_history(self, session_id: str) -> BaseChatMessageHistory:
        return self._sessions.setdefault(session_id, ChatMessageHistory())

    def _build_rag_chain(self):
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.build_rewrite_prompt()),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        retriever = self._vectorstore.as_retriever(search_kwargs={"k": 20 })

        history_aware_retriever = create_history_aware_retriever(
            self._llm, retriever, rewrite_prompt
        )

        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.build_answer_prompt("VerilyAI")),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        answer_chain = create_stuff_documents_chain(
            self._llm, answer_prompt, document_prompt=prompts.get_cited_chunk_template()
        )

        return create_retrieval_chain(history_aware_retriever, answer_chain)

    def stream_ask(self, session_id: str, question: str) -> Iterator[str]:
        for chunk in self._chain.stream(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
        ):
            if "answer" in chunk:
                yield chunk["answer"]

    def get_context(self, session_id: str) -> list[Document]:
        return []

    def reset(self, session_id: str):
        self._sessions.pop(session_id, None)


def summarize_document(llm, vectorstore, filename: str) -> Iterator[str]:
    try:
        results = vectorstore.get(where={"title": filename})
        if not results or not results["documents"]:
            yield "no content found for summarizing."
            return

        full_text = "\n\n".join(results["documents"])
        messages = [
            SystemMessage(
                content="You are a helpful assistant. Summarize the following document concisely in 3-5 bullet points. Focus on the main topics and key findings."),
            HumanMessage(content=f"Document text:\n{full_text[:30000]}")
        ]

        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text
    except Exception as exc:
        yield f"error fail summarizing: {str(exc)}"

def answer_with_full_context(llm, vectorstore, filename: str, question: str) -> Iterator[str]:
    try:
        results = vectorstore.get(where={"title": filename})
        if not results or not results["documents"]:
            yield "No content found."
            return
        full_text = "\n\n".join(results["documents"])[:50000]
        messages = [
            SystemMessage(content="Answer strictly from the document text."),
            HumanMessage(content=f"Document text:\n{full_text}\n\nQuestion: {question}")
        ]
        for chunk in llm.stream(messages):
            text = getattr(chunk, "content", None)
            if text:
                yield text
    except Exception as exc:
        yield f"Error: {str(exc)}"
