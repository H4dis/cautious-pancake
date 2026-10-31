from __future__ import annotations
import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


@st.cache_resource(show_spinner=False)
def get_reranker():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')


def rerank_documents(query: str, documents: list[tuple[Document, float]], top_n: int = 4) -> list[
    tuple[Document, float]]:
    if not documents:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc, _ in documents]
    scores = reranker.predict(pairs)

    ranked = [(doc, score) for (doc, _), score in zip(documents, scores)]
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked[:top_n]