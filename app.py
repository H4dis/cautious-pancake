from __future__ import annotations
import secrets
import re
import streamlit as st
from config import (
    APP_TITLE,
    ConfigError,
    get_settings,
    DATA_DIR,
    CHROMA_DIR,
)
from dotenv import load_dotenv
load_dotenv()
from embeddings import APINinjasEmbeddings, VectorDiskCache
from vector_store import get_vectorstore, Manifest, drop_workspace, delete_source, retrieve
from rag_engine import build_llm, Turn, condense_question, to_passages, build_messages, stream_answer, describe_library
from prompts import build_system_prompt
from pipeline import ingest_sources

APP_ICON = "📜"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")


def inject_brand_css():
    # (همان کدهای CSS قبلی را اینجا قرار بده)
    pass


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str):
    return APINinjasEmbeddings(api_key=api_key, cache=VectorDiskCache())


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str):
    return build_llm(api_key)


def is_valid_workspace_id(value: str | None) -> bool:
    return bool(value) and bool(re.match(r"^[a-f0-9]{16}$", str(value)))


def new_workspace_id() -> str:
    return secrets.token_hex(8)


def resolve_workspace_id() -> str:
    existing = st.query_params.get("ws")
    if is_valid_workspace_id(existing):
        return str(existing)
    workspace_id = new_workspace_id()
    st.query_params["ws"] = workspace_id
    return workspace_id


def init_state() -> None:
    defaults = {"turns": [], "pending_question": None, "last_ingest_report": None}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_ingest_report(report: dict):
    for name, count in report["added"]:
        st.success(f"**{name}** — {count} passages indexed.")
    if report["skipped"]:
        st.info("Already indexed, skipped: " + ", ".join(f"**{n}**" for n in report["skipped"]))
    for name, error in report["failed"]:
        st.error(f"**{name}** — {error.display()}")


def render_sidebar(vectorstore, manifest: Manifest, workspace_id: str) -> None:
    with st.sidebar:
        st.header("Knowledge base")
        source_count, chunk_count = manifest.summary()
        st.caption(f"{source_count} source(s) · {chunk_count} passages indexed")

        with st.form("ingest_form", clear_on_submit=False):
            urls_raw = st.text_area("Website URLs", height=90)
            pdfs = st.file_uploader("PDF files", type=["pdf"], accept_multiple_files=True)
            force = st.checkbox("Force re-index", value=False)
            submitted = st.form_submit_button("Build knowledge base", type="primary", use_container_width=True)

        if submitted:
            urls = [u.strip() for u in (urls_raw or "").splitlines() if u.strip()]
            pdfs = pdfs or []
            if urls or pdfs:
                progress = st.progress(0.0)
                report = ingest_sources(urls=urls, pdfs=pdfs, vectorstore=vectorstore, manifest=manifest, force=force,
                                        progress_bar=progress)
                progress.empty()
                st.session_state.last_ingest_report = report
                render_ingest_report(report)

        if manifest.records:
            st.divider()
            for record in sorted(manifest.records.values(), key=lambda r: r.ingested_at, reverse=True):
                icon = "🌐" if record.kind == "web" else "📄"
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"{icon} **{record.title}**")
                with right:
                    if st.button("✕", key=f"del_{record.source_id}", use_container_width=True):
                        try:
                            delete_source(vectorstore, record.source_id)
                            manifest.remove(record.source_id)
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

        st.divider()
        with st.expander("Session"):
            if st.button("Clear chat", use_container_width=True):
                st.session_state.turns = []
                st.rerun()
            if st.button("Delete everything", use_container_width=True):
                drop_workspace(workspace_id)
                st.session_state.turns = []
                st.query_params.clear()
                st.rerun()


def render_history() -> None:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)
            if turn.citations:
                with st.expander(f"Sources ({len(turn.citations)})"):
                    for citation in turn.citations:
                        st.markdown(f"- {citation}")


def collect_citations(answer: str, passages) -> list[str]:
    cited_numbers = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)}
    seen: list[str] = []
    for passage in passages:
        if passage.number in cited_numbers and passage.citation not in seen:
            seen.append(passage.citation)
    return seen


def answer_question(question: str, vectorstore, manifest: Manifest, llm) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        passages = []
        try:
            with st.spinner("Searching…"):
                search_query = condense_question(llm, question, st.session_state.turns)
                from vector_store import retrieve
                scored = retrieve(vectorstore, search_query)
                passages = to_passages(scored)
        except Exception as exc:
            st.error(f"Search error: {exc}")
            return

        system_prompt = build_system_prompt(app_name=APP_TITLE,
                                            library=describe_library([r.title for r in manifest.records.values()]))
        messages = build_messages(system_prompt=system_prompt, history=st.session_state.turns, question=question,
                                  passages=passages)

        try:
            answer = st.write_stream(stream_answer(llm, messages))
        except Exception as exc:
            st.error(f"Generation error: {exc}")
            return

        answer = answer if isinstance(answer, str) else str(answer)
        citations = collect_citations(answer, passages)

        if citations:
            with st.expander(f"Sources ({len(citations)})", expanded=False):
                for passage in passages:
                    if passage.citation not in citations: continue
                    st.markdown(f"**[{passage.number}] {passage.citation}**")
                    st.caption(f"relevance {passage.score:.2f}")
                    st.markdown(f"> {passage.text[:700]}...")
                    st.divider()

        st.session_state.turns.append(Turn(question=question, answer=answer, citations=citations))


def main() -> None:
    init_state()
    st.title(f"{APP_ICON} {APP_TITLE}")

    try:
        settings = get_settings()
    except ConfigError as exc:
        st.error(exc.display())
        st.stop()
        return

    workspace_id = resolve_workspace_id()
    manifest = Manifest(workspace_id)
    manifest.touch()

    try:
        embeddings = get_embeddings(settings.api_ninjas_key)
        vectorstore = get_vectorstore(workspace_id, embeddings)
        llm = get_llm(settings.gapgpt_api_key)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
        return

    render_sidebar(vectorstore, manifest, workspace_id)
    render_history()

    question = st.session_state.pending_question or st.chat_input("Ask a question about your sources…")
    st.session_state.pending_question = None

    if question:
        if not manifest.records:
            with st.chat_message("assistant"):
                st.warning("The knowledge base is empty, add a source first.")
            return
        answer_question(question, vectorstore, manifest, llm)


if __name__ == "__main__":
    main()