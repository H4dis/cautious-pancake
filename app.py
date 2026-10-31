from __future__ import annotations
import secrets
import re
import streamlit as st
from config import APP_TITLE, ConfigError, get_settings, DATA_DIR, CHROMA_DIR
from dotenv import load_dotenv

load_dotenv()
from embeddings import APINinjasEmbeddings, VectorDiskCache
from vector_store import get_vectorstore, Manifest, drop_workspace, delete_source
from rag_engine import build_llm, RAGAgent, summarize_document, answer_with_full_context
from pipeline import ingest_sources

APP_ICON = "📜"

st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide", initial_sidebar_state="expanded")


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
                    if st.button("📝 Summarize", key=f"sum_{record.source_id}"):
                        st.session_state.pending_question = f"__SUMMARIZE__:{record.title}"
                        st.rerun()

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
                if "rag_agent" in st.session_state:
                    st.session_state.rag_agent.reset("default_user")
                st.rerun()
            if st.button("Delete everything", use_container_width=True):
                drop_workspace(workspace_id)
                st.session_state.turns = []
                st.query_params.clear()
                st.rerun()


def render_history() -> None:
    for turn in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])


def answer_question(question: str, vectorstore, manifest: Manifest, llm) -> None:
    if question.startswith("__SUMMARIZE__:"):
        filename = question.split(":", 1)[1]
        with st.chat_message("user"):
            st.markdown(f"Summarize: **{filename}**")
        with st.chat_message("assistant"):
            try:
                summary_stream = summarize_document(llm, vectorstore, filename)
                answer = st.write_stream(summary_stream)
                st.session_state.turns.append({"question": f"Summarize {filename}", "answer": answer})
            except Exception as exc:
                st.error(f"Summarization error: {exc}")
        return

    with st.chat_message("user"):
        st.markdown(question)

    # حالت انتخاب رو بیرون از chat_message گذاشتیم تا قاطی نشه
    mode = st.radio(
        "Answer mode",
        ["Vector Search (Fast)", "Full Document (Accurate)"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"mode_{len(st.session_state.turns)}"  # کلید یکتا برای هر سوال
    )

    with st.chat_message("assistant"):
        if mode == "Full Document (Accurate)":
            # حالت متن کامل
            filename = list(manifest.records.values())[0].title if manifest.records else ""
            stream = answer_with_full_context(llm, vectorstore, filename, question)
            answer = st.write_stream(stream)
            st.session_state.turns.append({"question": question, "answer": answer})

        else:
            # حالت جستجوی برداری
            if "rag_agent" not in st.session_state:
                st.session_state.rag_agent = RAGAgent(llm, vectorstore)

            stream = st.session_state.rag_agent.stream_ask(
                session_id="default_user",
                question=question
            )
            answer = st.write_stream(stream)

            if answer and "(Source:" in answer:
                st.session_state.turns.append({"question": question, "answer": answer})

                with st.expander("Sources", expanded=False):
                    with st.spinner("Loading context…"):
                        docs = vectorstore.similarity_search(question, k=3)
                        for i, doc in enumerate(docs):
                            title = doc.metadata.get("title", "Unknown")
                            page = doc.metadata.get("section", "N/A")
                            st.markdown(f"**[{i + 1}] {title} (Page: {page})**")
                            st.markdown(f"> {doc.page_content[:300]}...")
                            st.divider()
            else:
                st.session_state.turns.append({"question": question, "answer": answer})

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