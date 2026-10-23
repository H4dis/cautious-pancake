# RAG Chatbot with Streamlit

A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your custom PDF documents, with source page citations. Built as part of the Datheon AI Engineering tasks (Task 2 & Task 4).

![Status](https://img.shields.io/badge/status-working-success)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.3%2B-red)
![LangChain](https://img.shields.io/badge/langchain-0.3%2B-green)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the App](#-running-the-app)
- [Project Structure](#-project-structure)
- [Design Decisions](#-design-decisions)
- [Known Limitations](#-known-limitations)
- [Troubleshooting](#-troubleshooting)
- [Lessons Learned](#-lessons-learned)
- [License](#-license)

---

## ✨ Features

- 📄 **PDF Ingestion** — Upload PDF documents and automatically chunk them into searchable passages.
- 🔍 **Semantic Search** — Uses vector embeddings + ChromaDB for similarity-based retrieval.
- 💬 **Conversational Memory** — Remembers previous turns and resolves pronouns (e.g., "what about the second point?").
- 🌊 **Streaming Responses** — Answers appear token-by-token in real-time via `st.write_stream`.
- 📚 **Source Citations** — Every answer is traced back to the exact page in the original PDF.
- 🎨 **Clean UI** — Responsive Streamlit interface with chat history and sidebar controls.

---

## 🏗 Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF       │────▶│ Text         │────▶│  Chunks     │
│   Upload    │     │ Extraction   │     │ (1200 char) │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │◀────│  LLM         │◀────│  ChromaDB   │
│   + Sources │     │ (GapGPT)     │     │ (Embeddings)│
└─────────────┘     └──────────────┘     └─────────────┘
                           ▲
                           │
                    ┌──────────────┐
                    │  User Query  │
                    └──────────────┘
```

**Flow:**
1. User uploads a PDF → parsed with `pypdf`.
2. Text is split with `RecursiveCharacterTextSplitter` (chunk=1200, overlap=180).
3. Each chunk is embedded and stored in ChromaDB.
4. When a user asks a question, relevant chunks are retrieved via cosine similarity.
5. Retrieved context + conversation history is sent to the LLM.
6. The answer streams back, with page citations attached.

---

## ⚙️ Prerequisites

- Python 3.10 or higher
- Two API keys (see [Configuration](#-configuration)):
  - **Embedding**: [API Ninjas](https://api-ninjas.com/api/embeddings) or Hugging Face (local)
  - **LLM**: [GapGPT](https://gapgpt.app) or any OpenAI-compatible endpoint

---

## 🚀 Installation

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/verily-ai.git
cd verily-ai

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`:**
```txt
streamlit>=1.30
langchain>=0.3
langchain-core>=0.3
langchain-community>=0.3
langchain-openai>=0.2
langchain-chroma>=0.1
chromadb>=0.5
pypdf>=4.0
requests>=2.31
python-dotenv>=1.0
```

---

## 🔑 Configuration

Create a `.env` file in the project root (next to `app.py`):

```env
# Embedding provider (API Ninjas)
API_NINJAS_KEY="your_api_ninjas_key_here"

# LLM provider (GapGPT)
GAPGPT_API_KEY="your_gapgpt_key_here"
```

Alternatively, for Streamlit Cloud deployment, create `.streamlit/secrets.toml`:

```toml
API_NINJAS_KEY = "your_api_ninjas_key_here"
GAPGPT_API_KEY = "your_gapgpt_key_here"
```

> ⚠️ **Never commit `.env` or `secrets.toml` to Git.** Add them to `.gitignore`.

**`.gitignore`:**
```
.env
.streamlit/secrets.toml
__pycache__/
venv/
*.pyc
.arzens_data/
```

---

## 🏃 Running the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**What to do:**
1. Upload a **text-based PDF** (not scanned) in the sidebar.
2. Click **Build knowledge base**.
3. Ask a question in the chat box.
4. View the answer + source citations.

---

## 📁 Project Structure

```
verily-ai/
├── app.py                          # Main Streamlit app + RAG pipeline
├── requirements.txt                # Python dependencies
├── .env                            # API keys (gitignored)
├── .streamlit/
│   └── secrets.toml                # Streamlit Cloud secrets (gitignored)
├── .gitignore
├── README.md
└── .arzens_data/                   # Local ChromaDB + cache (gitignored)
```

---

## 🎯 Design Decisions

### Why LangChain instead of raw SDK?
LangChain provides a unified interface across providers (OpenAI, Gemini, GapGPT). Switching LLM providers only requires changing `base_url` and model name — no rewrite of the RAG logic.

### Why ChromaDB instead of FAISS?
ChromaDB has a simpler API for persistence, supports metadata filtering out of the box, and integrates natively with LangChain. FAISS is faster for very large datasets but requires manual persistence.

### Why chunk size 1200 / overlap 180?
- **1200 chars** ≈ 300 tokens, well within most embedding model limits (5000 char cap on API Ninjas).
- **180 char overlap** preserves context across chunk boundaries (e.g., a sentence split across two chunks).
- This is a balanced default; for dense technical PDFs, reduce to 800/100.

### Why `st.write_stream` for streaming?
Streamlit's native `st.write_stream` accepts any Python generator and renders tokens as they arrive. This is cleaner than manually managing `placeholder.markdown()` loops, and works with LangChain's `.stream()` out of the box.

### Why separate providers for Embedding and LLM?
GapGPT (chosen for LLM due to Iran accessibility) does not offer an Embeddings endpoint. API Ninjas offers Embeddings but not Chat. Using two providers keeps each service doing what it does best. This is documented in `DECISIONS.md` (see below).

---

## ⚠️ Known Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| **Scanned PDFs not supported** | Image-based PDFs return "no extractable text" | Run OCR first (e.g., Adobe Acrobat, iLovePDF) |
| **JavaScript-heavy URLs fail** | `requests` can't execute JS | Save page as PDF, then upload |
| **API Ninjas: 5000 char limit** | Very long chunks may be truncated | Keep chunk size ≤ 1200 |
| **API Ninjas: monthly quota** | Free tier limited | Use Hugging Face local embeddings |
| **GapGPT: no Embeddings** | Cannot use one provider for both | Two-provider setup (documented above) |
| **Single-file app.py** | ~2200 lines, hard to navigate | Split into `rag.py` + `ui.py` (future work) |

---

## 🔧 Troubleshooting

### `No Gemini API key found` or `No API Ninjas key found`
- Check that `.env` or `.streamlit/secrets.toml` exists and contains the exact key name.
- Variable names are **case-sensitive**: `API_NINJAS_KEY` ≠ `api_ninjas_key`.
- Try adding a debug print in `resolve_api_ninjas_key()`:
  ```python
  print(f"DEBUG: value length = {len(value) if value else 0}")
  ```

### `contains no extractable text`
- Your PDF is scanned. Run it through OCR first.
- Or test with a text-based PDF (e.g., an arXiv paper).

### `429 Too Many Requests`
- API Ninjas free tier quota exhausted.
- Wait a minute, or switch to Hugging Face local embeddings.

### `missing ScriptRunContext`
- You ran `python app.py` instead of `streamlit run app.py`.
- Streamlit requires its own runner.

---

## 📝 Lessons Learned

This project was built as part of the Datheon AI Engineering tasks. Key takeaways:

1. **Provider selection matters more than framework choice.** Choosing a provider before verifying its capabilities (does it support Embeddings? What's the rate limit?) costs more time than rewriting the LLM call.
2. **Test with the simplest input first.** A 2-page text PDF is the correct first test, not a 50-page scanned document.
3. **Commit early, commit often.** Even local-only commits (`git commit -m "wip"`) prevent losing work when switching branches or reverting.
4. **Secrets belong in `.env`, never in chat.** API keys posted publicly are the #1 cause of unexpected bills.
5. **Streamlit apps need `streamlit run`.** The PyCharm "Run" button won't work for Streamlit.

---

## 📄 License

MIT License — see `LICENSE` for details.

---

## 🙏 Acknowledgments

- Built on top of [](https://github.com/RAG-System) (original Streamlit RAG template).
- Inspired by LangChain's structured output examples.
- Thanks to the Datheon team for the task specifications.
