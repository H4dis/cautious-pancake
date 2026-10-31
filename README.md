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

## Features
- Upload PDF documents and build a knowledge base.
- Ask questions with source citations (Source, Page).
- Chat memory using LangChain's `RunnableWithMessageHistory`.
- Cross-Encoder Reranking for better retrieval accuracy.


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


## Architecture Notes
- Uses LangChain LCEL (`create_retrieval_chain`, `create_history_aware_retriever`) instead of the deprecated `RetrievalQA`.
- Embeddings are generated via API Ninjas; LLM inference via GapGPT.
- Reranking is done locally with `sentence-transformers` (`ms-marco-MiniLM-L-6-v2`).
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
OPENAI_API_KEY = "your_openai_key_here"
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
ai/
+-- app.py                  # Main Streamlit app
+-- config.py               # Settings and API keys
+-- rag_engine.py           # RAG chain, memory, and streaming
+-- vector_store.py         # ChromaDB and Reranking
+-- prompts.py              # System prompts
+-- pipeline.py             # PDF ingestion
+-- tests/
|   +-- test_rag.py         # Basic unit tests
+-- requirements.txt
+-- README.md
+-- .arzens_data/           # Local ChromaDB + cache (gitignored)
```

---

## 🎯 Design Decisions

### Why LangChain instead of raw SDK?
LangChain provides a unified interface across providers (OpenAI, Gemini, GapGPT). Switching LLM providers only requires changing `base_url` and model name — no rewrite of the RAG logic.

### Why ChromaDB instead of FAISS?
ChromaDB has a simpler API for persistence, supports metadata filtering out of the box, and integrates natively with LangChain. FAISS is faster for very large datasets but requires manual persistence.

### Why Reranking?
Retrieval with a bi-encoder (ChromaDB) can return irrelevant chunks. A Cross-Encoder reranks the top 20 candidates and passes only the top 4 to the LLM, improving accuracy and reducing hallucinations.



## ⚠️ Known Limitations

Limitation	Impact	Workaround
Scanned PDFs not supported	Image-based PDFs return "no extractable text"	Run OCR first
JavaScript-heavy URLs fail	requests can't execute JS	Save page as PDF
API Ninjas: 5000 char limit	Very long chunks may be truncated	Keep chunk size <= 1200
Troubleshooting
---

## 🔧 Troubleshooting

No API Ninjas key found / No OpenAI key found
Check that .env or .streamlit/secrets.toml contains the exact variable names.

Variable names are case-sensitive: OPENAI_API_KEY != openai_api_key.

contains no extractable text
Your PDF is scanned. Run it through OCR first.

429 Too Many Requests
API Ninjas quota exhausted. Wait a minute.

missing ScriptRunContext
Run streamlit run app.py, not python app.py.


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
