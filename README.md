# 📖 Document Q&A — RAG-Powered Chat Over Your Own Files

Ask questions in plain language and get answers grounded in your own documents — every claim traceable back to the exact source passage. Built end-to-end with a local, CPU-friendly Retrieval-Augmented Generation (RAG) pipeline.

**[🔗 Live Demo on HuggingFace Spaces](#)** *(add your Space link here once deployed)*

---

## ✨ Features

- **Multi-format ingestion** — PDF, DOCX, ODT, TXT, Markdown, XLSX, CSV, PPTX, and plain-text code files (`.py`, `.java`, `.cpp`, `.c`, `.html`, `.css`)
- **Fully local inference** — no external API calls; runs entirely on CPU using a quantized open-weight model
- **Grounded, cited answers** — every response links back to the specific passages it was generated from
- **Chat-style interface** — persistent conversation history, live document library, and processing stats
- **Graceful failure handling** — scanned/image-only PDFs, empty files, and unsupported formats are caught and reported per-file instead of crashing the app

---

## 🏗️ Architecture

```
Document Upload
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│   Loader    │────▶│   Splitter   │────▶│    Embedder     │
│ (per format)│     │ (chunking)   │     │ (MiniLM-L6-v2)  │
└─────────────┘     └──────────────┘     └────────────────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │  FAISS Index    │
                                          │ (vector store)  │
                                          └────────────────┘
                                                   │
        User Question ────────────────────────────┤
                                                   ▼
                                          ┌────────────────┐
                                          │   Retriever     │
                                          │  (top-k search) │
                                          └────────────────┘
                                                   │
                                                   ▼
                                          ┌────────────────┐
                                          │   TinyLlama     │
                                          │ (RAG generation)│
                                          └────────────────┘
                                                   │
                                                   ▼
                                          Grounded Answer + Sources
```

| Stage | Tool |
|---|---|
| Orchestration | LangChain |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | FAISS (CPU) |
| LLM | TinyLlama-1.1B-Chat (Q4_K_M GGUF) via `llama-cpp-python` |
| UI | Streamlit |
| Deployment | HuggingFace Spaces (CPU basic tier) |

---

## 📁 Project Structure

```
rag-doc-qa/
├── streamlit_app.py          # Main entry point
├── app/
│   ├── config.py
│   ├── ingestion/document_loader.py
│   ├── processing/text_splitter.py
│   ├── embeddings/embedder.py
│   ├── retrieval/vector_store.py
│   └── generation/llm_chain.py
├── models/                   # GGUF model (auto-downloaded, not committed)
├── data/faiss_index/         # Persisted vector index (generated, not committed)
├── requirements.txt
├── packages.txt              # System deps for HF Spaces
└── README.md
```

---

## 🚀 Running Locally

**Requirements:** Python 3.11 (recommended via conda — this project hit version-compatibility issues on Python 3.13)

```bash
# Create environment
conda create -n ragqa python=3.11 -y
conda activate ragqa

# Install dependencies
pip install -r requirements.txt
pip install llama-cpp-python --prefer-binary

# Run
streamlit run streamlit_app.py
```

The TinyLlama model (~640MB) downloads automatically on first run.

---

## ⚠️ Known Limitations

- **No OCR** — scanned/image-only PDFs without a text layer aren't supported
- **Ephemeral storage on free-tier HF Spaces** — the vector index and model re-download/rebuild after the Space sleeps and wakes up
- **CPU-only inference** — response latency is higher than a hosted API; expect several seconds to ~1 minute per answer depending on hardware
- **No cross-session persistence** — uploaded documents don't persist between browser sessions

---

## 🔭 Possible Future Improvements

- OCR support for scanned documents (Tesseract)
- Reranking retrieved chunks with a cross-encoder for improved relevance
- Multi-turn conversational memory (follow-up question support)
- Archive (`.zip`) support with recursive extraction

---

## 🧑‍💻 Author

Built by Sualiha as part of an ML/backend portfolio project.