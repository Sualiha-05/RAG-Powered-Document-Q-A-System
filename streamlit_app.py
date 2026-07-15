import streamlit as st
import tempfile, os, time
from huggingface_hub import hf_hub_download
from app.config import LLM_MODEL_PATH

from app.ingestion.document_loader import load_document
from app.processing.text_splitter import split_text
from app.retrieval.vector_store import build_vector_store, load_vector_store, get_retriever
from app.generation.llm_chain import build_qa_chain

def ensure_model():
    """Downloads the TinyLlama GGUF model on first run if it isn't already present."""
    if not os.path.exists(LLM_MODEL_PATH):
        os.makedirs(os.path.dirname(LLM_MODEL_PATH), exist_ok=True)
        with st.spinner("First-time setup: downloading language model (~640MB)..."):
            downloaded_path = hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                local_dir=os.path.dirname(LLM_MODEL_PATH),
            )
            # Ensure filename matches what config.py expects
            if downloaded_path != LLM_MODEL_PATH:
                os.rename(downloaded_path, LLM_MODEL_PATH)
 
ensure_model()


st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📖", layout="wide")

# ────────────────────────────────────────────────────────────────────────────
# THEME — "reading room" aesthetic: paper, ink, and margin annotations
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --paper: #F7F6F2;
    --ink: #1C1E26;
    --ink-soft: #4A4D5A;
    --forest: #2F5233;
    --forest-dark: #21391F;
    --forest-light: #E7EEE4;
    --gold: #C9A227;
    --gold-light: #FBF3D9;
    --sage-border: #D7DED2;
    --card-bg: #FFFFFF;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--paper) !important;
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

[data-testid="stSidebar"] {
    background-color: #EFEEE7;
    border-right: 1px solid var(--sage-border);
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Lora', serif;
    color: var(--forest-dark);
}

/* ---------- Hero header ---------- */
.hero {
    padding: 28px 32px;
    background: linear-gradient(135deg, var(--forest-dark) 0%, var(--forest) 100%);
    border-radius: 14px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -40%; right: -10%;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(201,162,39,0.25) 0%, transparent 70%);
}
.hero .eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 600;
}
.hero .title {
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 2.3rem;
    color: #FFFFFF;
    margin: 4px 0 6px 0;
}
.hero .subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.98rem;
    color: rgba(255,255,255,0.82);
    max-width: 640px;
}

/* ---------- Stat cards ---------- */
.stat-card {
    background: var(--card-bg);
    border: 1px solid var(--sage-border);
    border-radius: 12px;
    padding: 16px 18px;
    text-align: left;
}
.stat-card .stat-value {
    font-family: 'Lora', serif;
    font-weight: 700;
    font-size: 1.7rem;
    color: var(--forest-dark);
}
.stat-card .stat-label {
    font-size: 0.8rem;
    color: var(--ink-soft);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] section {
    background-color: var(--card-bg);
    border: 1.5px dashed var(--forest);
    border-radius: 10px;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background-color: var(--forest);
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.5rem 1.1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background-color: var(--forest-dark);
    color: #FFFFFF;
    transform: translateY(-1px);
}

/* ---------- Document library cards ---------- */
.doc-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: var(--card-bg);
    border: 1px solid var(--sage-border);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
    font-size: 0.88rem;
}
.doc-card .doc-icon { font-size: 1.2rem; }
.doc-card .doc-name { font-weight: 600; color: var(--ink); }
.doc-card .doc-meta { font-size: 0.76rem; color: var(--ink-soft); }

/* ---------- Chat bubbles with entrance animation ---------- */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
[data-testid="stChatMessage"] {
    background-color: var(--card-bg);
    border: 1px solid var(--sage-border);
    border-radius: 10px;
    padding: 4px 6px;
    margin-bottom: 10px;
    animation: fadeSlideIn 0.35s ease-out;
}

/* ---------- Source citation cards ---------- */
.source-card {
    background: var(--gold-light);
    border-left: 4px solid var(--gold);
    border-radius: 4px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.86rem;
    color: var(--ink-soft);
}
.source-card .source-label {
    font-family: 'Lora', serif;
    font-weight: 600;
    color: var(--forest-dark);
    font-size: 0.84rem;
    margin-bottom: 4px;
    display: block;
}

/* ---------- Status pill ---------- */
.index-pill {
    display: inline-block;
    background-color: var(--forest);
    color: white;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 500;
    margin-top: 6px;
}

/* ---------- Empty state ---------- */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--ink-soft);
}
.empty-state .icon { font-size: 2.6rem; margin-bottom: 10px; }

/* Tabs styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: var(--forest-dark) !important;
}

hr { border-color: var(--sage-border) !important; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ────────────────────────────────────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_vector_store()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_library" not in st.session_state:
    st.session_state.doc_library = []   # list of {"name":, "chunks":, "size_kb":}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

EXT_ICONS = {
    "pdf": "📕", "docx": "📘", "odt": "📗", "txt": "📄", "md": "📝",
    "xlsx": "📊", "csv": "📊", "pptx": "📙",
    "py": "🐍", "java": "☕", "cpp": "⚙️", "c": "⚙️",
    "html": "🌐", "css": "🎨",
}

# ────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="eyebrow">Retrieval-Augmented Generation</div>
    <div class="title">📖 Document Q&A</div>
    <div class="subtitle">Ask questions in plain language. Every answer is grounded in — and traceable to — the exact passages in your uploaded documents.</div>
</div>
""", unsafe_allow_html=True)

# Stat cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{len(st.session_state.doc_library)}</div>
    <div class="stat-label">Documents Indexed</div></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{st.session_state.total_chunks}</div>
    <div class="stat-label">Searchable Passages</div></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="stat-card"><div class="stat-value">{len(st.session_state.chat_history)//2}</div>
    <div class="stat-label">Questions Asked</div></div>""", unsafe_allow_html=True)

st.write("")

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR — document tray + library
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📥 Document Tray")
    st.caption("Upload source material for the assistant to read.")

    uploaded_files = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "odt", "txt", "md", "xlsx", "csv", "pptx",
          "py", "java", "cpp", "c", "html", "css"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

    if uploaded_files and st.button("📚 Process Documents", use_container_width=True):
        all_chunks = []
        progress_bar = st.progress(0, text="Starting...")
        new_docs = []
        failed_files = []

        for idx, file in enumerate(uploaded_files):
            progress_bar.progress(
                int((idx / len(uploaded_files)) * 60),
                text=f"Reading {file.name}...",
            )
            file_bytes = file.getvalue()

            if len(file_bytes) == 0:
                failed_files.append((file.name, "File is empty."))
                continue

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.name}") as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name
                raw_text = load_document(tmp_path)
                doc_chunks = split_text(raw_text, source_name=file.name)
                all_chunks.extend(doc_chunks)
                new_docs.append({
                    "name": file.name,
                    "chunks": len(doc_chunks),
                    "size_kb": round(len(file_bytes) / 1024, 1),
                    "ext": file.name.lower().split(".")[-1],
                })
            except ValueError as e:
                failed_files.append((file.name, str(e)))
            except Exception as e:
                failed_files.append((file.name, f"Unexpected error: {e}"))
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        progress_bar.empty()

        if failed_files:
            for name, reason in failed_files:
                st.warning(f"⚠️ Skipped **{name}** — {reason}")

        if all_chunks:
            progress_bar = st.progress(75, text="Embedding chunks & building FAISS index...")
            st.session_state.vector_store = build_vector_store(all_chunks)
            progress_bar.progress(100, text="Done!")
            time.sleep(0.4)
            progress_bar.empty()

            st.session_state.doc_library.extend(new_docs)
            st.session_state.total_chunks += len(all_chunks)

            st.markdown(
                f'<span class="index-pill">✓ Indexed {len(all_chunks)} passages from {len(new_docs)} file(s)</span>',
                unsafe_allow_html=True,
            )
            st.rerun()
        elif not failed_files:
            st.info("No content found to index.")

    st.markdown("---")

    if st.session_state.doc_library:
        st.markdown("### 📚 Library")
        for doc in st.session_state.doc_library:
            icon = EXT_ICONS.get(doc["ext"], "📄")
            st.markdown(f"""
            <div class="doc-card">
                <span class="doc-icon">{icon}</span>
                <div>
                    <div class="doc-name">{doc['name']}</div>
                    <div class="doc-meta">{doc['chunks']} passages · {doc['size_kb']} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    if st.session_state.chat_history and st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption("💡 Answers are generated only from what you upload here — nothing is invented outside the source text.")

# ────────────────────────────────────────────────────────────────────────────
# MAIN — tabs: Chat / About
# ────────────────────────────────────────────────────────────────────────────
tab_chat, tab_about = st.tabs(["💬 Chat", "ℹ️ How it works"])

with tab_chat:
    if st.session_state.vector_store is None:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📄</div>
            <div><strong>No documents indexed yet</strong></div>
            <div>Upload a file in the sidebar and click <em>Process Documents</em> to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            avatar = "📖" if msg["role"] == "assistant" else "🧑"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])

        query = st.chat_input("Ask a question about your documents...")
        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user", avatar="🧑"):
                st.write(query)

            with st.chat_message("assistant", avatar="📖"):
                with st.spinner("Reading through your documents..."):
                    retriever = get_retriever(st.session_state.vector_store)
                    qa_chain = build_qa_chain(retriever)
                    result = qa_chain.invoke({"query": query})
                    answer = result["result"]
                    sources = result["source_documents"]

                    st.write(answer)

                    with st.expander(f"📎 View {len(sources)} source passage(s)"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"""
                            <div class="source-card">
                                <span class="source-label">Source {i+1} — {doc.metadata.get('source')}</span>
                                {doc.page_content[:300]}...
                            </div>
                            """, unsafe_allow_html=True)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

with tab_about:
    st.markdown("""
    #### How this works

    This tool uses **Retrieval-Augmented Generation (RAG)** — instead of relying purely on what a language model
    memorized during training, it searches your actual uploaded documents for the most relevant passages, then
    asks the model to answer using only that retrieved context.

    **Pipeline:**
    1. 📄 **Load** — your PDF/DOCX/TXT is parsed into raw text
    2. ✂️ **Chunk** — text is split into overlapping passages (~500 tokens each)
    3. 🧠 **Embed** — each passage is converted into a vector using `all-MiniLM-L6-v2`
    4. 🔍 **Retrieve** — FAISS finds the passages most semantically similar to your question
    5. 💬 **Generate** — TinyLlama-1.1B-Chat writes an answer grounded in those passages

    Every answer includes its source passages so you can verify the response yourself.
    """)