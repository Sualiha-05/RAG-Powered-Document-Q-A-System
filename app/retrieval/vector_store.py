import os
from langchain_community.vectorstores import FAISS
from app.embeddings.embedder import get_embedder
from app.config import FAISS_INDEX_PATH, TOP_K

def build_vector_store(documents):
    embedder = get_embedder()
    vector_store = FAISS.from_documents(documents, embedder)
    vector_store.save_local(FAISS_INDEX_PATH)
    return vector_store

def load_vector_store():
    if not os.path.exists(FAISS_INDEX_PATH):
        return None
    embedder = get_embedder()
    return FAISS.load_local(
        FAISS_INDEX_PATH, embedder, allow_dangerous_deserialization=True
    )

def get_retriever(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": TOP_K})