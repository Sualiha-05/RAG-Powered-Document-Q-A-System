import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector store
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "..", "data", "faiss_index")

# LLM
LLM_MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "tinyllama-1.1b-chat-q4_k_m.gguf")
LLM_N_CTX = 2048
LLM_N_THREADS = 4
LLM_MAX_TOKENS = 256
LLM_TEMPERATURE = 0.3

# Retrieval
TOP_K = 4