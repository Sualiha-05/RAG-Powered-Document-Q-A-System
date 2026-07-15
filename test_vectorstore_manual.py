# test_vectorstore_manual.py
from app.ingestion.document_loader import load_document
from app.processing.text_splitter import split_text
from app.retrieval.vector_store import build_vector_store, get_retriever

text = load_document("data/raw/LCTNotes.pdf")
chunks = split_text(text, source_name="your_test_file.pdf")

print("Building vector store (this downloads the embedding model on first run)...")
vector_store = build_vector_store(chunks)
print("Vector store built and saved.")

retriever = get_retriever(vector_store)
query = "What is logic used for in politics?"
results = retriever.get_relevant_documents(query)

print(f"\nQuery: {query}")
print(f"Retrieved {len(results)} chunks:\n")
for i, doc in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(doc.page_content[:200])
    print()