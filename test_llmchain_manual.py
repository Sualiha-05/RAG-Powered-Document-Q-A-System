# test_llmchain_manual.py
from app.ingestion.document_loader import load_document
from app.processing.text_splitter import split_text
from app.retrieval.vector_store import build_vector_store, get_retriever
from app.generation.llm_chain import build_qa_chain

text = load_document("data/raw/LCTNotes.pdf")
chunks = split_text(text, source_name="LCTNotes.pdf")

print("Building vector store...")
vector_store = build_vector_store(chunks)
retriever = get_retriever(vector_store)

print("Loading TinyLlama (this may take a moment)...")
qa_chain = build_qa_chain(retriever)

query = "What is logic used for in politics?"
print(f"\nQuery: {query}")
result = qa_chain.invoke({"query": query})

print("\n--- Answer ---")
print(result["result"])

print("\n--- Sources used ---")
for doc in result["source_documents"]:
    print("-", doc.metadata)