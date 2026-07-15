# test_splitter_manual.py
from app.ingestion.document_loader import load_document
from app.processing.text_splitter import split_text

text = load_document("data/raw/LCTNotes.pdf")  # same PDF as before
chunks = split_text(text, source_name="your_test_file.pdf")

print(f"Total chunks: {len(chunks)}")
print("\n--- First chunk ---")
print(chunks[0].page_content)
print("\nMetadata:", chunks[0].metadata)

print("\n--- Second chunk ---")
print(chunks[1].page_content)