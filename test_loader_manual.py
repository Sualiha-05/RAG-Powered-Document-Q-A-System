# test_loader_manual.py
from app.ingestion.document_loader import load_document

text = load_document("data/raw/LCTNotes.pdf")  # put any PDF here
print("Length of extracted text:", len(text))
print("First 300 chars:\n", text[:300])