from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(raw_text: str, source_name: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    return [
        Document(page_content=chunk, metadata={"source": source_name, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]