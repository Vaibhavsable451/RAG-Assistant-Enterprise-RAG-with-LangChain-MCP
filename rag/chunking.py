"""
Chunking: splits raw documents into overlapping text chunks before embedding.
Uses LangChain's recursive character splitter, tuned via config.py.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split a list of LangChain Documents into smaller overlapping chunks."""
    splitter = get_splitter()
    chunks = splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def chunk_text(text: str, source: str = "unknown") -> list[Document]:
    """Convenience wrapper for chunking a raw string instead of loaded Documents."""
    doc = Document(page_content=text, metadata={"source": source})
    return chunk_documents([doc])
