"""
Retriever: wraps the FAISS store's similarity search behind a simple interface.
"""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def get_relevant_chunks(store: FAISS, query: str, k: int = 4) -> list[Document]:
    """Return the top-k most similar chunks to the query."""
    return store.similarity_search(query, k=k)


def format_context(chunks: list[Document]) -> str:
    """Turn retrieved chunks into a single context block for the LLM prompt,
    with source attribution for traceability (governance requirement)."""
    parts = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        chunk_id = c.metadata.get("chunk_id", "?")
        parts.append(f"[source: {source} | chunk: {chunk_id}]\n{c.page_content}")
    return "\n\n---\n\n".join(parts)
