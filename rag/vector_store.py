"""
Vector store: local, file-based FAISS index.
Deliberately avoids a SQL/managed-DB dependency — the index persists to disk
under settings.VECTOR_STORE_DIR and is loaded back on startup.
"""
import os
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag.embeddings import get_embeddings
from config import settings


def build_vector_store(chunks: list[Document]) -> FAISS:
    """Create a fresh FAISS index from document chunks and persist it to disk."""
    embeddings = get_embeddings()
    store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
    store.save_local(settings.VECTOR_STORE_DIR)
    return store


def load_vector_store() -> FAISS | None:
    """Load a previously persisted FAISS index, if one exists."""
    index_path = os.path.join(settings.VECTOR_STORE_DIR, "index.faiss")
    if not os.path.exists(index_path):
        return None
    try:
        embeddings = get_embeddings()
        return FAISS.load_local(
            settings.VECTOR_STORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"[vector_store] Error loading FAISS index: {e}")
        return None


def add_to_vector_store(store: FAISS, chunks: list[Document]) -> FAISS:
    store.add_documents(chunks)
    store.save_local(settings.VECTOR_STORE_DIR)
    return store


def clear_vector_store() -> bool:
    """Remove persisted vector store files from disk."""
    try:
        if os.path.exists(settings.VECTOR_STORE_DIR):
            import shutil
            shutil.rmtree(settings.VECTOR_STORE_DIR)
        return True
    except Exception as e:
        print(f"[vector_store] Error clearing vector store: {e}")
        return False

