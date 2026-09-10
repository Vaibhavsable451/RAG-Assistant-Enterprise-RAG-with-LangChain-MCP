"""
Embeddings: wraps a local sentence-transformers model via LangChain's HuggingFaceEmbeddings.
No external embedding API call is required (keeps cost down, works offline once model is cached).
"""
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import settings

_embeddings_instance = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton accessor so the model is only loaded into memory once."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_instance
