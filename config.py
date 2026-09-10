"""
Central configuration for the RAG app.
All values are loaded from environment variables (.env) — never hardcode secrets.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store (local FAISS index on disk — intentionally no SQL server)
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./data/vector_store")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 120))

    # Governance
    ENABLE_PII_REDACTION: bool = os.getenv("ENABLE_PII_REDACTION", "true").lower() == "true"
    ENABLE_CONTENT_MODERATION: bool = os.getenv("ENABLE_CONTENT_MODERATION", "true").lower() == "true"
    AUDIT_LOG_PATH: str = os.getenv("AUDIT_LOG_PATH", "./data/audit_log.jsonl")

    # MCP
    MCP_SERVER_NAME: str = os.getenv("MCP_SERVER_NAME", "rag-governance-mcp")
    MCP_SERVER_PORT: int = int(os.getenv("MCP_SERVER_PORT", 8765))

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )


settings = Settings()
