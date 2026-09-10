"""
MCP server: exposes the RAG pipeline (retrieve + generate) as an MCP tool,
so any MCP-compatible client (Claude, other agents, IDEs) can query this
knowledge base directly, with governance checks applied on every call.

Run with:  python -m mcp.mcp_server
"""
from mcp.server.fastmcp import FastMCP

from rag.vector_store import load_vector_store
from rag.retriever import get_relevant_chunks, format_context
from rag.generator import generate_answer
from governance.guardrails import check_input, audit_log
from config import settings

mcp = FastMCP(settings.MCP_SERVER_NAME)


@mcp.tool()
def rag_query(question: str, top_k: int = 4) -> str:
    """
    Answer a question using the project's RAG knowledge base.
    Applies PII redaction and content moderation before/after the LLM call,
    and writes every call to the audit log.
    """
    gov_result = check_input(question)
    if not gov_result.allowed:
        audit_log("blocked_query", question, flags=gov_result.flags)
        return f"Query blocked by governance policy. Flags: {gov_result.flags}"

    store = load_vector_store()
    if store is None:
        return "No documents have been indexed yet. Ingest documents first."

    chunks = get_relevant_chunks(store, gov_result.redacted_text, k=top_k)
    context = format_context(chunks)
    answer = generate_answer(gov_result.redacted_text, context)

    audit_log("rag_query", gov_result.redacted_text, response=answer)
    return answer


@mcp.tool()
def health_check() -> str:
    """Simple liveness check for the MCP server."""
    return "ok"


if __name__ == "__main__":
    mcp.run()
