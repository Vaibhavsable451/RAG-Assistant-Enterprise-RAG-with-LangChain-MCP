"""
MCP server: exposes the RAG pipeline (retrieve + generate) and AI Governance suite
as MCP tools, so any MCP-compatible client (Claude, subagents, IDEs) can query
the knowledge base and governance engines directly.

Run with:  python -m mcp.mcp_server
"""
import json
from mcp.server.fastmcp import FastMCP

from rag.vector_store import load_vector_store
from rag.retriever import get_relevant_chunks, format_context
from rag.generator import generate_answer
from governance.guardrails import check_input, audit_log
from governance.model_registry import list_models
from governance.risk_scoring import calculate_risk_score, RiskFactors
from governance.audit_compliance import verify_audit_integrity
from config import settings

mcp = FastMCP(settings.MCP_SERVER_NAME)


@mcp.tool()
def rag_query(question: str, top_k: int = 4) -> str:
    """
    Answer a question using the project's RAG knowledge base.
    Applies security scans, PII redaction, risk scoring, policy checks,
    and SHA-256 audit logging on every call.
    """
    gov_result = check_input(question, user_id="MCPClient")
    if not gov_result.allowed:
        return f"Query blocked by governance policy. Flags: {gov_result.flags}"

    if gov_result.requires_approval:
        return f"Query requires Human-in-the-Loop approval before execution. Request ID: {gov_result.approval_request_id}"

    store = load_vector_store()
    if store is None:
        return "No documents have been indexed yet. Ingest documents first."

    chunks = get_relevant_chunks(store, gov_result.redacted_text, k=top_k)
    context = format_context(chunks)
    answer = generate_answer(gov_result.redacted_text, context)

    audit_log("rag_query", gov_result.redacted_text, response=answer)
    return answer


@mcp.tool()
def list_registered_models() -> str:
    """List all AI models in the Model Registry with their lifecycle stage and approval status."""
    models = list_models()
    return json.dumps([m.to_dict() for m in models], indent=2)


@mcp.tool()
def assess_model_risk(
    bias: float = 2.0,
    security: float = 1.0,
    privacy: float = 1.0,
    hallucination_rate: float = 2.0,
) -> str:
    """Assess composite AI Risk Score (0-100) and risk level from governance factors."""
    factors = RiskFactors(bias=bias, security=security, privacy=privacy, hallucination_rate=hallucination_rate)
    res = calculate_risk_score(factors)
    return json.dumps(res.to_dict(), indent=2)


@mcp.tool()
def verify_audit_trail_integrity() -> str:
    """Verify the SHA-256 cryptographic chain of the immutable audit log."""
    valid, msg = verify_audit_integrity()
    return f"Status: {'VALID' if valid else 'CORRUPTED'}. Details: {msg}"


@mcp.tool()
def health_check() -> str:
    """Simple liveness check for the MCP server."""
    return "ok"


if __name__ == "__main__":
    mcp.run()
