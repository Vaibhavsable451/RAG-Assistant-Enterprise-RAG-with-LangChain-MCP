"""
AI Governance Guardrails Orchestrator.

Responsibilities:
1. AI Security: Prompt injection, jailbreak, and toxic content scanning.
2. PII Detection & Redaction (Presidio engine with fallback regex).
3. AI Risk Scoring (0–100 composite scoring across 7 factors).
4. Policy Engine enforcement (pre-deployment and runtime policy checks).
5. Cryptographically chained SHA-256 Immutable Audit Logging.
6. Human-in-the-Loop approval escalation.
"""
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple

from config import settings
from governance.ai_security import check_prompt_injection, check_unsafe_output, log_threat
from governance.risk_scoring import evaluate_query_risk, RiskAssessment
from governance.policy_engine import run_runtime_policy_checks, PolicyCheckResult
from governance.audit_compliance import log_immutable_audit
from governance.human_in_loop import submit_approval_request

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
except Exception:
    _PRESIDIO_AVAILABLE = False

_BLOCKED_TERMS = {"kill", "bomb", "hack into", "child abuse", "malware", "exploit"}


@dataclass
class GovernanceResult:
    allowed: bool
    redacted_text: str
    flags: List[str]
    risk_assessment: Optional[dict] = None
    policy_check: Optional[dict] = None
    requires_approval: bool = False
    approval_request_id: Optional[str] = None
    audit_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def redact_pii(text: str) -> Tuple[str, bool]:
    """Redact PII (emails, phone numbers, SSNs, etc.) if presidio is available."""
    if not settings.ENABLE_PII_REDACTION:
        return text, False
        
    if _PRESIDIO_AVAILABLE:
        try:
            results = _analyzer.analyze(text=text, language="en")
            if results:
                anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
                return anonymized.text, True
        except Exception:
            pass

    # Simple fallback regex patterns if Presidio is not available or errors
    import re
    redacted = text
    pii_found = False

    # Email
    if re.search(r"[\w\.-]+@[\w\.-]+\.\w+", redacted):
        redacted = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "<EMAIL_REDACTED>", redacted)
        pii_found = True
    # Phone
    if re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", redacted):
        redacted = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "<PHONE_REDACTED>", redacted)
        pii_found = True

    return redacted, pii_found


def moderate(text: str) -> List[str]:
    """Return a list of flags for content that violates basic policy."""
    if not settings.ENABLE_CONTENT_MODERATION:
        return []
    lowered = text.lower()
    return [term for term in _BLOCKED_TERMS if term in lowered]


def check_input(user_text: str, user_id: str = "AnonymousUser") -> GovernanceResult:
    """Unified entry point for governance validation on user input."""
    # 1. AI Security scan (prompt injection / jailbreak)
    threat_report = check_prompt_injection(user_text)
    
    # 2. Content moderation flags
    flags = moderate(user_text)
    if threat_report.threat_detected:
        flags.append(f"SECURITY_{threat_report.threat_type}")

    # 3. PII Detection & Redaction
    redacted_text, pii_found = redact_pii(user_text)

    # 4. AI Risk Scoring
    risk_assessment = evaluate_query_risk(user_text, flags=flags, pii_detected=pii_found)

    # 5. Policy Engine runtime verification
    policy_result = run_runtime_policy_checks(
        user_query=user_text,
        risk_score=risk_assessment.composite_score,
        security_threat=threat_report.threat_detected,
        pii_found=pii_found,
    )

    allowed = policy_result.enforced_action != "BLOCK" and len(flags) == 0
    requires_approval = policy_result.enforced_action == "REQUIRE_APPROVAL" or risk_assessment.requires_human_approval

    approval_req_id = None
    if requires_approval and allowed:
        req = submit_approval_request(
            request_type="HIGH_RISK_QUERY",
            requester=user_id,
            subject=user_text[:100],
            risk_score=risk_assessment.composite_score,
            payload={"query": user_text, "risk": risk_assessment.to_dict()},
        )
        approval_req_id = req.request_id

    # 6. SHA-256 Immutable Audit Log entry
    policy_decision = "BLOCK" if not allowed else ("REQUIRE_APPROVAL" if requires_approval else "ALLOW")
    audit_hash = log_immutable_audit(
        event_type="QUERY_EXECUTION",
        actor=user_id,
        model_version=settings.GROQ_MODEL,
        details={
            "query_length": len(user_text),
            "flags": flags,
            "risk_score": risk_assessment.composite_score,
            "risk_level": risk_assessment.level,
            "pii_redacted": pii_found,
            "threat_detected": threat_report.threat_detected,
        },
        policy_decision=policy_decision,
    )

    return GovernanceResult(
        allowed=allowed,
        redacted_text=redacted_text,
        flags=flags,
        risk_assessment=risk_assessment.to_dict(),
        policy_check=policy_result.to_dict(),
        requires_approval=requires_approval,
        approval_request_id=approval_req_id,
        audit_hash=audit_hash,
    )


def audit_log(event_type: str, query: str, response: str = "", flags: list = None):
    """Backward compatibility wrapper for legacy audit_log calls."""
    log_immutable_audit(
        event_type=event_type,
        actor="System",
        model_version=settings.GROQ_MODEL,
        details={"query": query[:100], "response_snippet": response[:100], "flags": flags or []},
    )
