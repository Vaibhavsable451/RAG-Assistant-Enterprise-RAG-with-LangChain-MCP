"""
AI Governance layer.

Responsibilities:
1. PII detection/redaction on user input before it reaches the LLM or logs.
2. Lightweight content moderation on both input and output.
3. Append-only audit logging of every query/response pair for traceability.

This is intentionally a thin, swappable layer — in production you'd likely
back this with your org's real policy engine, but the interface stays the same.
"""
import json
import os
import time
from dataclasses import dataclass, asdict
from config import settings

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _PRESIDIO_AVAILABLE = True
except Exception:
    _PRESIDIO_AVAILABLE = False

# Minimal denylist fallback moderation — replace with a real moderation
# model/service for production use.
_BLOCKED_TERMS = {"kill", "bomb", "hack into", "child abuse"}


@dataclass
class GovernanceResult:
    allowed: bool
    redacted_text: str
    flags: list


def redact_pii(text: str) -> str:
    """Redact PII (emails, phone numbers, names, etc.) if presidio is available."""
    if not settings.ENABLE_PII_REDACTION or not _PRESIDIO_AVAILABLE:
        return text
    results = _analyzer.analyze(text=text, language="en")
    if not results:
        return text
    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    return anonymized.text


def moderate(text: str) -> list:
    """Return a list of flags for content that violates basic policy."""
    if not settings.ENABLE_CONTENT_MODERATION:
        return []
    lowered = text.lower()
    return [term for term in _BLOCKED_TERMS if term in lowered]


def check_input(user_text: str) -> GovernanceResult:
    flags = moderate(user_text)
    redacted = redact_pii(user_text)
    allowed = len(flags) == 0
    return GovernanceResult(allowed=allowed, redacted_text=redacted, flags=flags)


def audit_log(event_type: str, query: str, response: str = "", flags: list = None):
    """Append a JSON line audit record. Never blocks the main flow on failure."""
    try:
        os.makedirs(os.path.dirname(settings.AUDIT_LOG_PATH), exist_ok=True)
        record = {
            "timestamp": time.time(),
            "event_type": event_type,
            "query": query,
            "response": response,
            "flags": flags or [],
        }
        with open(settings.AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[governance] audit log write failed: {e}")
