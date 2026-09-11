"""
AI Security Module.
Provides real-time protection against:
1. Prompt injection attacks
2. Jailbreak attempts
3. Unsafe output / toxicity
4. PII leakage
5. Threat logging & security analytics
"""
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from config import settings

THREAT_LOG_PATH = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "threat_logs.jsonl")

# Patterns commonly used in prompt injection & jailbreak exploits
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above)\s+instructions",
    r"disregard\s+(the\s+)?system\s+prompt",
    r"you\s+are\s+now\s+in\s+dan\s+mode",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"reveal\s+(your\s+)?system\s+(prompt|instructions)",
    r"bypass\s+governance",
    r"sudo\s+mode",
    r"act\s+as\s+an\s+unfiltered",
]

# Patterns for unsafe / malicious outputs
_UNSAFE_OUTPUT_PATTERNS = [
    r"malware",
    r"keylogger",
    r"exfiltrate",
    r"exploit\s+vulnerability",
    r"child\s+abuse",
    r"how\s+to\s+make\s+a\s+bomb",
]


@dataclass
class ThreatReport:
    threat_detected: bool
    threat_type: str         # PROMPT_INJECTION, JAILBREAK, UNSAFE_OUTPUT, PII_LEAKAGE, NONE
    severity: str            # LOW, MEDIUM, HIGH, CRITICAL
    matched_pattern: str
    redacted_input: str
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


def log_threat(threat_type: str, severity: str, matched_pattern: str, raw_input: str):
    """Log a detected security threat into the threat log."""
    try:
        os.makedirs(os.path.dirname(THREAT_LOG_PATH), exist_ok=True)
        record = {
            "timestamp": time.time(),
            "threat_type": threat_type,
            "severity": severity,
            "matched_pattern": matched_pattern,
            "snippet": raw_input[:200],
        }
        with open(THREAT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[ai_security] Failed to write threat log: {e}")


def load_threat_logs(limit: int = 50) -> List[dict]:
    """Retrieve recent threat logs."""
    if not os.path.exists(THREAT_LOG_PATH):
        return []
    logs = []
    try:
        with open(THREAT_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line))
        return logs[-limit:]
    except Exception as e:
        print(f"[ai_security] Failed to read threat logs: {e}")
        return []


def check_prompt_injection(user_input: str) -> ThreatReport:
    """Scan user input for prompt injection or jailbreak patterns."""
    lowered = user_input.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            threat_type = "JAILBREAK" if "dan" in pattern or "jailbreak" in pattern else "PROMPT_INJECTION"
            log_threat(threat_type, "CRITICAL", pattern, user_input)
            return ThreatReport(
                threat_detected=True,
                threat_type=threat_type,
                severity="CRITICAL",
                matched_pattern=pattern,
                redacted_input=user_input,
                timestamp=time.time(),
            )

    return ThreatReport(
        threat_detected=False,
        threat_type="NONE",
        severity="LOW",
        matched_pattern="",
        redacted_input=user_input,
        timestamp=time.time(),
    )


def check_unsafe_output(output_text: str) -> ThreatReport:
    """Scan generated model output for dangerous or policy-violating content."""
    lowered = output_text.lower()
    for pattern in _UNSAFE_OUTPUT_PATTERNS:
        if re.search(pattern, lowered):
            log_threat("UNSAFE_OUTPUT", "HIGH", pattern, output_text)
            return ThreatReport(
                threat_detected=True,
                threat_type="UNSAFE_OUTPUT",
                severity="HIGH",
                matched_pattern=pattern,
                redacted_input=output_text,
                timestamp=time.time(),
            )

    return ThreatReport(
        threat_detected=False,
        threat_type="NONE",
        severity="LOW",
        matched_pattern="",
        redacted_input=output_text,
        timestamp=time.time(),
    )
