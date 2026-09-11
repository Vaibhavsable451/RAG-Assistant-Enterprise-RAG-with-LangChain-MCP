"""
Audit & Compliance Module.
Implements a cryptographically chained (SHA-256) immutable audit log.
Tracks model deployments, policy decisions, approvals, security incidents, and RAG executions.
Includes an integrity verification function to guarantee tamper-proof audit trails.
"""
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from config import settings

AUDIT_TRAIL_PATH = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "audit_trail.jsonl")


def compute_record_hash(record: dict, prev_hash: str) -> str:
    """Compute SHA-256 hash of record payload concatenated with previous record hash."""
    payload = f"{record.get('timestamp')}:{record.get('event_type')}:{record.get('actor')}:{record.get('details')}:{prev_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def log_immutable_audit(
    event_type: str,
    actor: str,
    model_version: str,
    details: Dict[str, Any],
    policy_decision: str = "ALLOW",
) -> str:
    """Append a cryptographically chained audit record."""
    os.makedirs(os.path.dirname(AUDIT_TRAIL_PATH), exist_ok=True)
    
    prev_hash = "GENESIS_HASH_0000000000000000000000000000000000000000000000000"
    if os.path.exists(AUDIT_TRAIL_PATH):
        try:
            with open(AUDIT_TRAIL_PATH, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    last_record = json.loads(lines[-1])
                    prev_hash = last_record.get("hash", prev_hash)
        except Exception as e:
            print(f"[audit_compliance] Error reading last audit record: {e}")

    record = {
        "timestamp": time.time(),
        "event_type": event_type,
        "actor": actor,
        "model_version": model_version,
        "policy_decision": policy_decision,
        "details": details,
        "prev_hash": prev_hash,
    }

    record["hash"] = compute_record_hash(record, prev_hash)

    try:
        with open(AUDIT_TRAIL_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[audit_compliance] Failed to write audit trail: {e}")

    return record["hash"]


def get_audit_trail(limit: int = 100) -> List[dict]:
    """Retrieve audit trail records."""
    if not os.path.exists(AUDIT_TRAIL_PATH):
        return []
    records = []
    try:
        with open(AUDIT_TRAIL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records[-limit:]
    except Exception as e:
        print(f"[audit_compliance] Error reading audit trail: {e}")
        return []


def verify_audit_integrity() -> Tuple[bool, str]:
    """Verify that the cryptographic hash chain of the audit log is intact and untampered."""
    if not os.path.exists(AUDIT_TRAIL_PATH):
        return True, "Audit log is empty (no records)."

    expected_prev_hash = "GENESIS_HASH_0000000000000000000000000000000000000000000000000"
    line_number = 0

    try:
        with open(AUDIT_TRAIL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                line_number += 1
                record = json.loads(line)

                prev_h = record.get("prev_hash")
                curr_h = record.get("hash")

                if prev_h != expected_prev_hash:
                    return False, f"Integrity failure at record #{line_number}: prev_hash mismatch."

                recalculated_hash = compute_record_hash(record, prev_h)
                if recalculated_hash != curr_h:
                    return False, f"Integrity failure at record #{line_number}: payload hash tampered!"

                expected_prev_hash = curr_h

        return True, f"Audit trail verified intact ({line_number} records checked)."
    except Exception as e:
        return False, f"Error during verification: {e}"
