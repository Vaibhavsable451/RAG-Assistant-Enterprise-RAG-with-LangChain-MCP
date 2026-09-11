"""
Policy Engine.
Defines organization-wide AI policies, runs pre-deployment verification,
enforces runtime policy checks, and automatically blocks non-compliant models/queries.
"""
import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from config import settings
from governance.model_registry import ModelRecord, LifecycleStage

POLICIES_FILE = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "policies.json")


@dataclass
class PolicyRule:
    rule_id: str
    name: str
    category: str  # Pre-deployment, Runtime, Data, Security
    description: str
    enabled: bool = True
    action: str = "BLOCK"  # BLOCK, FLAG, REQUIRE_APPROVAL
    threshold: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyCheckResult:
    passed: bool
    violations: List[str]
    warnings: List[str]
    enforced_action: str  # ALLOW, BLOCK, REQUIRE_APPROVAL

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_POLICIES = [
    PolicyRule(
        rule_id="POL-001",
        name="Production Model Approval",
        category="Pre-deployment",
        description="Only models with Approved status and in Production lifecycle stage can serve user traffic.",
        enabled=True,
        action="BLOCK",
    ),
    PolicyRule(
        rule_id="POL-002",
        name="Maximum Risk Score Limit",
        category="Pre-deployment",
        description="Models with Risk Score above 75.0 (Critical) are blocked from deployment.",
        enabled=True,
        action="BLOCK",
        threshold=75.0,
    ),
    PolicyRule(
        rule_id="POL-003",
        name="Mandatory PII Redaction",
        category="Runtime",
        description="All user queries containing sensitive PII must be redacted before sending to LLM.",
        enabled=True,
        action="FLAG",
    ),
    PolicyRule(
        rule_id="POL-004",
        name="Prompt Injection Prevention",
        category="Security",
        description="Queries containing prompt injection or jailbreak patterns are automatically blocked.",
        enabled=True,
        action="BLOCK",
    ),
    PolicyRule(
        rule_id="POL-005",
        name="High-Risk Approval Requirement",
        category="Runtime",
        description="Queries evaluated with Risk Score >= 60.0 require Human-in-the-Loop approval.",
        enabled=True,
        action="REQUIRE_APPROVAL",
        threshold=60.0,
    ),
]


def load_policies() -> List[PolicyRule]:
    if not os.path.exists(POLICIES_FILE):
        save_policies(DEFAULT_POLICIES)
        return DEFAULT_POLICIES
    try:
        with open(POLICIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [PolicyRule(**item) for item in data]
    except Exception as e:
        print(f"[policy_engine] Error loading policies: {e}")
        return DEFAULT_POLICIES


def save_policies(policies: List[PolicyRule]):
    os.makedirs(os.path.dirname(POLICIES_FILE), exist_ok=True)
    with open(POLICIES_FILE, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in policies], f, indent=2)


def run_pre_deployment_policy_checks(model: ModelRecord) -> PolicyCheckResult:
    """Run compliance checks before deploying or activating a model."""
    policies = load_policies()
    violations = []
    warnings = []

    for rule in policies:
        if not rule.enabled or rule.category != "Pre-deployment":
            continue

        if rule.rule_id == "POL-001":
            if model.approval_status != "Approved":
                violations.append(f"[{rule.rule_id}] Model '{model.name}' approval status is '{model.approval_status}' (must be Approved).")
            if model.lifecycle_stage not in {LifecycleStage.APPROVED, LifecycleStage.PRODUCTION}:
                violations.append(f"[{rule.rule_id}] Model lifecycle stage is '{model.lifecycle_stage}' (must be Approved or Production).")

        elif rule.rule_id == "POL-002":
            if model.risk_score > rule.threshold:
                violations.append(f"[{rule.rule_id}] Model risk score {model.risk_score} exceeds maximum policy threshold {rule.threshold}.")

    passed = len(violations) == 0
    enforced_action = "ALLOW" if passed else "BLOCK"
    return PolicyCheckResult(
        passed=passed,
        violations=violations,
        warnings=warnings,
        enforced_action=enforced_action,
    )


def run_runtime_policy_checks(
    user_query: str,
    risk_score: float,
    security_threat: bool = False,
    pii_found: bool = False,
) -> PolicyCheckResult:
    """Run compliance checks on active query execution."""
    policies = load_policies()
    violations = []
    warnings = []
    enforced_action = "ALLOW"

    for rule in policies:
        if not rule.enabled:
            continue

        if rule.rule_id == "POL-004" and security_threat:
            violations.append(f"[{rule.rule_id}] Security threat detected: prompt injection or unsafe content.")
            enforced_action = "BLOCK"

        elif rule.rule_id == "POL-005" and risk_score >= rule.threshold and enforced_action != "BLOCK":
            warnings.append(f"[{rule.rule_id}] Query risk score {risk_score} requires Human-in-the-Loop review.")
            enforced_action = "REQUIRE_APPROVAL"

        elif rule.rule_id == "POL-003" and pii_found:
            warnings.append(f"[{rule.rule_id}] PII detected and sanitized under mandatory redaction policy.")

    passed = len(violations) == 0
    return PolicyCheckResult(
        passed=passed,
        violations=violations,
        warnings=warnings,
        enforced_action=enforced_action if not passed else enforced_action,
    )
