"""
AI Risk Scoring Engine.
Evaluates model and query risk scores (0–100) and risk levels (Low, Medium, High, Critical)
based on 7 core governance factors:
1. Bias
2. Data quality
3. Security
4. Privacy
5. Hallucination / error rate
6. Explainability
7. Model performance
"""
from dataclasses import dataclass, asdict
from typing import Dict, Any


class RiskLevel:
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


@dataclass
class RiskFactors:
    bias: float = 2.0               # 0 (Low risk) to 10 (High risk)
    data_quality: float = 2.0       # 0 (High quality/Low risk) to 10 (Poor quality)
    security: float = 1.0           # 0 (Secure) to 10 (Severe threat)
    privacy: float = 1.0            # 0 (No PII) to 10 (Sensitive PII)
    hallucination_rate: float = 2.0 # 0 (Low error) to 10 (High error)
    explainability: float = 2.0     # 0 (Highly explainable) to 10 (Black box)
    performance: float = 2.0        # 0 (Optimal) to 10 (Degraded)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskAssessment:
    composite_score: float         # 0 - 100
    level: str                     # Low, Medium, High, Critical
    factors: RiskFactors
    requires_human_approval: bool
    summary: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = self.factors.to_dict()
        return d


# Factor weights summing to 1.0
FACTOR_WEIGHTS = {
    "security": 0.20,
    "privacy": 0.20,
    "bias": 0.15,
    "hallucination_rate": 0.15,
    "data_quality": 0.10,
    "explainability": 0.10,
    "performance": 0.10,
}


def calculate_risk_score(factors: RiskFactors) -> RiskAssessment:
    """Calculate composite 0–100 risk score and level from factors."""
    factor_dict = factors.to_dict()
    
    # Weighted average (each factor is 0-10, composite is scaled to 0-100)
    weighted_sum = sum(factor_dict[k] * FACTOR_WEIGHTS[k] for kk in FACTOR_WEIGHTS for k in [kk])
    composite_score = round(min(100.0, max(0.0, weighted_sum * 10.0)), 1)

    if composite_score <= 25.0:
        level = RiskLevel.LOW
    elif composite_score <= 50.0:
        level = RiskLevel.MEDIUM
    elif composite_score <= 75.0:
        level = RiskLevel.HIGH
    else:
        level = RiskLevel.CRITICAL

    requires_approval = composite_score >= 60.0

    summary = (
        f"Risk Score: {composite_score}/100 [{level}]. "
        f"Security: {factors.security}/10, Privacy: {factors.privacy}/10, "
        f"Bias: {factors.bias}/10, Hallucination: {factors.hallucination_rate}/10."
    )

    return RiskAssessment(
        composite_score=composite_score,
        level=level,
        factors=factors,
        requires_human_approval=requires_approval,
        summary=summary,
    )


def evaluate_query_risk(user_query: str, flags: list = None, pii_detected: bool = False) -> RiskAssessment:
    """Evaluate runtime risk score for an incoming query."""
    flags = flags or []
    sec_val = min(10.0, 2.0 + len(flags) * 3.5)
    priv_val = 8.0 if pii_detected else 1.0
    bias_val = 6.0 if any(term in user_query.lower() for term in ["gender", "race", "stereotype"]) else 2.0
    
    factors = RiskFactors(
        security=sec_val,
        privacy=priv_val,
        bias=bias_val,
        data_quality=2.0,
        hallucination_rate=2.0,
        explainability=2.0,
        performance=1.0,
    )
    return calculate_risk_score(factors)
