"""
Human-in-the-Loop Governance Module.
Manages approval queues for:
1. High-risk prediction requests (Risk Score >= 60)
2. Manual model deployment/promotion approvals
3. Human override decisions and escalation tracking
"""
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from config import settings

APPROVALS_FILE = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "approvals.json")


class ApprovalStatus:
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ESCALATED = "ESCALATED"


@dataclass
class ApprovalRequest:
    request_id: str
    request_type: str       # HIGH_RISK_QUERY, MODEL_PROMOTION, POLICY_OVERRIDE
    requester: str
    subject: str
    risk_score: float
    status: str = ApprovalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    reviewed_at: Optional[float] = None
    reviewer: Optional[str] = None
    review_notes: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def load_approval_requests() -> List[ApprovalRequest]:
    if not os.path.exists(APPROVALS_FILE):
        return []
    try:
        with open(APPROVALS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ApprovalRequest(**item) for item in data]
    except Exception as e:
        print(f"[human_in_loop] Error loading approvals: {e}")
        return []


def save_approval_requests(requests: List[ApprovalRequest]):
    os.makedirs(os.path.dirname(APPROVALS_FILE), exist_ok=True)
    with open(APPROVALS_FILE, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in requests], f, indent=2)


def submit_approval_request(
    request_type: str,
    requester: str,
    subject: str,
    risk_score: float,
    payload: Optional[dict] = None,
) -> ApprovalRequest:
    """Submit a request for Human-in-the-Loop review."""
    requests = load_approval_requests()
    req_id = f"REQ-{int(time.time()*1000)%1000000:06d}"
    req = ApprovalRequest(
        request_id=req_id,
        request_type=request_type,
        requester=requester,
        subject=subject,
        risk_score=risk_score,
        status=ApprovalStatus.PENDING,
        created_at=time.time(),
        payload=payload or {},
    )
    requests.append(req)
    save_approval_requests(requests)
    return req


def review_request(
    request_id: str,
    reviewer: str,
    decision: str,  # APPROVED, REJECTED, ESCALATED
    notes: str = "",
) -> bool:
    """Review and act on a pending approval request."""
    requests = load_approval_requests()
    for req in requests:
        if req.request_id == request_id:
            req.status = decision
            req.reviewer = reviewer
            req.reviewed_at = time.time()
            req.review_notes = notes
            save_approval_requests(requests)
            return True
    return False


def get_pending_approvals() -> List[ApprovalRequest]:
    requests = load_approval_requests()
    return [r for r in requests if r.status == ApprovalStatus.PENDING]
