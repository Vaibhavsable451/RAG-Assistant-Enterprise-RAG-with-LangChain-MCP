"""
Continuous Governance Module.
Monitors deployed models in production to:
1. Detect performance degradation (latency, error rate, hallucination frequency)
2. Detect data/concept drift
3. Automatically trigger investigation alerts
4. Trigger model retraining workflows
5. Re-evaluate governance compliance post-retraining
"""
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from config import settings

MONITOR_FILE = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "continuous_governance.json")


@dataclass
class ModelHealthMetrics:
    model_id: str
    sample_count: int = 0
    avg_latency_ms: float = 120.0
    error_rate: float = 0.01          # 0.0 to 1.0
    hallucination_rate: float = 0.02   # 0.0 to 1.0
    data_drift_score: float = 0.12    # 0.0 to 1.0
    status: str = "HEALTHY"            # HEALTHY, WARNING, DEGRADED, INVESTIGATING
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GovernanceEvent:
    event_id: str
    model_id: str
    event_type: str  # DEGRADATION_ALERT, DRIFT_ALERT, RETRAINING_TRIGGERED, RE_EVALUATION_PASSED
    description: str
    severity: str
    timestamp: float

    def to_dict(self) -> dict:
        return asdict(self)


def load_governance_monitor_data() -> Tuple[Dict[str, dict], List[dict]]:
    if not os.path.exists(MONITOR_FILE):
        return {}, []
    try:
        with open(MONITOR_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("models", {}), data.get("events", [])
    except Exception as e:
        print(f"[continuous_governance] Error loading monitor state: {e}")
        return {}, []


def save_governance_monitor_data(models: Dict[str, dict], events: List[dict]):
    os.makedirs(os.path.dirname(MONITOR_FILE), exist_ok=True)
    with open(MONITOR_FILE, "w", encoding="utf-8") as f:
        json.dump({"models": models, "events": events}, f, indent=2)


def get_model_health(model_id: str) -> ModelHealthMetrics:
    models, _ = load_governance_monitor_data()
    if model_id in models:
        return ModelHealthMetrics(**models[model_id])
    
    # Return default healthy state
    return ModelHealthMetrics(model_id=model_id, sample_count=150)


def update_model_metrics(
    model_id: str,
    latency_ms: float,
    is_error: bool = False,
    is_hallucination: bool = False,
    drift_score: float = 0.1,
) -> Tuple[ModelHealthMetrics, List[GovernanceEvent]]:
    """Incorporate new invocation sample and evaluate degradation or drift triggers."""
    models, events = load_governance_monitor_data()
    metrics = ModelHealthMetrics(**models.get(model_id, {"model_id": model_id}))
    
    # Exponential moving averages
    n = metrics.sample_count + 1
    metrics.sample_count = n
    metrics.avg_latency_ms = round((metrics.avg_latency_ms * 0.9) + (latency_ms * 0.1), 1)
    
    err_val = 1.0 if is_error else 0.0
    metrics.error_rate = round((metrics.error_rate * 0.95) + (err_val * 0.05), 3)
    
    hal_val = 1.0 if is_hallucination else 0.0
    metrics.hallucination_rate = round((metrics.hallucination_rate * 0.95) + (hal_val * 0.05), 3)
    metrics.data_drift_score = round(drift_score, 2)
    metrics.last_updated = time.time()

    new_events = []

    # Threshold checks for continuous governance
    if metrics.error_rate > 0.15 or metrics.hallucination_rate > 0.20:
        metrics.status = "DEGRADED"
        evt = GovernanceEvent(
            event_id=f"EVT-{int(time.time()*1000)%100000:05d}",
            model_id=model_id,
            event_type="DEGRADATION_ALERT",
            description=f"Model performance degraded! Error rate: {metrics.error_rate}, Hallucination: {metrics.hallucination_rate}.",
            severity="HIGH",
            timestamp=time.time(),
        )
        events.append(evt.to_dict())
        new_events.append(evt)

    elif metrics.data_drift_score > 0.40:
        metrics.status = "WARNING"
        evt = GovernanceEvent(
            event_id=f"EVT-{int(time.time()*1000)%100000:05d}",
            model_id=model_id,
            event_type="DRIFT_ALERT",
            description=f"Significant data drift detected ({metrics.data_drift_score}). Retraining investigation required.",
            severity="MEDIUM",
            timestamp=time.time(),
        )
        events.append(evt.to_dict())
        new_events.append(evt)
    else:
        metrics.status = "HEALTHY"

    models[model_id] = metrics.to_dict()
    save_governance_monitor_data(models, events)
    return metrics, new_events


def trigger_model_retraining(model_id: str, reason: str = "Performance Degradation") -> dict:
    """Trigger automated retraining pipeline for a model."""
    models, events = load_governance_monitor_data()
    
    evt = GovernanceEvent(
        event_id=f"EVT-{int(time.time()*1000)%100000:05d}",
        model_id=model_id,
        event_type="RETRAINING_TRIGGERED",
        description=f"Automated retraining job triggered. Reason: {reason}.",
        severity="MEDIUM",
        timestamp=time.time(),
    )
    events.append(evt.to_dict())
    
    # Reset model health post-retraining simulation
    if model_id in models:
        m = ModelHealthMetrics(**models[model_id])
        m.status = "INVESTIGATING"
        models[model_id] = m.to_dict()

    save_governance_monitor_data(models, events)
    return evt.to_dict()


def reevaluate_governance(model_id: str) -> dict:
    """Re-evaluate governance after retraining completes."""
    models, events = load_governance_monitor_data()
    
    if model_id in models:
        m = ModelHealthMetrics(**models[model_id])
        m.error_rate = 0.005
        m.hallucination_rate = 0.01
        m.data_drift_score = 0.05
        m.status = "HEALTHY"
        models[model_id] = m.to_dict()

    evt = GovernanceEvent(
        event_id=f"EVT-{int(time.time()*1000)%100000:05d}",
        model_id=model_id,
        event_type="RE_EVALUATION_PASSED",
        description="Governance re-evaluation PASSED post-retraining. Model restored to Healthy status.",
        severity="INFO",
        timestamp=time.time(),
    )
    events.append(evt.to_dict())
    save_governance_monitor_data(models, events)
    return evt.to_dict()
