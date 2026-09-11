"""
Model Registry & Governance.
Manages model metadata, owners, versioning, training dataset lineage,
approval workflow, deployment history, and lifecycle states:
Development → Testing → Approved → Production → Retired.
"""
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from config import settings

REGISTRY_FILE = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "model_registry.json")


class LifecycleStage:
    DEVELOPMENT = "Development"
    TESTING = "Testing"
    APPROVED = "Approved"
    PRODUCTION = "Production"
    RETIRED = "Retired"

    VALID_STAGES = {DEVELOPMENT, TESTING, APPROVED, PRODUCTION, RETIRED}

    VALID_TRANSITIONS = {
        DEVELOPMENT: {TESTING, RETIRED},
        TESTING: {APPROVED, DEVELOPMENT, RETIRED},
        APPROVED: {PRODUCTION, RETIRED},
        PRODUCTION: {RETIRED},
        RETIRED: {DEVELOPMENT}
    }


@dataclass
class DeploymentRecord:
    timestamp: float
    deployed_by: str
    environment: str
    status: str
    notes: str = ""


@dataclass
class ModelRecord:
    model_id: str
    name: str
    version: str
    owner_team: str
    training_dataset: str
    dataset_version: str
    lifecycle_stage: str = LifecycleStage.DEVELOPMENT
    approval_status: str = "Pending"  # Pending, Approved, Rejected
    risk_score: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deployment_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelRecord":
        return cls(**data)


def _load_registry() -> Dict[str, dict]:
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[model_registry] Failed to load registry: {e}")
        return {}


def _save_registry(registry: Dict[str, dict]):
    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def register_model(
    model_id: str,
    name: str,
    version: str,
    owner_team: str,
    training_dataset: str,
    dataset_version: str,
    metadata: Optional[dict] = None,
) -> ModelRecord:
    """Register a new model in Development stage."""
    registry = _load_registry()
    record = ModelRecord(
        model_id=model_id,
        name=name,
        version=version,
        owner_team=owner_team,
        training_dataset=training_dataset,
        dataset_version=dataset_version,
        lifecycle_stage=LifecycleStage.DEVELOPMENT,
        approval_status="Pending",
        created_at=time.time(),
        updated_at=time.time(),
        metadata=metadata or {},
    )
    registry[model_id] = record.to_dict()
    _save_registry(registry)
    return record


def get_model(model_id: str) -> Optional[ModelRecord]:
    registry = _load_registry()
    if model_id in registry:
        return ModelRecord.from_dict(registry[model_id])
    return None


def list_models() -> List[ModelRecord]:
    registry = _load_registry()
    models = [ModelRecord.from_dict(v) for v in registry.values()]
    # Ensure default Groq active model is listed if registry is empty
    if not models:
        default_model = ModelRecord(
            model_id="groq-llama-3.3-70b",
            name=f"Groq {settings.GROQ_MODEL}",
            version="v1.0.0",
            owner_team="AI Platform Team",
            training_dataset="Corporate Knowledgebase RAG",
            dataset_version="v2026.1",
            lifecycle_stage=LifecycleStage.PRODUCTION,
            approval_status="Approved",
            risk_score=15.0,
            deployment_history=[{
                "timestamp": time.time(),
                "deployed_by": "CI/CD Automation",
                "environment": "Azure App Service",
                "status": "Success",
                "notes": "Production deployment via main branch"
            }],
        )
        registry[default_model.model_id] = default_model.to_dict()
        _save_registry(registry)
        return [default_model]
    return models


def transition_lifecycle(model_id: str, new_stage: str, updated_by: str = "System") -> bool:
    """Transition a model to a new lifecycle stage following valid state rules."""
    registry = _load_registry()
    if model_id not in registry:
        return False
    record = ModelRecord.from_dict(registry[model_id])
    current_stage = record.lifecycle_stage

    if new_stage not in LifecycleStage.VALID_STAGES:
        raise ValueError(f"Invalid stage '{new_stage}'. Must be one of {LifecycleStage.VALID_STAGES}")

    allowed = LifecycleStage.VALID_TRANSITIONS.get(current_stage, set())
    if new_stage not in allowed and new_stage != current_stage:
        raise ValueError(f"Invalid transition from '{current_stage}' to '{new_stage}'. Allowed: {allowed}")

    record.lifecycle_stage = new_stage
    record.updated_at = time.time()
    if new_stage == LifecycleStage.APPROVED:
        record.approval_status = "Approved"
    elif new_stage == LifecycleStage.PRODUCTION:
        record.deployment_history.append({
            "timestamp": time.time(),
            "deployed_by": updated_by,
            "environment": "Production",
            "status": "Active",
            "notes": f"Promoted to {new_stage}"
        })

    registry[model_id] = record.to_dict()
    _save_registry(registry)
    return True


def get_active_production_model() -> ModelRecord:
    models = list_models()
    for m in models:
        if m.lifecycle_stage == LifecycleStage.PRODUCTION:
            return m
    return models[0]
