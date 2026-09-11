"""
Data Governance & Lineage Module.
Manages dataset lineage, dataset version tracking, data quality validation,
sensitive field classification (PUBLIC, CONFIDENTIAL, RESTRICTED, SECRET), and data drift monitoring.
"""
import json
import os
import time
import math
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from config import settings

LINEAGE_FILE = os.path.join(os.path.dirname(settings.AUDIT_LOG_PATH), "data_lineage.json")


class DataClassification:
    PUBLIC = "PUBLIC"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"
    SECRET = "SECRET"


@dataclass
class DatasetVersionRecord:
    dataset_id: str
    version_tag: str
    source_files: List[str]
    total_chunks: int
    data_quality_score: float  # 0.0 to 100.0
    classification: str        # PUBLIC, CONFIDENTIAL, RESTRICTED, SECRET
    pii_count_detected: int
    created_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DataDriftReport:
    drift_detected: bool
    drift_score: float  # 0.0 (No drift) to 1.0 (Severe drift)
    baseline_version: str
    current_version: str
    metrics: Dict[str, float]
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


def load_lineage_records() -> List[DatasetVersionRecord]:
    if not os.path.exists(LINEAGE_FILE):
        return []
    try:
        with open(LINEAGE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [DatasetVersionRecord(**item) for item in data]
    except Exception as e:
        print(f"[data_governance] Failed to load lineage: {e}")
        return []


def save_lineage_records(records: List[DatasetVersionRecord]):
    os.makedirs(os.path.dirname(LINEAGE_FILE), exist_ok=True)
    with open(LINEAGE_FILE, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def run_data_quality_check(chunks: list) -> Dict[str, Any]:
    """Perform data quality checks on indexed document chunks."""
    if not chunks:
        return {
            "quality_score": 0.0,
            "avg_length": 0,
            "empty_chunks": 0,
            "duplicate_chunks": 0,
            "passed": False,
        }

    texts = [c.page_content for c in chunks if hasattr(c, "page_content")]
    empty_count = sum(1 for t in texts if not t.strip())
    unique_count = len(set(texts))
    duplicate_count = len(texts) - unique_count
    avg_len = sum(len(t) for t in texts) / max(1, len(texts))

    # Calculate quality score out of 100
    quality = 100.0
    quality -= (empty_count / max(1, len(texts))) * 50.0
    quality -= (duplicate_count / max(1, len(texts))) * 30.0
    if avg_len < 50:
        quality -= 20.0
    
    quality_score = round(max(0.0, quality), 1)
    passed = quality_score >= 70.0

    return {
        "quality_score": quality_score,
        "avg_length": round(avg_len, 1),
        "empty_chunks": empty_count,
        "duplicate_chunks": duplicate_count,
        "total_chunks": len(texts),
        "passed": passed,
    }


def classify_sensitive_fields(text: str) -> str:
    """Classify data security level based on sensitive keywords and patterns."""
    lowered = text.lower()
    if any(k in lowered for k in ["top secret", "password", "api_key", "private_key", "ssn"]):
        return DataClassification.SECRET
    elif any(k in lowered for k in ["confidential", "salary", "medical", "financial report", "tax"]):
        return DataClassification.RESTRICTED
    elif any(k in lowered for k in ["internal only", "draft", "strategy", "proprietary"]):
        return DataClassification.CONFIDENTIAL
    return DataClassification.PUBLIC


def record_dataset_ingestion(
    source_files: List[str],
    chunks: list,
    pii_count: int = 0,
) -> DatasetVersionRecord:
    """Record dataset lineage and versioning when documents are ingested."""
    records = load_lineage_records()
    version_num = len(records) + 1
    version_tag = f"v2026.09.{version_num:02d}"

    quality_res = run_data_quality_check(chunks)
    
    # Classify sensitivity based on content sample
    sample_text = " ".join([c.page_content[:200] for c in chunks[:5]]) if chunks else ""
    classification = classify_sensitive_fields(sample_text)

    record = DatasetVersionRecord(
        dataset_id=f"DS-{version_num:04d}",
        version_tag=version_tag,
        source_files=source_files,
        total_chunks=len(chunks),
        data_quality_score=quality_res["quality_score"],
        classification=classification,
        pii_count_detected=pii_count,
        created_at=time.time(),
        metadata=quality_res,
    )

    records.append(record)
    save_lineage_records(records)
    return record


def track_data_drift(baseline_record: DatasetVersionRecord, current_record: DatasetVersionRecord) -> DataDriftReport:
    """Calculate data drift between baseline and current dataset versions."""
    base_len = baseline_record.metadata.get("avg_length", 500)
    curr_len = current_record.metadata.get("avg_length", 500)
    length_diff = abs(curr_len - base_len) / max(1, base_len)

    base_chunks = baseline_record.total_chunks
    curr_chunks = current_record.total_chunks
    chunk_vol_diff = abs(curr_chunks - base_chunks) / max(1, base_chunks)

    drift_score = round(min(1.0, (length_diff * 0.5) + (chunk_vol_diff * 0.5)), 2)
    drift_detected = drift_score > 0.35

    summary = (
        f"Data Drift Score: {drift_score}. Length change: {int(length_diff * 100)}%, Volume change: {int(chunk_vol_diff * 100)}%."
        if drift_detected
        else f"No significant data drift detected between {baseline_record.version_tag} and {current_record.version_tag}."
    )

    return DataDriftReport(
        drift_detected=drift_detected,
        drift_score=drift_score,
        baseline_version=baseline_record.version_tag,
        current_version=current_record.version_tag,
        metrics={"length_diff_pct": length_diff, "volume_diff_pct": chunk_vol_diff},
        summary=summary,
    )
