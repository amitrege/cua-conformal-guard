"""Audit traces for guarded CUA runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from cua_guard.io import write_jsonl
from cua_guard.types import TrajectoryStep


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AuditRecord:
    """One guard decision written to a trace."""

    timestamp: str
    run_id: str
    step_index: int
    observation: dict[str, Any]
    proposed_action: dict[str, Any]
    score: float
    threshold: float
    decision: str
    reason: str
    executed: bool
    classifier: dict[str, Any] = field(default_factory=dict)
    guard: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_step(
        cls,
        step: TrajectoryStep,
        step_index: int,
        run_id: str,
        labels: dict[str, Any] | None = None,
    ) -> "AuditRecord":
        metadata = step.decision.metadata
        return cls(
            timestamp=utc_now(),
            run_id=run_id,
            step_index=step_index,
            observation=step.observation.to_dict(),
            proposed_action=step.proposal.to_dict()["action"],
            score=step.decision.score,
            threshold=step.decision.threshold,
            decision=step.decision.decision,
            reason=step.decision.reason,
            executed=step.executed,
            classifier=dict(metadata.get("classifier", {})),
            guard={
                "type": metadata.get("type", "conformal_action_guard"),
                "version": metadata.get("version", ""),
                "mode": metadata.get("mode", ""),
                "alpha": metadata.get("alpha"),
            },
            labels=dict(labels or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "observation": self.observation,
            "proposed_action": self.proposed_action,
            "score": self.score,
            "threshold": self.threshold,
            "decision": self.decision,
            "reason": self.reason,
            "executed": self.executed,
            "classifier": self.classifier,
            "guard": self.guard,
            "labels": self.labels,
        }


class AuditLogger(Protocol):
    def log(self, record: AuditRecord) -> None:
        """Write one audit record."""


class JsonlAuditLogger:
    """Append guard decisions to a JSONL trace."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_json_dumps(record.to_dict()))
            handle.write("\n")


class InMemoryAuditLogger:
    """Audit logger used by tests and notebooks."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    def log(self, record: AuditRecord) -> None:
        self.records.append(record)


def write_audit_records(path: str | Path, records: list[AuditRecord]) -> None:
    write_jsonl(path, [record.to_dict() for record in records])


def _json_dumps(data: dict[str, Any]) -> str:
    import json

    return json.dumps(data, sort_keys=True)
