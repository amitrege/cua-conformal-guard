"""Core data types shared by classifiers, calibration, and runtime guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HarmCategory(str, Enum):
    """Common harm labels for CUA actions.

    Callers can still pass plain strings. The enum gives examples without making
    the ontology closed.
    """

    DESTRUCTIVE_CHANGE = "destructive_change"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    FINANCIAL_ACTION = "financial_action"
    CREDENTIAL_EXPOSURE = "credential_exposure"
    PRIVACY_EXPOSURE = "privacy_exposure"
    ADMIN_ACTION = "admin_action"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Observation:
    """A compact representation of the current computer state."""

    text: str
    app: str = ""
    window: str = ""
    url: str = ""
    screenshot_path: str = ""
    screenshot_bytes_b64: str = ""
    ocr_text: str = ""
    accessibility_tree: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Observation":
        data = data or {}
        return cls(
            text=str(data.get("text", "")),
            app=str(data.get("app", "")),
            window=str(data.get("window", "")),
            url=str(data.get("url", "")),
            screenshot_path=str(data.get("screenshot_path", "")),
            screenshot_bytes_b64=str(data.get("screenshot_bytes_b64", "")),
            ocr_text=str(data.get("ocr_text", "")),
            accessibility_tree=str(data.get("accessibility_tree", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "app": self.app,
            "window": self.window,
            "url": self.url,
            "screenshot_path": self.screenshot_path,
            "screenshot_bytes_b64": self.screenshot_bytes_b64,
            "ocr_text": self.ocr_text,
            "accessibility_tree": self.accessibility_tree,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class ActionProposal:
    """A proposed GUI action before it is executed."""

    observation: Observation
    action_type: str
    target: str = ""
    text: str = ""
    coordinates: tuple[float, float] | None = None
    target_metadata: dict[str, Any] = field(default_factory=dict)
    raw_agent_message: str = ""
    parsed_command: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionProposal":
        obs = Observation.from_dict(data.get("observation"))
        action = data.get("action", data)
        coords = action.get("coordinates")
        if coords is not None:
            coords = (float(coords[0]), float(coords[1]))
        return cls(
            observation=obs,
            action_type=str(action.get("type", action.get("action_type", ""))),
            target=str(action.get("target", "")),
            text=str(action.get("text", "")),
            coordinates=coords,
            target_metadata=dict(action.get("target_metadata", {})),
            raw_agent_message=str(action.get("raw_agent_message", "")),
            parsed_command=dict(action.get("parsed_command", {})),
            metadata=dict(action.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        action: dict[str, Any] = {
            "type": self.action_type,
            "target": self.target,
            "text": self.text,
            "target_metadata": self.target_metadata,
            "raw_agent_message": self.raw_agent_message,
            "parsed_command": self.parsed_command,
            "metadata": self.metadata,
        }
        if self.coordinates is not None:
            action["coordinates"] = list(self.coordinates)
        return {"observation": self.observation.to_dict(), "action": action}

    def render_for_classifier(self) -> str:
        """Render context and action into a stable text string for classifiers."""

        action_summary = " ".join(
            part
            for part in [self.action_type, self.target, self.text]
            if part.strip()
        )
        pieces = [
            # Repeat the proposed action so it is not drowned out by long GUI text.
            f"proposed_action: {action_summary}",
            f"proposed_action: {action_summary}",
            f"action_type: {self.action_type}",
            f"target: {self.target}",
            f"text: {self.text}",
            f"app: {self.observation.app}",
            f"window: {self.observation.window}",
            f"url: {self.observation.url}",
            f"ocr_text: {self.observation.ocr_text}",
            f"accessibility_tree: {self.observation.accessibility_tree}",
            f"target_metadata: {self.target_metadata}",
            f"raw_agent_message: {self.raw_agent_message}",
            f"parsed_command: {self.parsed_command}",
            f"screen_context: {self.observation.text}",
        ]
        return "\n".join(piece for piece in pieces if piece.strip())


@dataclass(frozen=True)
class LabeledAction:
    """A proposed action with a human or oracle safety label."""

    proposal: ActionProposal
    unsafe: bool
    id: str = ""
    reason: str = ""
    harm_categories: tuple[str, ...] = ()
    severity: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledAction":
        return cls(
            proposal=ActionProposal.from_dict(data),
            unsafe=_as_bool(data.get("unsafe", data.get("label", False))),
            id=str(data.get("id", "")),
            reason=str(data.get("reason", "")),
            harm_categories=tuple(str(item) for item in data.get("harm_categories", [])),
            severity=str(data.get("severity", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.proposal.to_dict()
        out["unsafe"] = self.unsafe
        out["id"] = self.id
        out["reason"] = self.reason
        out["harm_categories"] = list(self.harm_categories)
        out["severity"] = self.severity
        return out


@dataclass(frozen=True)
class LabeledTrajectory:
    """A complete or partial CUA trajectory with one safety label."""

    steps: tuple[ActionProposal, ...]
    unsafe: bool
    id: str = ""
    reason: str = ""
    harm_categories: tuple[str, ...] = ()
    severity: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledTrajectory":
        steps = tuple(ActionProposal.from_dict(step) for step in data.get("steps", []))
        if not steps:
            raise ValueError("A labeled trajectory must contain at least one step.")
        return cls(
            steps=steps,
            unsafe=_as_bool(data.get("unsafe", data.get("label", False))),
            id=str(data.get("id", "")),
            reason=str(data.get("reason", "")),
            harm_categories=tuple(str(item) for item in data.get("harm_categories", [])),
            severity=str(data.get("severity", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "steps": [step.to_dict() for step in self.steps],
            "unsafe": self.unsafe,
            "reason": self.reason,
            "harm_categories": list(self.harm_categories),
            "severity": self.severity,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "unsafe"}
    return bool(value)


@dataclass(frozen=True)
class GuardDecision:
    """The guard's decision for a proposed action."""

    decision: str
    score: float
    threshold: float
    reason: str
    proposal: ActionProposal
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "score": self.score,
            "threshold": self.threshold,
            "reason": self.reason,
            "proposal": self.proposal.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class TrajectoryStep:
    """One runtime step recorded by the guarded runner."""

    observation: Observation
    proposal: ActionProposal
    decision: GuardDecision
    executed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "proposal": self.proposal.to_dict(),
            "decision": self.decision.to_dict(),
            "executed": self.executed,
        }
