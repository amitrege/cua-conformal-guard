"""Core data types shared by classifiers, calibration, and runtime guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    """A compact textual representation of the current computer state."""

    text: str
    app: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Observation":
        data = data or {}
        return cls(
            text=str(data.get("text", "")),
            app=str(data.get("app", "")),
            url=str(data.get("url", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "app": self.app,
            "url": self.url,
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
            metadata=dict(action.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        action: dict[str, Any] = {
            "type": self.action_type,
            "target": self.target,
            "text": self.text,
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
            f"url: {self.observation.url}",
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledAction":
        return cls(
            proposal=ActionProposal.from_dict(data),
            unsafe=bool(data.get("unsafe", data.get("label", False))),
            id=str(data.get("id", "")),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        out = self.proposal.to_dict()
        out["unsafe"] = self.unsafe
        out["id"] = self.id
        out["reason"] = self.reason
        return out


@dataclass(frozen=True)
class LabeledTrajectory:
    """A complete or partial CUA trajectory with one safety label."""

    steps: tuple[ActionProposal, ...]
    unsafe: bool
    id: str = ""
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabeledTrajectory":
        steps = tuple(ActionProposal.from_dict(step) for step in data.get("steps", []))
        if not steps:
            raise ValueError("A labeled trajectory must contain at least one step.")
        return cls(
            steps=steps,
            unsafe=bool(data.get("unsafe", data.get("label", False))),
            id=str(data.get("id", "")),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "steps": [step.to_dict() for step in self.steps],
            "unsafe": self.unsafe,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GuardDecision:
    """The guard's decision for a proposed action."""

    decision: str
    score: float
    threshold: float
    reason: str
    proposal: ActionProposal

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
