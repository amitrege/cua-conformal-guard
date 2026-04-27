"""Small adapter interface for host CUA runtimes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cua_guard.types import ActionProposal, GuardDecision, Observation


class ActionAdapter(ABC):
    """Translate host runtime actions into `ActionProposal` objects."""

    name = "base"

    @abstractmethod
    def to_proposal(
        self,
        observation: Observation | dict[str, Any],
        action: dict[str, Any] | str,
    ) -> ActionProposal:
        """Convert a host action into the guard's action schema."""

    def decision_to_host(self, decision: GuardDecision) -> dict[str, Any]:
        """Convert a guard decision into a simple host-readable result."""

        return {
            "adapter": self.name,
            "allowed": decision.allowed,
            "decision": decision.decision,
            "score": decision.score,
            "threshold": decision.threshold,
            "reason": decision.reason,
        }


def coerce_observation(observation: Observation | dict[str, Any]) -> Observation:
    if isinstance(observation, Observation):
        return observation
    return Observation.from_dict(observation)
