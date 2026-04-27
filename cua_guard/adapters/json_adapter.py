"""Adapter for the native CUA Guard JSON action schema."""

from __future__ import annotations

from typing import Any

from cua_guard.adapters.base import ActionAdapter, coerce_observation
from cua_guard.types import ActionProposal, Observation


class JsonActionAdapter(ActionAdapter):
    """Accepts native guard JSON or a flat action dictionary."""

    name = "json"

    def to_proposal(
        self,
        observation: Observation | dict[str, Any],
        action: dict[str, Any] | str,
    ) -> ActionProposal:
        obs = coerce_observation(observation)
        if isinstance(action, str):
            action = {"type": "text", "target": action}
        if "observation" in action:
            return ActionProposal.from_dict(action)
        return ActionProposal.from_dict({"observation": obs.to_dict(), "action": action})
