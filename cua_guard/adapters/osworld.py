"""Adapter for OSWorld-style desktop action dictionaries."""

from __future__ import annotations

from typing import Any

from cua_guard.adapters.base import ActionAdapter, coerce_observation
from cua_guard.types import ActionProposal, Observation


class OSWorldActionAdapter(ActionAdapter):
    """Convert desktop action dictionaries into guard proposals."""

    name = "osworld"

    def to_proposal(
        self,
        observation: Observation | dict[str, Any],
        action: dict[str, Any] | str,
    ) -> ActionProposal:
        obs = coerce_observation(observation)
        if isinstance(action, str):
            action = {"action": action}
        action_type = str(
            action.get("action_type", action.get("type", action.get("action", "")))
        ).lower()
        target = str(action.get("target", action.get("element", "")))
        text = str(action.get("text", action.get("content", "")))
        coords = _coordinates(action)
        return ActionProposal(
            observation=obs,
            action_type=action_type or "desktop",
            target=target,
            text=text,
            coordinates=coords,
            target_metadata={
                "window": action.get("window", ""),
                "element": action.get("element", ""),
            },
            parsed_command=dict(action),
            metadata={"adapter": self.name},
        )


def _coordinates(action: dict[str, Any]) -> tuple[float, float] | None:
    if "coordinates" in action and action["coordinates"] is not None:
        coords = action["coordinates"]
        return (float(coords[0]), float(coords[1]))
    if "x" in action and "y" in action:
        return (float(action["x"]), float(action["y"]))
    return None
