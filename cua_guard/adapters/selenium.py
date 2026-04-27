"""Adapter for Selenium-like browser commands."""

from __future__ import annotations

from typing import Any

from cua_guard.adapters.base import ActionAdapter, coerce_observation
from cua_guard.types import ActionProposal, Observation


class SeleniumActionAdapter(ActionAdapter):
    """Convert common Selenium command dictionaries into guard proposals."""

    name = "selenium"

    def to_proposal(
        self,
        observation: Observation | dict[str, Any],
        action: dict[str, Any] | str,
    ) -> ActionProposal:
        obs = coerce_observation(observation)
        if isinstance(action, str):
            action = {"command": action}
        command = str(action.get("command", action.get("type", ""))).lower()
        selector = str(action.get("selector", action.get("element", action.get("target", ""))))
        text = str(action.get("text", action.get("value", "")))
        url = str(action.get("url", ""))
        return ActionProposal(
            observation=obs,
            action_type=command or "selenium",
            target=selector or url,
            text=text,
            target_metadata={"selector": selector, "url": url},
            parsed_command=dict(action),
            metadata={"adapter": self.name},
        )
