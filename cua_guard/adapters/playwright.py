"""Adapter for Playwright-like browser actions.

The adapter is dependency-free. It expects dictionaries shaped like the commands
your own Playwright wrapper is about to execute.
"""

from __future__ import annotations

from typing import Any

from cua_guard.adapters.base import ActionAdapter, coerce_observation
from cua_guard.types import ActionProposal, Observation


class PlaywrightActionAdapter(ActionAdapter):
    """Convert common Playwright command dictionaries into guard proposals."""

    name = "playwright"

    def to_proposal(
        self,
        observation: Observation | dict[str, Any],
        action: dict[str, Any] | str,
    ) -> ActionProposal:
        obs = coerce_observation(observation)
        if isinstance(action, str):
            action = {"method": action}
        method = str(action.get("method", action.get("type", ""))).lower()
        selector = str(action.get("selector", action.get("target", "")))
        text = str(action.get("text", action.get("value", "")))
        url = str(action.get("url", ""))
        files = action.get("files", action.get("file", ""))
        target = selector or url or str(action.get("label", ""))
        if method in {"fill", "type"} and text:
            target = target or "text input"
        if method in {"set_input_files", "upload_file"}:
            text = str(files)
        return ActionProposal(
            observation=obs,
            action_type=method or "playwright",
            target=target,
            text=text,
            target_metadata={
                "selector": selector,
                "url": url,
                "files": files,
            },
            parsed_command=dict(action),
            metadata={"adapter": self.name},
        )
