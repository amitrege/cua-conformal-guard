"""Minimal CUA agent interfaces and wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from cua_guard.runtime.guard import ConformalActionGuard
from cua_guard.types import ActionProposal, GuardDecision, Observation


class ComputerUseAgent(ABC):
    """A policy that proposes the next computer action from an observation."""

    @abstractmethod
    def propose(self, observation: Observation) -> ActionProposal:
        """Return the next proposed action."""


class ScriptedAgent(ComputerUseAgent):
    """A deterministic agent useful for demos and tests."""

    def __init__(self, action_specs: list[dict]) -> None:
        self.action_specs = list(action_specs)
        self.index = 0

    def propose(self, observation: Observation) -> ActionProposal:
        if self.index >= len(self.action_specs):
            spec = {"type": "done", "target": "finish"}
        else:
            spec = self.action_specs[self.index]
            self.index += 1
        return ActionProposal(
            observation=observation,
            action_type=str(spec.get("type", "")),
            target=str(spec.get("target", "")),
            text=str(spec.get("text", "")),
        )


@dataclass(frozen=True)
class GuardedProposal:
    proposal: ActionProposal
    decision: GuardDecision


class GuardedAgent:
    """Composes a base CUA agent with a conformal action guard."""

    def __init__(self, base_agent: ComputerUseAgent, guard: ConformalActionGuard) -> None:
        self.base_agent = base_agent
        self.guard = guard

    def propose(self, observation: Observation) -> GuardedProposal:
        proposal = self.base_agent.propose(observation)
        decision = self.guard.evaluate(proposal)
        return GuardedProposal(proposal=proposal, decision=decision)
