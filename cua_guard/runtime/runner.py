"""Episode runner for guarded CUA workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from cua_guard.runtime.agent import ComputerUseAgent
from cua_guard.runtime.environment import ToyComputerEnvironment
from cua_guard.runtime.guard import ConformalActionGuard
from cua_guard.types import GuardDecision, TrajectoryStep


@dataclass(frozen=True)
class EscalationResolution:
    """Host decision for an escalated action."""

    approved: bool
    reason: str = ""


EscalationCallback = Callable[[GuardDecision], bool | EscalationResolution]


@dataclass
class EpisodeResult:
    steps: list[TrajectoryStep] = field(default_factory=list)
    unsafe_event: bool = False
    stopped_by_guard: bool = False
    escalation_count: int = 0
    unresolved_escalation: bool = False

    def to_dict(self) -> dict:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "unsafe_event": self.unsafe_event,
            "stopped_by_guard": self.stopped_by_guard,
            "escalation_count": self.escalation_count,
            "unresolved_escalation": self.unresolved_escalation,
        }


def run_episode(
    agent: ComputerUseAgent,
    env: ToyComputerEnvironment,
    guard: ConformalActionGuard,
    max_steps: int = 10,
    on_escalate: EscalationCallback | None = None,
) -> EpisodeResult:
    result = EpisodeResult()
    for _ in range(max_steps):
        observation = env.observe()
        proposal = agent.propose(observation)
        decision = guard.evaluate(proposal)
        executed = False
        halt = False
        if decision.allowed:
            env.step(proposal)
            executed = True
        elif decision.decision == "escalate":
            result.escalation_count += 1
            if on_escalate is None:
                result.stopped_by_guard = True
                result.unresolved_escalation = True
                halt = True
            else:
                resolution = _resolve_escalation(on_escalate(decision))
                if resolution.approved:
                    env.step(proposal)
                    executed = True
                else:
                    result.stopped_by_guard = True
                    halt = True
        else:
            result.stopped_by_guard = True
            halt = True
        result.steps.append(
            TrajectoryStep(
                observation=observation,
                proposal=proposal,
                decision=decision,
                executed=executed,
            )
        )
        if proposal.action_type == "done" or halt:
            break
    result.unsafe_event = env.unsafe_event
    return result


def _resolve_escalation(value: bool | EscalationResolution) -> EscalationResolution:
    if isinstance(value, EscalationResolution):
        return value
    if isinstance(value, bool):
        return EscalationResolution(approved=value)
    raise TypeError("on_escalate must return bool or EscalationResolution.")
