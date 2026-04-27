"""End-to-end integration example.

Trains a classifier, calibrates a guard, and uses the guard to gate a tiny
fake CUA that proposes a few actions against a fake environment. No browser,
no desktop — just enough to make the call pattern concrete.

Run it from the repo root:

    python3 examples/integrate_with_agent.py

What this demonstrates:

1. Training a danger classifier from labeled JSONL.
2. Calibrating a threshold for a target miss rate.
3. Wrapping a real agent loop with `guard.evaluate(proposal)` so unsafe
   actions never reach `env.step(...)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cua_guard import (
    ConformalActionGuard,
    GCRCThresholdCalibrator,
    NaiveBayesDangerClassifier,
)
from cua_guard.io import load_labeled_actions
from cua_guard.types import ActionProposal, Observation


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "examples" / "train_actions.jsonl"
CALIBRATION_PATH = REPO_ROOT / "examples" / "calibration_actions.jsonl"


class FakeAgent:
    """A scripted agent: walks through a list of (action_type, target) pairs."""

    def __init__(self, plan: list[tuple[str, str]]) -> None:
        self.plan = plan
        self.cursor = 0

    def propose(self, observation: Observation) -> ActionProposal:
        if self.cursor >= len(self.plan):
            action_type, target = "done", "finish"
        else:
            action_type, target = self.plan[self.cursor]
            self.cursor += 1
        return ActionProposal(
            observation=observation,
            action_type=action_type,
            target=target,
        )


class FakeEnv:
    """A fake environment that 'executes' actions by appending to a log."""

    def __init__(self) -> None:
        self.executed: list[str] = []
        self.screen_text = (
            "Project settings page. Buttons: Search docs, Save notes, "
            "Send invite, Delete project."
        )

    def observe(self) -> Observation:
        return Observation(app="fake-desktop", text=self.screen_text)

    def step(self, proposal: ActionProposal) -> None:
        self.executed.append(f"{proposal.action_type} {proposal.target}".strip())


def build_guard(alpha: float = 0.10) -> ConformalActionGuard:
    """Train a classifier, calibrate a threshold, return a ready-to-use guard."""

    train_records = load_labeled_actions(TRAIN_PATH)
    classifier = NaiveBayesDangerClassifier().fit(train_records)

    calibration_records = load_labeled_actions(CALIBRATION_PATH)
    calibrator = GCRCThresholdCalibrator(alpha=alpha)
    result = calibrator.calibrate_actions(classifier, calibration_records)

    print(
        f"calibrated threshold={result.threshold:.4f} "
        f"alpha={result.alpha:.4f} "
        f"conservative_risk={result.conservative_risk:.4f} "
        f"feasible={result.feasible}"
    )
    if warning := result.boundary_warning():
        print(f"warning: {warning}")

    return ConformalActionGuard.from_calibration(classifier, result, mode="block")


def run_loop(agent: FakeAgent, env: FakeEnv, guard: ConformalActionGuard) -> None:
    """The integration pattern: score the proposal, decide, then execute or stop."""

    for _ in range(10):
        observation = env.observe()
        proposal = agent.propose(observation)

        decision = guard.evaluate(proposal)
        verdict = "allow" if decision.allowed else decision.decision
        print(
            f"  {proposal.action_type:>6} {proposal.target!r:30}  "
            f"score={decision.score:.3f}  threshold={decision.threshold:.3f}  -> {verdict}"
        )

        if proposal.action_type == "done":
            break

        if decision.allowed:
            env.step(proposal)
        else:
            print(f"  gate intervened: {decision.reason}")
            break


def main() -> None:
    print("=== building guard from labeled examples ===")
    guard = build_guard(alpha=0.10)

    print("\n=== running guarded agent loop ===")
    plan = [
        ("click", "Search docs"),       # safe
        ("click", "Save notes"),        # safe
        ("click", "Delete project"),    # unsafe — gate should stop here
        ("click", "Send invite"),       # never reached
    ]
    agent = FakeAgent(plan)
    env = FakeEnv()
    run_loop(agent, env, guard)

    print("\n=== environment log ===")
    if not env.executed:
        print("  (nothing executed)")
    for entry in env.executed:
        print(f"  executed: {entry}")


if __name__ == "__main__":
    main()
