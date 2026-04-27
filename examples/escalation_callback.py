"""Escalation example: route above-threshold actions through a reviewer.

Calibrates a guard in `escalate` mode and shows three escalation policies you
might wire into a real deployment:

1. `auto_deny`   - default safe behavior when no human is around.
2. `auto_approve` - a permissive callback for automated tests/demos.
3. `prompt_user` - terminal prompt simulating a human-in-the-loop reviewer.

Run from the repo root:

    python3 examples/escalation_callback.py            # auto-deny
    python3 examples/escalation_callback.py approve    # auto-approve
    python3 examples/escalation_callback.py prompt     # ask the terminal user

The same wiring would route to Slack, a stricter LLM reviewer, or a queue.
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
from cua_guard.runtime.agent import ScriptedAgent
from cua_guard.runtime.environment import ToyComputerEnvironment
from cua_guard.runtime.runner import EscalationResolution, run_episode


REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO_ROOT / "examples" / "train_actions.jsonl"
CALIBRATION_PATH = REPO_ROOT / "examples" / "calibration_actions.jsonl"


def build_guard(alpha: float = 0.10) -> ConformalActionGuard:
    classifier = NaiveBayesDangerClassifier().fit(load_labeled_actions(TRAIN_PATH))
    result = GCRCThresholdCalibrator(alpha=alpha).calibrate_actions(
        classifier, load_labeled_actions(CALIBRATION_PATH)
    )
    print(
        f"calibrated threshold={result.threshold:.4f} alpha={result.alpha:.4f} "
        f"feasible={result.feasible}"
    )
    return ConformalActionGuard.from_calibration(classifier, result, mode="escalate")


def auto_deny(decision) -> EscalationResolution:
    return EscalationResolution(approved=False, reason="no reviewer available")


def auto_approve(decision) -> EscalationResolution:
    return EscalationResolution(approved=True, reason="permissive demo policy")


def prompt_user(decision) -> EscalationResolution:
    proposal = decision.proposal
    print()
    print("--- escalation ---")
    print(f"  action: {proposal.action_type} {proposal.target}")
    print(f"  score:  {decision.score:.3f}  (threshold {decision.threshold:.3f})")
    print(f"  reason: {decision.reason}")
    answer = input("approve? [y/N]: ").strip().lower()
    approved = answer in {"y", "yes"}
    return EscalationResolution(
        approved=approved,
        reason="approved by terminal reviewer" if approved else "denied by terminal reviewer",
    )


POLICIES = {
    "deny": auto_deny,
    "approve": auto_approve,
    "prompt": prompt_user,
}


def main(argv: list[str]) -> int:
    policy_name = argv[1] if len(argv) > 1 else "deny"
    if policy_name not in POLICIES:
        print(f"unknown policy {policy_name!r}; pick one of {sorted(POLICIES)}")
        return 2
    on_escalate = POLICIES[policy_name]

    guard = build_guard(alpha=0.10)
    agent = ScriptedAgent(
        [
            {"type": "click", "target": "Search docs"},
            {"type": "click", "target": "Delete project"},
            {"type": "done", "target": "finish"},
        ]
    )
    env = ToyComputerEnvironment()

    print(f"\n=== running episode with escalation policy: {policy_name!r} ===")
    result = run_episode(agent, env, guard, on_escalate=on_escalate)

    for index, step in enumerate(result.steps):
        decision = step.decision
        verdict = "allow" if decision.allowed else decision.decision
        print(
            f"  step {index}: {step.proposal.action_type} {step.proposal.target!r}  "
            f"score={decision.score:.3f}  -> {verdict}  executed={step.executed}"
        )

    print(
        f"\nstopped_by_guard={result.stopped_by_guard}  "
        f"escalations={result.escalation_count}  "
        f"unsafe_event={result.unsafe_event}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
