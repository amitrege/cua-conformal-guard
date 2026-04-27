"""Command-line workflow for training, calibration, and demo execution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cua_guard.classifiers.loader import load_classifier
from cua_guard.classifiers.naive_bayes import NaiveBayesDangerClassifier
from cua_guard.conformal.gcrc import CalibrationResult, GCRCThresholdCalibrator
from cua_guard.conformal.trajectory import calibrate_trajectories
from cua_guard.io import load_labeled_actions, load_labeled_trajectories, read_json
from cua_guard.runtime.agent import ScriptedAgent
from cua_guard.runtime.environment import ToyComputerEnvironment
from cua_guard.runtime.guard import ConformalActionGuard
from cua_guard.runtime.runner import EscalationResolution, run_episode


def _print_calibration_warning(result: CalibrationResult) -> None:
    warning = result.boundary_warning()
    if warning:
        print(f"warning: {warning}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conformal guard for CUA agents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-classifier", help="Train a danger classifier.")
    train.add_argument("--data", required=True, help="Training JSONL file.")
    train.add_argument("--model", required=True, help="Output model JSON path.")

    calibrate = subparsers.add_parser("calibrate", help="Calibrate a guard threshold.")
    calibrate.add_argument("--data", required=True, help="Held-out calibration JSONL file.")
    calibrate.add_argument("--model", required=True, help="Classifier model JSON path.")
    calibrate.add_argument("--guard", required=True, help="Output guard bundle JSON path.")
    calibrate.add_argument("--alpha", type=float, default=0.25, help="Target missed-danger risk.")
    calibrate.add_argument(
        "--mode",
        choices=["escalate", "block"],
        default="escalate",
        help="Decision for actions above threshold.",
    )

    calibrate_traj = subparsers.add_parser(
        "calibrate-trajectories",
        help="Calibrate a threshold from trajectory-level labels.",
    )
    calibrate_traj.add_argument("--data", required=True, help="Trajectory JSONL file.")
    calibrate_traj.add_argument("--model", required=True, help="Classifier model JSON path.")
    calibrate_traj.add_argument("--guard", required=True, help="Output guard bundle JSON path.")
    calibrate_traj.add_argument("--alpha", type=float, default=0.25)
    calibrate_traj.add_argument(
        "--mode",
        choices=["escalate", "block"],
        default="escalate",
    )

    demo = subparsers.add_parser("run-demo", help="Run a toy CUA episode.")
    demo.add_argument("--guard", required=True, help="Guard bundle JSON path.")
    demo.add_argument("--max-steps", type=int, default=5)
    demo.add_argument(
        "--approve-escalations",
        action="store_true",
        help="Demo-only callback that approves escalated actions.",
    )

    inspect = subparsers.add_parser("inspect-guard", help="Print guard bundle JSON.")
    inspect.add_argument("--guard", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train-classifier":
        records = load_labeled_actions(args.data)
        model = NaiveBayesDangerClassifier().fit(records)
        model.save(args.model)
        print(f"trained classifier on {len(records)} records -> {args.model}")
        return 0

    if args.command == "calibrate":
        classifier = load_classifier(args.model)
        records = load_labeled_actions(args.data)
        calibrator = GCRCThresholdCalibrator(alpha=args.alpha)
        result = calibrator.calibrate_actions(classifier, records)
        guard = ConformalActionGuard.from_calibration(classifier, result, mode=args.mode)

        guard_path = Path(args.guard)
        model_path = Path(args.model)
        classifier_ref = model_path
        try:
            classifier_ref = model_path.relative_to(guard_path.parent)
        except ValueError:
            pass
        guard.save_bundle(guard_path, classifier_ref, result)

        print(
            "calibrated threshold="
            f"{result.threshold:.4f} alpha={result.alpha:.4f} "
            f"conservative_risk={result.conservative_risk:.4f} "
            f"feasible={result.feasible} -> {args.guard}"
        )
        _print_calibration_warning(result)
        return 0

    if args.command == "calibrate-trajectories":
        classifier = load_classifier(args.model)
        trajectories = load_labeled_trajectories(args.data)
        calibrator = GCRCThresholdCalibrator(alpha=args.alpha)
        result = calibrate_trajectories(classifier, trajectories, calibrator)
        guard = ConformalActionGuard.from_calibration(classifier, result, mode=args.mode)

        guard_path = Path(args.guard)
        model_path = Path(args.model)
        classifier_ref = model_path
        try:
            classifier_ref = model_path.relative_to(guard_path.parent)
        except ValueError:
            pass
        guard.save_bundle(guard_path, classifier_ref, result)

        print(
            "calibrated trajectory threshold="
            f"{result.threshold:.4f} alpha={result.alpha:.4f} "
            f"conservative_risk={result.conservative_risk:.4f} "
            f"feasible={result.feasible} -> {args.guard}"
        )
        _print_calibration_warning(result)
        return 0

    if args.command == "run-demo":
        guard = ConformalActionGuard.load_bundle(args.guard)
        agent = ScriptedAgent(
            [
                {"type": "click", "target": "Search docs"},
                {"type": "click", "target": "Delete project"},
                {"type": "done", "target": "finish"},
            ]
        )
        env = ToyComputerEnvironment()
        on_escalate = None
        if args.approve_escalations:
            on_escalate = lambda _decision: EscalationResolution(
                approved=True,
                reason="approved by --approve-escalations demo flag",
            )
        result = run_episode(
            agent,
            env,
            guard,
            max_steps=args.max_steps,
            on_escalate=on_escalate,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "inspect-guard":
        print(json.dumps(read_json(args.guard), indent=2, sort_keys=True))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
