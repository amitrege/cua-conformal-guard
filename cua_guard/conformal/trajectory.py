"""Trajectory-level conformal calibration helpers."""

from __future__ import annotations

from collections.abc import Sequence

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.conformal.gcrc import CalibrationResult, GCRCThresholdCalibrator
from cua_guard.types import ActionProposal, LabeledTrajectory


def trajectory_score(
    classifier: DangerClassifier, steps: Sequence[ActionProposal]
) -> float:
    """Score a trajectory by its riskiest proposed action."""

    if not steps:
        raise ValueError("Cannot score an empty trajectory.")
    return max(classifier.score(step) for step in steps)


def calibrate_trajectories(
    classifier: DangerClassifier,
    trajectories: Sequence[LabeledTrajectory],
    calibrator: GCRCThresholdCalibrator,
) -> CalibrationResult:
    """Calibrate a threshold for trajectory-level missed-danger risk."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    scores = [trajectory_score(classifier, item.steps) for item in trajectories]
    labels = [item.unsafe for item in trajectories]
    return calibrator.calibrate(scores, labels)
