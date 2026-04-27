"""Conformal safety gates for computer-use agents."""

from cua_guard.types import (
    ActionProposal,
    GuardDecision,
    HarmCategory,
    LabeledAction,
    LabeledTrajectory,
    Observation,
)
from cua_guard.classifiers.naive_bayes import NaiveBayesDangerClassifier
from cua_guard.classifiers.rules import KeywordDangerScorer
from cua_guard.conformal.gcrc import CalibrationResult, GCRCThresholdCalibrator
from cua_guard.evaluation import EvaluationReport, evaluate_labeled_actions
from cua_guard.runtime.guard import ConformalActionGuard

__all__ = [
    "ActionProposal",
    "CalibrationResult",
    "ConformalActionGuard",
    "EvaluationReport",
    "GCRCThresholdCalibrator",
    "GuardDecision",
    "HarmCategory",
    "KeywordDangerScorer",
    "LabeledAction",
    "LabeledTrajectory",
    "NaiveBayesDangerClassifier",
    "Observation",
    "evaluate_labeled_actions",
]
