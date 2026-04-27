"""Runtime guard that applies a calibrated threshold to proposed actions."""

from __future__ import annotations

from pathlib import Path

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.classifiers.loader import load_classifier
from cua_guard.conformal.gcrc import CalibrationResult
from cua_guard.io import read_json, write_json
from cua_guard.types import ActionProposal, GuardDecision


class ConformalActionGuard:
    """Allow, escalate, or block CUA actions using a calibrated danger threshold."""

    def __init__(
        self,
        classifier: DangerClassifier,
        threshold: float,
        mode: str = "escalate",
        alpha: float | None = None,
    ) -> None:
        if mode not in {"escalate", "block"}:
            raise ValueError("mode must be either 'escalate' or 'block'.")
        self.classifier = classifier
        self.threshold = float(threshold)
        self.mode = mode
        self.alpha = alpha

    @classmethod
    def from_calibration(
        cls,
        classifier: DangerClassifier,
        calibration: CalibrationResult,
        mode: str = "escalate",
    ) -> "ConformalActionGuard":
        return cls(
            classifier=classifier,
            threshold=calibration.threshold,
            mode=mode,
            alpha=calibration.alpha,
        )

    def evaluate(self, proposal: ActionProposal) -> GuardDecision:
        score = self.classifier.score(proposal)
        if score < self.threshold:
            return GuardDecision(
                decision="allow",
                score=score,
                threshold=self.threshold,
                reason="score below calibrated danger threshold",
                proposal=proposal,
            )
        return GuardDecision(
            decision=self.mode,
            score=score,
            threshold=self.threshold,
            reason="score exceeded calibrated danger threshold",
            proposal=proposal,
        )

    def save_bundle(
        self,
        path: str | Path,
        classifier_path: str | Path,
        calibration: CalibrationResult,
    ) -> None:
        write_json(
            path,
            {
                "type": "conformal_action_guard",
                "classifier_path": str(classifier_path),
                "threshold": self.threshold,
                "mode": self.mode,
                "alpha": self.alpha,
                "calibration": calibration.to_dict(),
            },
        )

    @classmethod
    def load_bundle(cls, path: str | Path) -> "ConformalActionGuard":
        data = read_json(path)
        classifier_path = Path(data["classifier_path"])
        if not classifier_path.is_absolute():
            classifier_path = Path(path).parent / classifier_path
        classifier = load_classifier(classifier_path)
        return cls(
            classifier=classifier,
            threshold=float(data["threshold"]),
            mode=str(data.get("mode", "escalate")),
            alpha=data.get("alpha"),
        )
