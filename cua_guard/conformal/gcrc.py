"""Generalized conformal risk control for danger-score thresholds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.types import LabeledAction


LossFn = Callable[[float, bool, float], float]
EDGE_TOLERANCE = 5e-5


def missed_unsafe_loss(threshold: float, unsafe: bool, score: float) -> float:
    """Default loss: an unsafe action was allowed by the gate."""

    return 1.0 if unsafe and score < threshold else 0.0


@dataclass(frozen=True)
class CalibrationResult:
    """Result of threshold calibration."""

    threshold: float
    alpha: float
    empirical_risk: float
    conservative_risk: float
    n: int
    feasible: bool
    loss_bound: float
    threshold_index: int
    grid: list[float]
    envelope_risks: list[float]
    score_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "alpha": self.alpha,
            "empirical_risk": self.empirical_risk,
            "conservative_risk": self.conservative_risk,
            "n": self.n,
            "feasible": self.feasible,
            "loss_bound": self.loss_bound,
            "threshold_index": self.threshold_index,
            "boundary": self.boundary,
            "boundary_warning": self.boundary_warning(),
            "grid": self.grid,
            "envelope_risks": self.envelope_risks,
            "score_summary": self.score_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationResult":
        return cls(
            threshold=float(data["threshold"]),
            alpha=float(data["alpha"]),
            empirical_risk=float(data["empirical_risk"]),
            conservative_risk=float(data["conservative_risk"]),
            n=int(data["n"]),
            feasible=bool(data["feasible"]),
            loss_bound=float(data["loss_bound"]),
            threshold_index=int(data["threshold_index"]),
            grid=[float(item) for item in data["grid"]],
            envelope_risks=[float(item) for item in data["envelope_risks"]],
            score_summary={
                str(key): float(value)
                for key, value in data.get("score_summary", {}).items()
            },
        )

    @property
    def boundary(self) -> str | None:
        """Return whether the selected threshold is at a calibration grid edge."""

        if not self.grid:
            return None
        if self.threshold_index <= 0 or self.threshold <= self.grid[0] + EDGE_TOLERANCE:
            return "min"
        if (
            self.threshold_index >= len(self.grid) - 1
            or self.threshold >= self.grid[-1] - EDGE_TOLERANCE
        ):
            return "max"
        return None

    def boundary_warning(self) -> str | None:
        """Explain why an edge threshold should be treated as an audit signal."""

        if self.boundary == "max":
            return (
                "calibrated threshold is at or near the maximum grid value; the "
                "guard may allow nearly all actions. Use more calibration data, "
                "lower alpha, or improve the classifier."
            )
        if self.boundary == "min":
            return (
                "calibrated threshold is at or near the minimum grid value; the "
                "guard may block nearly all actions, or calibration may be "
                "infeasible. Use more calibration data, adjust alpha, or improve "
                "the classifier."
            )
        return None


class GCRCThresholdCalibrator:
    """Calibrate an allow threshold for a danger score.

    Larger scores mean more dangerous. Larger thresholds allow more actions.
    Calibration searches from conservative to aggressive and accepts a threshold
    only if every threshold up to it has conservative empirical risk <= alpha.
    """

    def __init__(
        self,
        alpha: float,
        loss_bound: float = 1.0,
        grid_size: int = 201,
        loss_fn: LossFn = missed_unsafe_loss,
    ) -> None:
        if not 0 <= alpha <= loss_bound:
            raise ValueError("alpha must be between 0 and loss_bound.")
        if loss_bound <= 0:
            raise ValueError("loss_bound must be positive.")
        if grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        self.alpha = float(alpha)
        self.loss_bound = float(loss_bound)
        self.grid_size = int(grid_size)
        self.loss_fn = loss_fn

    def calibrate(
        self,
        scores: Sequence[float],
        unsafe_labels: Sequence[bool],
        grid: Sequence[float] | None = None,
    ) -> CalibrationResult:
        if len(scores) != len(unsafe_labels):
            raise ValueError("scores and labels must have the same length.")
        if not scores:
            raise ValueError("At least one calibration example is required.")

        clean_scores = [min(1.0, max(0.0, float(score))) for score in scores]
        labels = [bool(label) for label in unsafe_labels]
        thresholds = sorted(set(float(item) for item in (grid or self._default_grid(clean_scores))))

        best_index = -1
        best_empirical = 0.0
        best_conservative = self.loss_bound
        envelope_risks: list[float] = []
        prefix_max = 0.0

        for index, threshold in enumerate(thresholds):
            losses = [
                self.loss_fn(threshold, unsafe, score)
                for score, unsafe in zip(clean_scores, labels)
            ]
            empirical = sum(losses) / len(losses)
            conservative = (sum(losses) + self.loss_bound) / (len(losses) + 1)
            prefix_max = max(prefix_max, conservative)
            envelope_risks.append(prefix_max)
            if prefix_max <= self.alpha:
                best_index = index
                best_empirical = empirical
                best_conservative = conservative

        feasible = best_index >= 0
        if feasible:
            threshold = thresholds[best_index]
        else:
            threshold = thresholds[0]
            losses = [
                self.loss_fn(threshold, unsafe, score)
                for score, unsafe in zip(clean_scores, labels)
            ]
            best_empirical = sum(losses) / len(losses)
            best_conservative = (sum(losses) + self.loss_bound) / (len(losses) + 1)
            best_index = 0

        return CalibrationResult(
            threshold=threshold,
            alpha=self.alpha,
            empirical_risk=best_empirical,
            conservative_risk=best_conservative,
            n=len(clean_scores),
            feasible=feasible,
            loss_bound=self.loss_bound,
            threshold_index=best_index,
            grid=thresholds,
            envelope_risks=envelope_risks,
            score_summary=_score_summary(clean_scores),
        )

    def calibrate_actions(
        self, classifier: DangerClassifier, records: Sequence[LabeledAction]
    ) -> CalibrationResult:
        scores = [classifier.score(record.proposal) for record in records]
        labels = [record.unsafe for record in records]
        return self.calibrate(scores, labels)

    def _default_grid(self, scores: Sequence[float]) -> list[float]:
        uniform = [i / (self.grid_size - 1) for i in range(self.grid_size)]
        around_scores: list[float] = []
        eps = 1e-9
        for score in scores:
            around_scores.extend([max(0.0, score - eps), score, min(1.0, score + eps)])
        return sorted(set(uniform + around_scores))


def conservative_risk_at_threshold(
    scores: Sequence[float],
    unsafe_labels: Sequence[bool],
    threshold: float,
    loss_bound: float = 1.0,
    loss_fn: LossFn = missed_unsafe_loss,
) -> float:
    if len(scores) != len(unsafe_labels):
        raise ValueError("scores and labels must have the same length.")
    if not scores:
        raise ValueError("At least one example is required.")
    losses = [
        loss_fn(threshold, bool(label), min(1.0, max(0.0, float(score))))
        for score, label in zip(scores, unsafe_labels)
    ]
    return (sum(losses) + loss_bound) / (len(losses) + 1)


def _score_summary(scores: Sequence[float]) -> dict[str, float]:
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": variance**0.5,
    }
