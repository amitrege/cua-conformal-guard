"""Classifier interface used by the conformal guard."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from cua_guard.types import ActionProposal, LabeledAction


class DangerClassifier(ABC):
    """Maps a proposed CUA action to a danger score in [0, 1]."""

    @abstractmethod
    def fit(self, records: list[LabeledAction]) -> "DangerClassifier":
        """Train or update the classifier."""

    @abstractmethod
    def score(self, proposal: ActionProposal) -> float:
        """Return a danger score. Larger means more dangerous."""

    def score_batch(self, proposals: list[ActionProposal]) -> list[float]:
        """Score several proposals.

        Classifiers can override this for vectorized backends. The default keeps
        the interface dependency-free.
        """

        return [self.score(proposal) for proposal in proposals]

    def metadata(self) -> dict[str, Any]:
        """Return audit metadata for saved models and runtime decisions."""

        return {
            "type": self.__class__.__name__,
            "score_range": [0.0, 1.0],
        }

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the classifier."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "DangerClassifier":
        """Load a persisted classifier."""
