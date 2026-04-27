"""Classifier interface used by the conformal guard."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from cua_guard.types import ActionProposal, LabeledAction


class DangerClassifier(ABC):
    """Maps a proposed CUA action to a danger score in [0, 1]."""

    @abstractmethod
    def fit(self, records: list[LabeledAction]) -> "DangerClassifier":
        """Train or update the classifier."""

    @abstractmethod
    def score(self, proposal: ActionProposal) -> float:
        """Return a danger score. Larger means more dangerous."""

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the classifier."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "DangerClassifier":
        """Load a persisted classifier."""
