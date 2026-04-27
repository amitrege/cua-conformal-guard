"""Load classifier artifacts without exposing callers to concrete classes."""

from __future__ import annotations

from pathlib import Path

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.classifiers.naive_bayes import NaiveBayesDangerClassifier
from cua_guard.classifiers.rules import KeywordDangerScorer
from cua_guard.io import read_json


def load_classifier(path: str | Path) -> DangerClassifier:
    data = read_json(path)
    classifier_type = data.get("type")
    if classifier_type == "naive_bayes":
        return NaiveBayesDangerClassifier.from_dict(data)
    if classifier_type == "keyword":
        return KeywordDangerScorer.load(path)
    raise ValueError(f"Unsupported classifier artifact type: {classifier_type!r}")
