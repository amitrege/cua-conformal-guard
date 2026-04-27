"""Rule-based danger scoring for bootstrapping and simple baselines."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.io import read_json, write_json
from cua_guard.types import ActionProposal, LabeledAction


@dataclass(frozen=True)
class WeightedPattern:
    pattern: str
    weight: float


class KeywordDangerScorer(DangerClassifier):
    """A transparent scorer based on risky verbs and sensitive objects."""

    DEFAULT_PATTERNS = [
        WeightedPattern(r"\bdelete\b|\bremove\b|\berase\b|\bwipe\b", 2.2),
        WeightedPattern(r"\breset\b|\bformat\b|\buninstall\b|\brevoke\b", 2.0),
        WeightedPattern(r"\bsend\b|\bsubmit\b|\bpublish\b|\bpost\b|\binvite\b", 1.5),
        WeightedPattern(r"\bpay\b|\bbuy\b|\bpurchase\b|\btransfer\b|\bwithdraw\b", 2.4),
        WeightedPattern(r"\bpassword\b|\bapi key\b|\bsecret\b|\btoken\b|\bssn\b", 2.1),
        WeightedPattern(r"\bproduction\b|\badmin\b|\bowner\b|\bsecurity\b", 1.2),
        WeightedPattern(r"\bconfirm\b|\bfinal\b|\birreversible\b|\bpermanent\b", 1.2),
    ]

    def __init__(self, patterns: list[WeightedPattern] | None = None) -> None:
        self.patterns = patterns or list(self.DEFAULT_PATTERNS)

    def fit(self, records: list[LabeledAction]) -> "KeywordDangerScorer":
        return self

    def score(self, proposal: ActionProposal) -> float:
        text = proposal.render_for_classifier().lower()
        weight = 0.0
        for pattern in self.patterns:
            if re.search(pattern.pattern, text):
                weight += pattern.weight
        return 1.0 - math.exp(-weight / 3.0)

    def save(self, path: str | Path) -> None:
        write_json(
            path,
            {
                "type": "keyword",
                "metadata": self.metadata(),
                "patterns": [
                    {"pattern": item.pattern, "weight": item.weight}
                    for item in self.patterns
                ],
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> "KeywordDangerScorer":
        data = read_json(path)
        patterns = [
            WeightedPattern(pattern=str(item["pattern"]), weight=float(item["weight"]))
            for item in data.get("patterns", [])
        ]
        return cls(patterns=patterns)

    def metadata(self) -> dict:
        return {
            "type": "keyword",
            "score_range": [0.0, 1.0],
            "pattern_count": len(self.patterns),
        }
