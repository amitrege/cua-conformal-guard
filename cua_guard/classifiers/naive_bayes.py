"""A dependency-free trainable text danger classifier."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from cua_guard.classifiers.base import DangerClassifier
from cua_guard.io import read_json, write_json
from cua_guard.types import ActionProposal, LabeledAction, Observation


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class NaiveBayesDangerClassifier(DangerClassifier):
    """Multinomial naive Bayes over rendered observation/action text.

    The classifier is intentionally simple and auditable. It is good enough for
    tests, examples, and bootstrapping a conformal guard. Real deployments should
    swap in a stronger classifier while keeping the same `score` interface.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        min_token_count: int = 1,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.alpha = float(alpha)
        self.min_token_count = int(min_token_count)
        self.class_doc_counts = {False: 0, True: 0}
        self.token_counts = {False: Counter(), True: Counter()}
        self.total_tokens = {False: 0, True: 0}
        self.vocabulary: set[str] = set()
        self.model_metadata = dict(model_metadata or {})
        self._fitted = False

    def fit(self, records: list[LabeledAction]) -> "NaiveBayesDangerClassifier":
        if not records:
            raise ValueError("Cannot train danger classifier on an empty dataset.")

        raw_counts = Counter()
        rendered: list[tuple[bool, list[str]]] = []
        for record in records:
            tokens = tokenize(record.proposal.render_for_classifier())
            raw_counts.update(tokens)
            rendered.append((record.unsafe, tokens))

        self.vocabulary = {
            token for token, count in raw_counts.items() if count >= self.min_token_count
        }
        if not self.vocabulary:
            raise ValueError("Training data produced an empty vocabulary.")

        self.class_doc_counts = {False: 0, True: 0}
        self.token_counts = {False: Counter(), True: Counter()}
        self.total_tokens = {False: 0, True: 0}

        for unsafe, tokens in rendered:
            self.class_doc_counts[unsafe] += 1
            filtered = [token for token in tokens if token in self.vocabulary]
            self.token_counts[unsafe].update(filtered)
            self.total_tokens[unsafe] += len(filtered)

        self._fitted = True
        self.model_metadata.update(
            {
                "type": "naive_bayes",
                "training_records": len(records),
                "unsafe_records": self.class_doc_counts[True],
                "safe_records": self.class_doc_counts[False],
                "score_range": [0.0, 1.0],
            }
        )
        return self

    def score(self, proposal: ActionProposal) -> float:
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before scoring.")

        tokens = [token for token in tokenize(proposal.render_for_classifier()) if token in self.vocabulary]
        log_safe = self._log_class_probability(False, tokens)
        log_unsafe = self._log_class_probability(True, tokens)
        return _logistic(log_unsafe - log_safe)

    def _log_class_probability(self, unsafe: bool, tokens: list[str]) -> float:
        total_docs = self.class_doc_counts[False] + self.class_doc_counts[True]
        prior = (self.class_doc_counts[unsafe] + self.alpha) / (total_docs + 2 * self.alpha)
        vocab_size = len(self.vocabulary)
        denominator = self.total_tokens[unsafe] + self.alpha * vocab_size
        logp = math.log(prior)
        counts = Counter(tokens)
        for token, count in counts.items():
            numerator = self.token_counts[unsafe][token] + self.alpha
            logp += count * math.log(numerator / denominator)
        return logp

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "NaiveBayesDangerClassifier":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "naive_bayes",
            "alpha": self.alpha,
            "min_token_count": self.min_token_count,
            "metadata": self.metadata(),
            "class_doc_counts": {
                "safe": self.class_doc_counts[False],
                "unsafe": self.class_doc_counts[True],
            },
            "token_counts": {
                "safe": dict(self.token_counts[False]),
                "unsafe": dict(self.token_counts[True]),
            },
            "total_tokens": {
                "safe": self.total_tokens[False],
                "unsafe": self.total_tokens[True],
            },
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NaiveBayesDangerClassifier":
        model = cls(
            alpha=float(data.get("alpha", 1.0)),
            min_token_count=int(data.get("min_token_count", 1)),
            model_metadata=dict(data.get("metadata", {})),
        )
        doc_counts = data["class_doc_counts"]
        token_counts = data["token_counts"]
        total_tokens = data["total_tokens"]
        model.class_doc_counts = {
            False: int(doc_counts["safe"]),
            True: int(doc_counts["unsafe"]),
        }
        model.token_counts = {
            False: Counter(token_counts["safe"]),
            True: Counter(token_counts["unsafe"]),
        }
        model.total_tokens = {
            False: int(total_tokens["safe"]),
            True: int(total_tokens["unsafe"]),
        }
        model.vocabulary = set(data["vocabulary"])
        model._fitted = True
        return model

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "type": "naive_bayes",
            "score_range": [0.0, 1.0],
            "alpha": self.alpha,
            "min_token_count": self.min_token_count,
            "vocabulary_size": len(self.vocabulary),
        }
        metadata.update(self.model_metadata)
        return metadata


def _logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def proposal_from_text(screen: str, action: str) -> ActionProposal:
    """Convenience helper used by tests and notebooks."""

    return ActionProposal(
        observation=Observation(text=screen),
        action_type="text",
        target=action,
    )
