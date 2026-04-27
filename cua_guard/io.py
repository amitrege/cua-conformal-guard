"""JSON and JSONL helpers for datasets and saved guard artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from cua_guard.types import LabeledAction, LabeledTrajectory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def load_labeled_actions(path: str | Path) -> list[LabeledAction]:
    return [LabeledAction.from_dict(record) for record in read_jsonl(path)]


def load_labeled_trajectories(path: str | Path) -> list[LabeledTrajectory]:
    return [LabeledTrajectory.from_dict(record) for record in read_jsonl(path)]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
