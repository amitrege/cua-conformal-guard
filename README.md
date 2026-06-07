# cua-conformal-guard

A safety filter for computer-use agents. The agent proposes an action, the
filter scores it, and anything above a threshold gets blocked (or escalated)
before the environment runs it.

The threshold is not chosen by hand. You provide labeled actions and a
target miss rate, and the library uses conformal prediction to derive the
cutoff. On fresh data drawn the same way as the calibration set, the
expected fraction of unsafe actions that get through is at most `alpha`.

```text
agent → proposal → score → compare to threshold → run or block
                                 ↑
                       picked from labeled data
```

## Install

```bash
pip install -e .
```

Python 3.10+. Standard library only.

## Quickstart

```bash
python3 -m cua_guard.cli train-classifier \
    --data examples/train_actions.jsonl \
    --model runs/danger.json

python3 -m cua_guard.cli calibrate \
    --data examples/calibration_actions.jsonl \
    --model runs/danger.json \
    --guard runs/guard.json \
    --alpha 0.10 --mode block

python3 -m cua_guard.cli run-demo --guard runs/guard.json
```

This trains the bundled naive-Bayes scorer on 16 examples, calibrates a
threshold at alpha=0.10, and runs a scripted agent that proposes "Search
docs" (allowed) and then "Delete project" (blocked).

## Using it from Python

```python
from cua_guard import (
    ConformalActionGuard,
    GCRCThresholdCalibrator,
    NaiveBayesDangerClassifier,
)
from cua_guard.io import load_labeled_actions

classifier = NaiveBayesDangerClassifier().fit(load_labeled_actions("train.jsonl"))
calibration = GCRCThresholdCalibrator(alpha=0.10).calibrate_actions(
    classifier, load_labeled_actions("calibration.jsonl"),
)
guard = ConformalActionGuard.from_calibration(classifier, calibration, mode="block")

# inside your agent loop
decision = guard.evaluate(proposal)
if decision.allowed:
    env.step(proposal)
```

## Layout

| Path | Contents |
|---|---|
| `cua_guard/classifiers/` | Naive-Bayes scorer and a keyword baseline, plus the interface to plug in your own. |
| `cua_guard/conformal/`   | The calibrator, for action- and trajectory-level labels. |
| `cua_guard/runtime/`     | The guard, the episode runner, and a toy environment used in the demo. |
| `cua_guard/adapters/`    | Translators for Playwright, Selenium, OSWorld, and the native JSON schema. |
| `cua_guard/evaluation.py`| Metrics on held-out data: miss rate, false positive rate, per-category risk. |
| `cua_guard/audit.py`     | JSONL audit traces, one record per guard decision. |
| `cua_guard/cli.py`       | `train-classifier`, `calibrate`, `calibrate-trajectories`, `run-demo`, `evaluate`, `inspect-guard`. |

## Docs

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — what the gate does and what the bound says.
- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) — train, calibrate, evaluate, and wire it into your agent.
- [`docs/DESIGN.md`](docs/DESIGN.md) — the calibration math and the warnings the CLI prints.
- [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) — JSON schemas for labeled data, adapter input, and audit logs.

Two runnable examples in [`examples/`](examples/):
[`integrate_with_agent.py`](examples/integrate_with_agent.py) for the
end-to-end pattern, and
[`escalation_callback.py`](examples/escalation_callback.py) for a
human-in-the-loop variant.

## Tests

```bash
python3 -m unittest discover    # or: pytest
```

## License

MIT.
