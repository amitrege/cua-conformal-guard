# CUA Safety Guard

A calibrated gate that sits between a computer-use agent and the actual screen.
Every proposed action gets a danger score; the action only runs if the score is
below a threshold. The threshold isn't picked by hand — you give the gate
labeled examples and a tolerance like "miss at most 10% of unsafe actions",
and the calibrator picks the threshold so that number is roughly what you get.

```text
agent ─► proposal ─► [danger score] ─► score < t ? ─► run / block / escalate
                                          ▲
                                  calibrated from
                                  labeled examples
```

## 30-second quickstart

```bash
python3 -m unittest discover                              # confirm install

python3 -m cua_guard.cli train-classifier \
  --data examples/train_actions.jsonl \
  --model ./runs/danger.json

python3 -m cua_guard.cli calibrate \
  --data examples/calibration_actions.jsonl \
  --model ./runs/danger.json \
  --guard ./runs/guard.json \
  --alpha 0.10 --mode block

python3 -m cua_guard.cli run-demo --guard ./runs/guard.json
```

The toy agent tries "Search docs" (allowed) then "Delete project" (blocked).

## Where to go next

- **New here?** Read [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — what the library
  does and why the threshold gets calibrated, in plain language.
- **Want to use it?** Follow [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) —
  a step-by-step walkthrough that ends with the gate wired into your own agent.
- **Want to see code?** [`examples/integrate_with_agent.py`](examples/integrate_with_agent.py)
  is a runnable end-to-end script. [`examples/escalation_callback.py`](examples/escalation_callback.py)
  shows the human-in-the-loop path.
- **Going deeper?** [`docs/DESIGN.md`](docs/DESIGN.md) explains the calibration
  math and edge cases. [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) is the
  schema reference for labeled actions, trajectories, and audit logs.

## What's in the box

```text
cua_guard/
  classifiers/   naive-Bayes scorer + a transparent keyword baseline
  conformal/     threshold calibrator (action- and trajectory-level)
  runtime/       guard, agent wrapper, toy environment, episode runner
  adapters/      tiny translators for Playwright / Selenium / OSWorld / native JSON
  evaluation.py  held-out metrics: missed unsafe rate, FPR, per-category risk
  audit.py       JSONL audit traces of every guard decision
  cli.py         train, calibrate, run-demo, evaluate, inspect-guard
examples/        labeled actions, trajectories, runnable integration scripts
docs/            concepts, walkthrough, design notes, data format
tests/           unit tests; `python3 -m unittest discover`
```

No torch, no GPU, no network. Pure Python on the standard library.

## What this gives you and what it doesn't

Holds the expected miss rate for unsafe actions at or below `alpha`, on data
drawn the same way as your calibration set, averaged over fresh samples.

It does **not**:
- say anything about a single action.
- cover harms you didn't label.
- survive a big shift between calibration and deployment — recalibrate.
- give a useful threshold from a tiny calibration set (the CLI warns when this
  happens).

See [`docs/CONCEPTS.md`](docs/CONCEPTS.md#what-the-guarantee-covers) for the
caveats in plain language.
