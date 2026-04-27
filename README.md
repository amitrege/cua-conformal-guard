# CUA Safety Guard

A gate that sits between a computer-use agent and the actual screen. Every
proposed action gets a danger score, and the action only runs if the score is
below a threshold. The threshold isn't picked by hand — you give the gate
labeled examples and a tolerance like "miss at most 10% of unsafe actions",
and the calibrator picks the threshold so that number is roughly what you get.

## Why bother

Real CUA actions hit real screens. Wrong click, wrong file deleted. The usual
options:

- No safety net.
- A list of forbidden words. Fragile — "delete" on a draft note is fine,
  "delete" on a settings page isn't.
- Ask another model "is this safe?" and pick a threshold by guessing.

This is the third option but the threshold is calibrated. You provide labeled
actions, choose `alpha`, and the gate is set so it lets through at most about
`alpha` of the unsafe ones on data drawn the same way as your labels.

## Three commands

```bash
# Confirm the install works.
python3 -m unittest discover

# Train a text classifier on the example actions.
python3 -m cua_guard.cli train-classifier \
  --data examples/train_actions.jsonl \
  --model ./runs/danger.json

# Calibrate the threshold. alpha=0.10 means: target a 10% missed-unsafe rate.
python3 -m cua_guard.cli calibrate \
  --data examples/calibration_actions.jsonl \
  --model ./runs/danger.json \
  --guard ./runs/guard.json \
  --alpha 0.10 \
  --mode block

# Run the toy episode against the gate.
python3 -m cua_guard.cli run-demo --guard ./runs/guard.json
```

The toy agent tries "Search docs" (allowed) then "Delete project" (blocked,
episode ends).

For trajectory-level calibration — each example is a sequence of steps with one
label — use `calibrate-trajectories`. The trajectory score is the largest
action score in the sequence:

```bash
python3 -m cua_guard.cli calibrate-trajectories \
  --data examples/calibration_trajectories.jsonl \
  --model ./runs/danger.json \
  --guard ./runs/guard_traj.json \
  --alpha 0.25 \
  --mode block
```

Pass `--approve-escalations` to `run-demo` to exercise the escalation path with
an auto-approver.

## Block or escalate

Pick a `--mode` at calibration time:

- `block` — above-threshold action stops the episode.
- `escalate` — above-threshold action calls a function you supply. It returns
  approved or denied. No function provided means denied.

```python
from cua_guard.runtime.runner import EscalationResolution, run_episode

def review(decision):
    print(f"above threshold: {decision.proposal.action_type} {decision.proposal.target}")
    print(f"score {decision.score:.2f} >= threshold {decision.threshold:.2f}")
    return EscalationResolution(approved=False, reason="human denied")

result = run_episode(agent, env, guard, on_escalate=review)
```

That callback is where you'd put a human review step, a Slack message, or a
stricter second model.

## Wiring it into your own agent

The gate doesn't care how your agent decides what to do. It needs the current
observation, the proposed action, and a calibrated guard.

```python
from cua_guard.runtime.guard import ConformalActionGuard

guard = ConformalActionGuard.load_bundle("./runs/guard.json")

while not done:
    observation = your_env.observe()
    proposal = your_agent.propose(observation)
    decision = guard.evaluate(proposal)
    if decision.allowed:
        your_env.step(proposal)
    else:
        # decision.decision is "block" or "escalate"
        ...
```

`Observation` and `ActionProposal` are dataclasses in `cua_guard.types`. JSON
schema in `docs/DATA_FORMAT.md`.

## How the threshold gets picked

The calibrator walks candidate thresholds from strict to loose. At each
candidate it computes the miss rate on your labeled data, then inflates it
slightly: `(misses + 1) / (n + 1)` instead of `misses / n`. The `+1` accounts
for the worst case from one fresh test point. The chosen threshold is the
loosest one whose inflated rate stays at or below `alpha`.

That `+1` is what turns "what we saw on the calibration set" into "what we
should expect on a fresh point". Derivation in `docs/DESIGN.md`.

If the chosen threshold lands at the edge of the search grid, the CLI prints a
warning. Top edge: gate is essentially open. Bottom edge: gate is essentially
closed, or `alpha` is tighter than your data can support. Either way the number
isn't doing the work you want.

## What this does

Holds the expected miss rate for unsafe actions at or below `alpha`, on data
drawn the same way as your calibration set, averaged over fresh samples.

## What it doesn't

- Say anything about a single action.
- Cover harms you didn't label. If only deletions are marked unsafe, leaks
  aren't protected.
- Survive a big shift between calibration and deployment. Recalibrate when the
  apps or contexts change.
- Give a useful threshold from a small calibration set — that's what the
  boundary warning is for.

## Layout

```text
cua_guard/
  classifiers/      naive-Bayes scorer and a keyword baseline
  conformal/        threshold calibrator
  runtime/          guard, agent wrapper, toy environment, episode runner
  cli.py            train, calibrate, inspect, run-demo
examples/           labeled actions and trajectories used by the quickstart
tests/              unit tests; `python3 -m unittest discover`
docs/               design notes and data format reference
```

## Swapping the scorer

The bundled scorer is a naive-Bayes over text tokens — no torch, no GPU. To
use a real model, subclass `DangerClassifier` and implement `fit`, `score`,
`save`, `load`. The guard only ever calls `score(proposal) -> float in [0, 1]`,
larger means more dangerous.

## Labels

Example data covers deletes, sends, transfers, secret exposure, and a few
obviously safe things. For real use, label the actions your agent actually
proposes in the contexts it actually sees. Schema and labeling tips in
`docs/DATA_FORMAT.md`.
