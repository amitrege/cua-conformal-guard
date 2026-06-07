# Getting Started

By the end of this walkthrough you will have a calibrated guard, an
audit log, an evaluation report, and the gate wired into your own agent
loop. If you have not seen conformal calibration before, read
[`CONCEPTS.md`](CONCEPTS.md) first; the only piece you really need from
it is that `alpha` is a target miss rate.

## Setup

```bash
git clone <this repo>
cd cua_conformal_guard
python3 -m unittest discover     # sanity check
mkdir -p runs                    # outputs go here
```

Nothing to install beyond Python 3.10+.

## 1. Train the scorer

The scorer maps `(observation, action)` to a number in `[0, 1]`, where
larger means more dangerous. The bundled implementation is a multinomial
naive Bayes over the rendered text; it is small enough to read end to
end and sufficient for this walkthrough. For production use, see
[Swap the scorer](#swap-the-scorer).

```bash
python3 -m cua_guard.cli train-classifier \
    --data examples/train_actions.jsonl \
    --model runs/danger.json
```

`examples/train_actions.jsonl` contains 16 labeled actions. The fitted
model is written to `runs/danger.json` as plain JSON.

## 2. Calibrate a threshold

Pass the scorer a held-out labeled set and a target miss rate:

```bash
python3 -m cua_guard.cli calibrate \
    --data examples/calibration_actions.jsonl \
    --model runs/danger.json \
    --guard runs/guard.json \
    --alpha 0.10 --mode block
```

You should see something like:

```text
calibrated threshold=0.9439 alpha=0.1000 conservative_risk=0.0556 feasible=True -> runs/guard.json
```

- `threshold` is the value scores are compared against.
- `alpha` is the target miss rate you requested.
- `conservative_risk` is the `(L + 1) / (n + 1)` value the calibrator
  compared to `alpha`.
- `feasible=True` means at least one threshold satisfies the bound. If
  it is `False`, `alpha` is tighter than the calibration set can
  support.

`runs/guard.json` is the saved bundle: the threshold, the mode, a pointer
to the classifier, and the full calibration record.

```bash
python3 -m cua_guard.cli inspect-guard --guard runs/guard.json | head -30
```

### Try a few values of alpha

The threshold depends on both `alpha` and the data. Re-run the command
with `--alpha 0.50` and then `--alpha 0.05` to see how the threshold
moves. A tighter `alpha` produces a stricter gate.

A warning of the form `calibrated threshold is at or near the
maximum/minimum grid value` should not be ignored. The math reports a
feasible threshold, but the gate is effectively open or closed. See
[`CONCEPTS.md`](CONCEPTS.md#warnings-the-cli-prints).

## 3. Run the toy episode

A scripted agent proposes "click Search docs", then "click Delete
project", then "done". The first action should be allowed and the second
should be blocked.

```bash
python3 -m cua_guard.cli run-demo \
    --guard runs/guard.json \
    --audit-log runs/demo_trace.jsonl
```

The episode result prints as JSON. Two fields to check:

- `stopped_by_guard: true` — the gate intervened.
- `unsafe_event: false` — the destructive action never ran.

The audit log has one line per guard decision:

```bash
wc -l runs/demo_trace.jsonl
python3 -m json.tool < <(head -1 runs/demo_trace.jsonl)
```

Each record contains the observation, the action, the score, the
threshold, the decision, classifier metadata, guard metadata, and (for
evaluation traces) the labels.

## 4. Evaluate on held-out data

`run-demo` confirms that the wiring works. For metrics you can include
in a report, use `evaluate`:

```bash
python3 -m cua_guard.cli evaluate \
    --guard runs/guard.json \
    --data examples/test_actions.jsonl \
    --output runs/eval.json \
    --audit-log runs/eval_trace.jsonl
```

The report has:

- `missed_unsafe_rate` — the quantity `alpha` bounds. In expectation it
  should sit at or below `alpha`, though small samples can swing it.
- `false_positive_rate` — fraction of safe actions the gate intervened on.
- `intervention_rate`, `block_rate`, `escalation_rate` — how often the
  gate fired, and how.
- `risk_by_harm_category` — per-category miss rate, from each record's
  `harm_categories`.
- `boundary_warning` — propagated from calibration if relevant.
- `distribution_warnings` — fires when test scores fall outside the
  calibration range or the means drift apart.

Five test rows is too small a sample to read these rates literally; the
goal of this step is to see the shape of the report and the per-record
audit trace.

## 5. Wire the gate into your own agent

The library is agnostic to how your agent chooses actions; it only
needs the proposal before that action runs.

```python
from cua_guard.runtime.guard import ConformalActionGuard

guard = ConformalActionGuard.load_bundle("runs/guard.json")

while not done:
    observation = env.observe()
    proposal = agent.propose(observation)        # ActionProposal
    decision = guard.evaluate(proposal)
    if decision.allowed:
        env.step(proposal)
    else:
        handle_intervention(decision)            # block or escalate
```

`Observation` and `ActionProposal` are defined in `cua_guard.types`. If
your CUA already emits Playwright, Selenium, or OSWorld-shaped
dictionaries, use an [adapter](#adapters) to convert them.

A full runnable script is in
[`examples/integrate_with_agent.py`](../examples/integrate_with_agent.py).

### Escalate instead of block

Calibrate with `--mode escalate` and pass an `on_escalate` callback to
`run_episode`:

```python
from cua_guard.runtime.runner import EscalationResolution, run_episode

def review(decision):
    print(f"above threshold: {decision.proposal.action_type} {decision.proposal.target}")
    print(f"score {decision.score:.2f} >= threshold {decision.threshold:.2f}")
    return EscalationResolution(approved=False, reason="human denied")

result = run_episode(agent, env, guard, on_escalate=review)
```

The callback is the natural insertion point for a human review step, a
Slack message, or a slower and stricter model. Returning a plain `bool`
is also supported; the runner wraps it.

See [`examples/escalation_callback.py`](../examples/escalation_callback.py)
for a worked version.

## Adapters

Adapters convert host action dictionaries into `ActionProposal`. Four
are bundled:

- `JsonActionAdapter` — the native schema.
- `PlaywrightActionAdapter` — Playwright-like commands.
- `SeleniumActionAdapter` — Selenium-like commands.
- `OSWorldActionAdapter` — OSWorld desktop dictionaries.

```python
from cua_guard.adapters import PlaywrightActionAdapter

adapter = PlaywrightActionAdapter()
proposal = adapter.to_proposal(
    observation={"app": "browser", "url": page.url, "text": visible_text},
    action={"method": "click", "selector": "button.delete", "text": "Delete project"},
)
decision = guard.evaluate(proposal)
```

Adapters do not execute actions; they reshape them so the gate can
score them. For a custom format, subclass `ActionAdapter`. The JSON
adapter at `cua_guard/adapters/json_adapter.py` is about thirty lines
and serves as a template.

## Swap the scorer

The naive-Bayes scorer is enough for this walkthrough. For production
use, you will want something stronger: a fine-tuned classifier, an LLM
judge, or a multimodal model that reads screenshots.

The interface to satisfy is `DangerClassifier` in
`cua_guard/classifiers/base.py`:

```python
class DangerClassifier(ABC):
    def fit(self, records): ...
    def score(self, proposal) -> float: ...      # in [0, 1]
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

The guard only ever calls `score(proposal)`. Override `score_batch` if
you have a backend that vectorizes. Override `metadata()` to surface the
model name and data version — both end up in the audit log.

A new scorer produces a new score distribution, so the calibrated
threshold is no longer valid; recalibrate before deploying.

## Trajectory-level calibration

If your labels are at the episode level rather than per action, use:

```bash
python3 -m cua_guard.cli calibrate-trajectories \
    --data examples/calibration_trajectories.jsonl \
    --model runs/danger.json \
    --guard runs/guard_traj.json \
    --alpha 0.25 --mode block
```

The trajectory score is the max of its step scores. The runtime gate
still runs step by step, so the resulting threshold works in the same
gate, but it was tuned against episode labels rather than action labels.

Use trajectory calibration when you can label outcomes ("did this
episode delete a project?") but not individual clicks.

## When to recalibrate

- the scorer changed (retrain or swap)
- the action or observation schema changed
- the deployment surface changed (new app, prompt template, locale)
- `alpha` changed

The score-shift warning from `evaluate` compares evaluation scores
against the calibration summary; the check is approximate but catches
the obvious cases.

## Next

- [`DESIGN.md`](DESIGN.md) for the calibrator math and the rationale
  behind each warning.
- [`DATA_FORMAT.md`](DATA_FORMAT.md) for the JSON schemas.
- [`examples/integrate_with_agent.py`](../examples/integrate_with_agent.py)
  and [`examples/escalation_callback.py`](../examples/escalation_callback.py)
  for the end-to-end call patterns.
- `python3 -m unittest discover` after any change.
