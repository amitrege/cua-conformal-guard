# Getting Started

A hands-on walkthrough. By the end you'll have a calibrated guard, an audit
log, an evaluation report, and a working pattern for plugging the gate into
your own agent.

This doc assumes you've skimmed [`CONCEPTS.md`](CONCEPTS.md) — at minimum, you
know that `alpha` is a target miss rate and that the calibrator picks a
threshold from labeled examples.

Time budget: ~10 minutes of typing, plus however long you want to poke at the
artifacts.

---

## 0. Setup

```bash
git clone <this repo>
cd cua_conformal_guard
python3 -m unittest discover    # ~1s, confirms install
mkdir -p runs                   # all artifacts will land here
```

No dependencies to install. The library is pure standard library.

---

## 1. Train the danger scorer

The scorer's job is `(observation, action) → [0, 1]` where larger means more
dangerous. The bundled one is naive Bayes over the rendered action+context
text — small, transparent, no GPU. You'll likely want to swap it out later
(see [§7 Swapping the scorer](#7-swapping-the-scorer)).

```bash
python3 -m cua_guard.cli train-classifier \
  --data examples/train_actions.jsonl \
  --model ./runs/danger.json
```

What just happened: `examples/train_actions.jsonl` has 16 labeled actions.
Each line is one `(observation, action, unsafe)` example. The CLI fits a
multinomial naive Bayes over tokenized action+context text and writes the
fitted model to `./runs/danger.json` as JSON (no pickles).

Peek at the artifact if you're curious:

```bash
python3 -c "import json; d=json.load(open('runs/danger.json')); print('vocab size:', len(d['vocabulary']))"
```

---

## 2. Calibrate the threshold

Now we hand the scorer a held-out labeled set and a target risk, and the
calibrator picks a threshold.

```bash
python3 -m cua_guard.cli calibrate \
  --data examples/calibration_actions.jsonl \
  --model ./runs/danger.json \
  --guard ./runs/guard.json \
  --alpha 0.10 \
  --mode block
```

Expected output (numbers will vary slightly):

```text
calibrated threshold=0.9439 alpha=0.1000 conservative_risk=0.0556 feasible=True -> ./runs/guard.json
```

Read each field:

- `threshold` — the `t` to compare scores against.
- `alpha` — what you asked for.
- `conservative_risk` — the `(L+1)/(n+1)` value at the chosen threshold. This
  is what the calibrator actually compared to `alpha`.
- `feasible=True` — there exists a threshold whose conservative risk fits
  inside `alpha`. If you ever see `feasible=False`, your `alpha` is tighter
  than your calibration set can support — get more data or relax `alpha`.

The bundled artifact `./runs/guard.json` contains the threshold, the mode, a
pointer to the classifier, and the full calibration result for audit.

```bash
python3 -m cua_guard.cli inspect-guard --guard ./runs/guard.json | head -30
```

### Try a tighter and a looser alpha

The threshold isn't a free parameter — it's a function of `alpha` and your
data. To feel this, run the same command with `--alpha 0.50` and `--alpha
0.05` and watch the threshold move. Tighter `alpha` ⇒ stricter gate ⇒ smaller
threshold.

If the CLI prints a `warning: calibrated threshold is at or near the
maximum/minimum grid value`, take it seriously. It means the math says
"feasible" but the gate is effectively open or closed. See
[`CONCEPTS.md` §5](CONCEPTS.md#5-what-the-guarantee-covers).

---

## 3. Run the toy episode

A scripted agent proposes "click Search docs", then "click Delete project",
then "done". The first should be allowed; the second should be blocked.

```bash
python3 -m cua_guard.cli run-demo \
  --guard ./runs/guard.json \
  --audit-log ./runs/demo_trace.jsonl
```

You'll see a JSON dump of the episode result. Two things to verify:

- `stopped_by_guard: true` — the gate intervened.
- `unsafe_event: false` — the toy environment did *not* run the destructive
  action.

The audit log has one line per guard decision:

```bash
wc -l ./runs/demo_trace.jsonl
python3 -m json.tool < <(head -1 ./runs/demo_trace.jsonl)
```

Each record carries the observation, action, score, threshold, decision,
classifier metadata, guard metadata, and (for evaluation traces) labels. This
is what you'll grep through when something looks wrong in production.

---

## 4. Evaluate on held-out data

`run-demo` proves the wiring works. To get *numbers* you can put in a report,
use `evaluate`:

```bash
python3 -m cua_guard.cli evaluate \
  --guard ./runs/guard.json \
  --data examples/test_actions.jsonl \
  --output ./runs/eval.json \
  --audit-log ./runs/eval_trace.jsonl
```

The report includes:

- `missed_unsafe_rate` — the thing `alpha` is bounding. Should be ≤ `alpha` in
  expectation, modulo small-sample noise.
- `false_positive_rate` — fraction of safe actions the gate intervened on.
- `intervention_rate` / `block_rate` / `escalation_rate` — what fraction of
  the test set the gate touched and how.
- `risk_by_harm_category` — per-category miss rate, computed from the
  `harm_categories` field on each labeled action.
- `boundary_warning` — propagated from calibration if relevant.
- `distribution_warnings` — fires when test scores fall outside the
  calibration score range or the means drift apart.

A toy 5-row test set is too small to read the rates literally. The point of
the example is the *shape* of the report and the per-record audit trace.

---

## 5. Wire the gate into your own agent

This is the integration that matters. The library doesn't care how your agent
decides what to do — it only needs the proposal right before execution.

```python
from cua_guard.runtime.guard import ConformalActionGuard

guard = ConformalActionGuard.load_bundle("./runs/guard.json")

while not done:
    observation = your_env.observe()
    proposal = your_agent.propose(observation)   # ActionProposal
    decision = guard.evaluate(proposal)
    if decision.allowed:
        your_env.step(proposal)
    else:
        # decision.decision is "block" or "escalate"
        handle_intervention(decision)
```

`Observation` and `ActionProposal` are dataclasses in `cua_guard.types`. If
your CUA already produces actions in a Playwright/Selenium/OSWorld-shaped
dictionary, use an adapter — see [§6](#6-adapters).

A complete runnable script is in
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

The callback is where a human review step, a Slack message, or a
stricter-but-slower second model would go. Returning `True` / `False` works
too — the runner wraps it.

A worked example is in
[`examples/escalation_callback.py`](../examples/escalation_callback.py).

---

## 6. Adapters

The library doesn't know about browsers or desktops. Adapters translate
host-shaped action dicts into `ActionProposal`. Bundled and dependency-free:

- `JsonActionAdapter` — this repo's native schema.
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

Adapters never execute anything. They just shape data so the gate can score
it. If your CUA stack has a custom format, write a tiny subclass of
`ActionAdapter` — there's a 30-line example in
`cua_guard/adapters/json_adapter.py`.

---

## 7. Swapping the scorer

The bundled naive-Bayes scorer is good enough for tests, examples, and
bootstrapping. Real deployments should swap in something stronger (a finetuned
text classifier, an LLM-as-judge, a multimodal model that sees screenshots).

The contract is `DangerClassifier` in `cua_guard/classifiers/base.py`:

```python
class DangerClassifier(ABC):
    def fit(self, records): ...
    def score(self, proposal) -> float: ...   # in [0, 1], larger = more dangerous
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

The guard only ever calls `score(proposal)`. Everything else is for training
and persistence. Override `score_batch` when you have a backend that
vectorizes. Override `metadata()` to return model name, data version, and
score range — it ends up in audit logs.

Whatever scorer you ship, **recalibrate** when it changes. The threshold is a
number on that scorer's output; a new scorer needs a new threshold.

---

## 8. Trajectory-level calibration

Same flow, but each example is a sequence of steps with one label.

```bash
python3 -m cua_guard.cli calibrate-trajectories \
  --data examples/calibration_trajectories.jsonl \
  --model ./runs/danger.json \
  --guard ./runs/guard_traj.json \
  --alpha 0.25 --mode block
```

The trajectory score is the max action score over the sequence. The runtime
gate still operates step by step, so the threshold drops in unchanged — it
just got calibrated against episode-level labels.

Use this when you can label outcomes ("did this episode delete a project?")
but not individual clicks.

---

## 9. Recalibrating

Recalibrate when any of these change:

- The scorer (any retrain or model swap).
- The action/observation schema (new fields the scorer can read).
- The deployment surface (new app, new prompt style, new locale).
- Your `alpha` target.

The threshold is a function of *the scorer's output distribution* on *your
calibration data*. If either side shifts, the old threshold isn't doing the
work you think it is.

The score-shift warning in `evaluate` is a cheap smoke alarm. It compares
test scores to the calibration score range and means. It's not a proof of no
distribution shift, but it'll catch the obvious cases.

---

## 10. Where to go from here

- Read [`DESIGN.md`](DESIGN.md) for the math behind the calibrator and the
  reasoning behind each warning.
- Read [`DATA_FORMAT.md`](DATA_FORMAT.md) when you start producing your own
  labeled data.
- Skim [`examples/integrate_with_agent.py`](../examples/integrate_with_agent.py)
  and [`examples/escalation_callback.py`](../examples/escalation_callback.py)
  to see the call patterns end to end.
- Run `python3 -m unittest discover` after any code change. The tests are
  fast and cover most of the library surface.
