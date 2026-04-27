# Design Notes

For someone who's read [`CONCEPTS.md`](CONCEPTS.md), has the
[quickstart](GETTING_STARTED.md) running, and wants to know what the
calibrator is doing under the hood — the math, the search procedure, and the
edge cases that drive the warnings.

If you're new to the library, read [`CONCEPTS.md`](CONCEPTS.md) first. This
doc assumes you already know what `alpha` is and why the threshold is
calibrated.

## What the library does

A runtime gate for a CUA that scores every proposed action, blocks or escalates
dangerous ones, and has a knob whose meaning isn't gut feel — it's "we expect
to miss at most `alpha` of unsafe actions on data like our calibration set".

## The pieces

**Danger scorer.** Function from `(observation, action) -> [0, 1]`. Two are
bundled: a naive-Bayes over text and a keyword baseline. Plug in your own.
Scorers can expose metadata for audit logs, but the guard only needs `score`.

**Action schema.** `Observation` and `ActionProposal` are the library boundary.
They can hold text, OCR, accessibility text, screenshot references, app/window
data, raw agent messages, parsed executor commands, coordinates, and target
metadata. The schema is broader than the toy demo because real CUAs expose
different action shapes.

**Adapters.** Small translators from host action formats into `ActionProposal`.
They don't run browsers or desktops. They sit at the edge:

```text
host runtime action -> adapter -> ActionProposal -> guard
```

The current adapters cover native JSON, Playwright-like commands,
Selenium-like commands, and OSWorld-style desktop dictionaries.

**Threshold `t`.** A single number in `[0, 1]`. Decision rule:

```text
allow if score(observation, action) < t
otherwise block or escalate
```

Smaller `t` blocks more. Larger `t` allows more. Picking `t` is the whole
exercise.

**Labeled examples.** Each is `(observation, action, unsafe_bool)`. You need a
training set to fit a scorer and a separate calibration set to pick `t`. They
must not overlap.

**Audit records.** Every runtime or evaluation decision can be written as JSONL:
action, score, threshold, decision, classifier metadata, guard metadata, and
labels when labels exist. This is how you find bad labels, bad action parsing,
and score drift.

## Calibration

The safety loss is "an unsafe action got through":

```text
L(t, unsafe, score) = 1 if (unsafe and score < t) else 0
```

The empirical miss rate on the calibration set at threshold `t`:

```text
empirical(t) = (1/n) * sum_i L(t, u_i, s_i)
```

You could pick the largest `t` with `empirical(t) <= alpha` and call it done.
That's a hand-tuned threshold and gives you nothing about a fresh sample.

What this library uses instead:

```text
inflated(t) = (sum_i L(t, u_i, s_i) + B) / (n + 1)
```

where `B = 1` is the worst possible single loss. The chosen `t` is the largest
one where `inflated(t) <= alpha`.

The `+B` and `+1` together are "what if the next fresh point is the worst case
we could see". With that adjustment, the threshold carries a guarantee: the
expected miss rate on a fresh sample drawn the same way is at most `alpha`.
Expectation is over fresh samples — not a per-action claim.

That `(L + 1) / (n + 1)` swap is the whole trick.

## Infeasible alpha

If `alpha < 1 / (n + 1)`, no threshold can satisfy the bound — even a perfectly
safe calibration set has the inflation term blocking it. The library reports
`feasible = False` and falls back to the strictest threshold. That's the signal
to either get more calibration data or relax `alpha`.

## Boundary thresholds

The calibrator searches a finite grid. If the chosen threshold sits at the
largest grid value, anything the model could realistically output is below it
— the gate is open. At the smallest grid value, the gate is closed, or
calibration was infeasible. The math may say "feasible" in both cases, but
neither is useful behavior. The CLI prints a warning so an open gate doesn't
ship by accident.

## Evaluation reports

The evaluator takes a guard and a held-out labeled JSONL file. It does not
execute actions. It asks: "what would this guard have allowed, blocked, or
escalated?"

Reported metrics:

- missed unsafe rate
- false positive rate
- intervention rate
- block rate
- escalation rate
- risk by harm category
- boundary warnings from calibration
- simple score-shift warnings between calibration and evaluation

The score-shift checks are intentionally simple. They compare evaluation scores
to the calibration score range and mean/std. They are not a proof of no
distribution shift. They are cheap warning lights.

## Trajectory-level calibration

Same setup, but each labeled example is a sequence of actions with one safety
label. The trajectory score is:

```text
trajectory_score(steps) = max(score(step) for step in steps)
```

"Block this trajectory" is "block as soon as any step crosses the threshold".
The runtime guard already does that step by step, so a trajectory-calibrated
threshold drops in for action-level gating.

## Why the search runs in one direction

The calibrator walks thresholds from strict to loose and tracks the running
maximum of the inflated risk. A candidate is only accepted if every stricter
threshold also met the budget. For the default miss-rate loss this is the same
as walking until the budget breaks, but the running-max form keeps working for
losses that aren't monotone in the threshold — say, a future loss that also
penalizes blocking too many safe actions. Same calibration code, different loss
function.

## What it doesn't prove

- Nothing about an individual allowed action.
- Nothing about harms you didn't label. Bad label set, bad gate.
- Nothing about a deployment that's drifted from your calibration.
  Recalibrate.

## Where the code hooks in

One line before action execution:

```python
proposal = agent.propose(observation)
decision = guard.evaluate(proposal)
if decision.decision == "allow":
    environment.step(proposal)
elif decision.decision == "escalate":
    if your_human_or_host_approves(decision):
        environment.step(proposal)
else:
    abort(decision)
```

The guard doesn't know how the agent produced the action. `run_episode` wires
this same pattern through the included runner, so the demo and a real CUA stack
share the control flow.
