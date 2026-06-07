# Design

Notes for readers who have already skimmed
[`CONCEPTS.md`](CONCEPTS.md), have the
[walkthrough](GETTING_STARTED.md) working, and want to understand the
calibrator internals: the math, the search procedure, and the rationale
behind each warning.

## The pieces

**Danger scorer.** A function `score: (observation, action) → [0, 1]`. The
library ships two implementations, a naive-Bayes model and a keyword
baseline. The guard only ever calls `score(proposal)`; scorers can also
expose `metadata()` for the audit log.

**Action schema.** `Observation` and `ActionProposal` are the library's
boundary. Between them they hold text, OCR, accessibility tree,
screenshot references, app and window data, the raw agent message, the
parsed executor command, coordinates, and arbitrary `target_metadata`.
The schema is wider than the toy demo requires because production CUAs
use varied action formats.

**Adapters.** Translators from host action formats into `ActionProposal`:

```text
host runtime action → adapter → ActionProposal → guard
```

They reshape data; they don't run browsers or desktops.

**Threshold `t`.** A single number in `[0, 1]`. The decision rule is:

```text
allow if score(observation, action) < t
otherwise block or escalate
```

Smaller `t` blocks more, larger `t` allows more.

**Labeled examples.** Each labeled action is `(observation, action,
unsafe_bool)`. Training uses one set and calibration uses another. The
two must not overlap, because reusing the scorer's training data for
calibration invalidates the bound.

**Audit records.** Every runtime or evaluation decision can be written
as a line of JSONL: observation, proposed action, score, threshold,
decision, classifier metadata, guard metadata, and labels when they
exist. The trace is the primary debugging surface when a decision needs
to be revisited.

## Calibration

The safety loss is "an unsafe action got through":

```text
L(t, unsafe, score) = 1 if (unsafe and score < t) else 0
```

The empirical miss rate on the calibration set is:

```text
empirical(t) = (1/n) * sum_i L(t, unsafe_i, score_i)
```

Picking the largest `t` with `empirical(t) ≤ alpha` is equivalent to
hand-tuning the threshold on the calibration set, and provides no
guarantee on a fresh sample.

The library uses an inflated risk instead:

```text
inflated(t) = (sum_i L(t, unsafe_i, score_i) + B) / (n + 1)
```

where `B = 1` is the worst possible single-loss value. It picks the
largest `t` with `inflated(t) ≤ alpha`.

The `+ B` and `+ 1` together cover the worst case on the next fresh
point: an unsafe action that the gate would have allowed. With that
adjustment, the threshold's bound carries over from the calibration set
to fresh samples, and the expected miss rate (averaged over fresh
samples) is at most `alpha`.

## Infeasible alpha

If `alpha < 1 / (n + 1)`, no threshold can satisfy the bound, because
the inflation term alone exceeds `alpha` even on a perfectly safe
calibration set. The library returns `feasible = False` and falls back
to the strictest threshold; the fix is more calibration data or a
looser `alpha`.

## Boundary thresholds

The calibrator searches a grid: uniform points `0/200, 1/200, ...,
200/200` plus points clustered around each observed score. If the chosen
threshold sits at the largest grid value, almost nothing the scorer can
produce will exceed it, so the gate is effectively open. If it sits at
the smallest grid value, the gate is effectively closed (or calibration
was infeasible from the start). The math can report `feasible` in both
cases, so the CLI prints a warning when a boundary threshold is
selected.

## Search direction

The calibrator walks thresholds from strict to loose and tracks the
running maximum of the inflated risk. A candidate is only accepted if
every stricter threshold also met the budget.

For the default miss-rate loss this is equivalent to walking until the
budget breaks. The running-max form also handles losses that are not
monotone in `t`, such as one that penalizes intervening on too many safe
actions; only the loss function changes, not the calibration code
around it.

## Evaluation

`evaluate_labeled_actions` takes a guard and a held-out labeled JSONL
file. It does not execute anything; it reports what the guard would have
allowed, blocked, or escalated.

The report contains:

- missed unsafe rate
- false positive rate
- intervention rate, block rate, escalation rate
- risk by harm category
- the boundary warning from calibration, if any
- score-shift warnings from comparing evaluation scores against the
  calibration summary

The score-shift check is approximate: it compares the test score range
and mean against the calibration summary, which is enough to catch
obvious drift.

## Trajectory-level calibration

The same construction works at the trajectory level. Each labeled example
is a sequence of actions with one safety label, and the trajectory score
is:

```text
trajectory_score(steps) = max(score(step) for step in steps)
```

"Block the trajectory" reduces to "block as soon as any step crosses
the threshold," which is what the runtime guard already does step by
step. A trajectory-calibrated threshold works in the action-level gate
as is.

## What the bound doesn't say

- Nothing about a specific allowed action.
- Nothing about harms that weren't labeled.
- Nothing about deployments that have drifted away from the calibration
  data.
- Nothing that survives a scorer swap or retrain.

Harm categories and distribution warnings make these failure modes
visible, but they do not strengthen the bound.

## Where the gate plugs in

The integration is one extra check before each action runs:

```python
proposal = agent.propose(observation)
decision = guard.evaluate(proposal)
if decision.decision == "allow":
    env.step(proposal)
elif decision.decision == "escalate":
    if reviewer_approves(decision):
        env.step(proposal)
else:
    abort(decision)
```

`run_episode` implements this pattern in the bundled runner, so the
demo and a real CUA stack share the same control flow.
