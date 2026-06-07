# Concepts

Why the threshold gets calibrated and what the resulting bound covers. To
get something running, skip to [`GETTING_STARTED.md`](GETTING_STARTED.md).

## The problem

A computer-use agent reads a screen and produces an action. The action runs
in a real environment, so the wrong click can delete a project, send an
email, or transfer money. You want something between the agent and the
environment that catches the bad ones.

Roughly:

```
score = danger(observation, action)
allow if score < threshold
```

Two open questions: where does `danger` come from, and how do you pick
the threshold? This library focuses on the threshold; it ships a basic
classifier so you can experiment, but most users will want to replace it
with something stronger.

## Picking the threshold by hand doesn't work

Setting `threshold = 0.5` because that looks reasonable doesn't say
anything about a miss rate. Even if 0.5 worked yesterday, retraining the
classifier or running it on a slightly different population changes what
0.5 corresponds to.

A blocklist of risky verbs (delete, send, transfer) has the opposite
problem: it catches the verb and ignores the context. Clicking "Delete"
on a throwaway draft is fine; the same click on a settings page is
catastrophic.

What you want is a procedure that tunes the threshold to a target rate on
labeled data and gives you a number you can put in a report ("we miss at
most 10% of unsafe actions on average").

## What calibration provides

The calibrator needs three things: a trained scorer that maps
`(observation, action)` to a number in `[0, 1]`, a held-out set of labeled
actions, and a target miss rate `alpha`. It returns a threshold with this
property:

> On fresh samples drawn the same way as the calibration set, the expected
> fraction of unsafe actions the gate allows through is at most `alpha`.

Caveats:

- The bound is an expectation over fresh samples. It does not say
  anything about any one action; that action either got through or it
  didn't.
- The "drawn the same way" qualifier matters. New apps, new prompt
  templates, or a new locale put you outside the assumption, and the
  bound no longer applies.
- The bound only covers harms you labeled. If you labeled deletes but
  not credential leaks, the gate has nothing to say about credential
  leaks.

## The math

The loss is 1 when an unsafe action gets through and 0 otherwise. The
empirical miss rate on the calibration set at threshold `t` is:

```
empirical(t) = sum(loss_i) / n
```

The library uses an inflated version:

```
inflated(t) = (sum(loss_i) + 1) / (n + 1)
```

It picks the largest `t` where `inflated(t) <= alpha`. The extra `+1`s
account for the worst case on the next fresh point: a missed unsafe
action. With that adjustment in the formula, the threshold's bound
carries over from the calibration set to fresh samples.

One consequence: if `alpha < 1 / (n + 1)`, no threshold can satisfy the
bound, because the inflation term alone exceeds `alpha`. The library
reports `feasible = False`, and the fix is to grow the calibration set
or loosen `alpha`.

## Block or escalate

When a score crosses the threshold, the gate either blocks the action and
stops the episode or escalates it to a callback. Pick `block` for
unattended automation, where a missed unsafe action is much costlier than
an extra stop. Pick `escalate` when there is a reviewer — a human, or a
slower and stricter model — that can resolve borderline cases. The
escalation rate becomes a tuning knob: choose `alpha` so the rate matches
the reviewer's capacity.

The mode is fixed at calibration time and saved in the guard bundle.

## Trajectory-level labels

Sometimes the natural label isn't on a single action but on the whole
sequence ("did this episode delete a project?"). The library handles this
by scoring a trajectory as its maximum step score:

```
trajectory_score(steps) = max(score(step) for step in steps)
```

"Block the trajectory" reduces to "block as soon as any step crosses
the threshold," which is what the runtime guard already does step by
step. A threshold calibrated on trajectory labels can be used in the
same runtime gate without changes.

## Warnings the CLI prints

The calibrator searches a finite grid of thresholds. If the chosen
threshold lands at the top of that grid, almost nothing the model can
produce will exceed it, so the gate is effectively open. If it lands at
the bottom, the gate is effectively closed. The math can still report
`feasible` in both cases, so the CLI prints a `boundary_warning`
whenever a boundary threshold is chosen.

The evaluator also compares evaluation scores to the calibration score
summary. If evaluation scores fall outside the calibration range, or the
means differ by more than two calibration standard deviations, it raises
a `distribution_warning`. The check is approximate; it catches the
obvious cases without claiming to detect every form of drift.

## When to recalibrate

Anything that changes the scorer's output distribution or the data the
system sees at runtime invalidates the threshold:

- retraining or swapping the scorer
- adding fields to observations or actions that the scorer reads
- a new app, prompt template, or locale
- a new value of `alpha`

## Code map

| Concept | File |
|---|---|
| Action and observation types | `cua_guard/types.py` |
| Scorer interface | `cua_guard/classifiers/base.py` |
| Bundled scorers | `cua_guard/classifiers/` |
| Calibrator | `cua_guard/conformal/gcrc.py` |
| Trajectory calibration | `cua_guard/conformal/trajectory.py` |
| Runtime gate | `cua_guard/runtime/guard.py` |
| Episode runner | `cua_guard/runtime/runner.py` |
| Host adapters | `cua_guard/adapters/` |
| Evaluation metrics | `cua_guard/evaluation.py` |
| Audit traces | `cua_guard/audit.py` |
| CLI | `cua_guard/cli.py` |

## Glossary

- **CUA** — computer-use agent. A model that produces GUI actions.
- **Action proposal** — an `(observation, action)` pair that hasn't run yet.
- **Danger scorer** — function from a proposal to a number in `[0, 1]`.
- **Threshold `t`** — the cutoff; `score < t` means allow.
- **`alpha`** — target miss rate.
- **Calibration set** — labeled actions held out from training.
- **Conservative risk** — the `(sum + 1) / (n + 1)` value the calibrator
  compares to `alpha`.
- **Feasible** — at least one threshold satisfies the bound.
- **Boundary threshold** — the chosen `t` is at the top or bottom of the
  search grid; the gate is open or closed regardless of the reported risk.
- **Audit trace** — JSONL of every guard decision.

## Next

- [`GETTING_STARTED.md`](GETTING_STARTED.md) for the hands-on walkthrough.
- [`DESIGN.md`](DESIGN.md) for the math and the warnings in more detail.
- [`DATA_FORMAT.md`](DATA_FORMAT.md) for the JSON schemas.
