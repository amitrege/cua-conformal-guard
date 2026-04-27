# Concepts

For someone who's read the README, knows ML and CUAs, but hasn't worked with
conformal calibration before. By the end you should understand *what* the
library does, *why* the threshold is calibrated instead of guessed, and *what*
the guarantee actually buys you.

If you just want to run something, jump to
[`GETTING_STARTED.md`](GETTING_STARTED.md). If you want the math, jump to
[`DESIGN.md`](DESIGN.md). This doc is the connective tissue.

---

## 1. The problem

A computer-use agent looks at a screen and emits an action: a click, a
keypress, a coordinate, a tool call. That action runs against a real
environment — a real browser, a real desktop, a real shell. If the model
proposes "click *Delete project*" on the wrong page, something gets deleted.
"Send email" on the wrong draft, something gets sent.

You want a *gate* that sits in front of action execution and stops the bad
ones:

```text
                  ┌──────────────────────────────────────────┐
                  │                  GATE                    │
   agent  ─────►  │   score(observation, action) → [0,1]     │  ─────► environment
                  │   if score < t: allow                    │
                  │   else        : block or escalate        │
                  └──────────────────────────────────────────┘
```

The score is "how dangerous does this action look in this context". The
threshold `t` controls the trade-off: lower `t` blocks more (safer, more false
positives), higher `t` allows more (smoother, more misses).

Two sub-problems:

1. **The scorer.** Where does `score(observation, action)` come from?
2. **The threshold.** How do you pick `t`?

This library focuses on (2) and ships a baseline for (1) you can swap out.

---

## 2. Why not just pick a threshold?

The obvious thing is to look at a few examples, eyeball a number like `t = 0.5`,
and ship it. This goes wrong in three ways:

- **It doesn't mean anything.** `0.5` is a number on the scorer's output. If
  the scorer's distribution shifts (new categories, new app, new model), `0.5`
  could correspond to *very* different behavior tomorrow than today. You have
  no reading on how often unsafe actions slip through.
- **It doesn't compose with goals.** You probably care about "miss at most 10%
  of unsafe actions" or "intervene on at most 5% of safe actions". A
  hand-picked threshold doesn't translate to either.
- **It doesn't survive review.** The first question anyone asks about an AI
  safety gate is "what's the false-negative rate?" If the answer is "we picked
  the threshold by looking at a few examples", that's not a defensible answer.

The other obvious thing is a wordlist: block any action containing "delete",
"send", etc. That's brittle in the opposite direction — context-blind. Click
*Delete* on a throwaway draft is fine; click *Delete* on a project settings
page is not. A scorer that sees both action and surrounding screen text can
distinguish these. A wordlist can't.

This library is the calibrated-threshold version: a scorer that uses context,
plus a procedure that picks `t` so a *labeled rate* (like missed-unsafe rate)
holds in expectation on fresh data drawn the same way as your labels.

---

## 3. What "calibrated" actually means

You give the calibrator three things:

- A trained scorer (`score: (obs, action) → [0, 1]`).
- A **calibration set**: labeled actions `(obs_i, action_i, unsafe_i)` that the
  scorer has not seen during training.
- A target risk `alpha ∈ [0, 1]`. Concretely: "I'm willing to miss at most this
  fraction of unsafe actions on data like the calibration set."

The calibrator returns a single number `t` such that:

> **The expected miss rate on a fresh sample, drawn the same way as the
> calibration set, is at most `alpha`.**

Three things to notice:

- It's an **expectation**, not a per-action guarantee. For any single action,
  the gate either let it through or it didn't. The math says: *on average over
  fresh samples*, the miss rate stays at or below `alpha`.
- "**Drawn the same way**" is load-bearing. If the calibration set is a fair
  reflection of what you'll see in deployment, the bound holds. If your
  deployment shifts (new app, new prompt style, different OS), the bound is
  about a world that no longer exists.
- The target is a **labeled rate**, not a property of any single action. The
  guarantee is only as good as what you labeled. If you labeled deletes and
  sends but ignored credential leaks, the gate doesn't protect against
  credential leaks no matter how small `alpha` is.

---

## 4. The trick: `(L + 1) / (n + 1)`

Why does picking `t` from `n` calibration points give you anything about a
fresh `n+1`-th point? Here's the intuition.

Define the loss for an action as 1 if it's unsafe and the gate let it through,
0 otherwise. The empirical miss rate at threshold `t` on the calibration set
is:

```text
empirical(t) = (1/n) * sum_i loss_i(t)
```

You could pick the largest `t` with `empirical(t) ≤ alpha` and call it done.
This is a hand-tuned threshold dressed up in math. It says nothing about a
fresh point.

The library inflates the rate slightly:

```text
inflated(t) = (sum_i loss_i(t) + 1) / (n + 1)
```

The chosen `t` is the largest one where `inflated(t) ≤ alpha`.

Read the formula as: "what if the next fresh point I see is the worst case I
could see — a missed unsafe action that adds 1 to the numerator and 1 to the
denominator?". With that adjustment baked in, the threshold carries the
guarantee that the expected miss rate on a fresh point (drawn the same way) is
at most `alpha`.

That `+1` swap is what turns "what we saw on the calibration set" into "what
we should expect on a fresh point". The full justification is in
[`DESIGN.md`](DESIGN.md#calibration); for everyday use the swap is the whole
trick.

A consequence worth understanding: if `alpha < 1 / (n + 1)`, no threshold can
satisfy the bound — even a perfectly safe calibration set has the inflation
term blocking it. The library reports `feasible = False` and the CLI warns
you. The fix is more calibration data or a looser `alpha`.

---

## 5. What the guarantee covers

In one sentence: **the expected fraction of unsafe actions that the gate lets
through, averaged over fresh samples drawn the same way as the calibration
set, is at most `alpha`**.

Things that follow from this:

- Tighter `alpha` → stricter gate → more false positives. There is no free
  lunch.
- More calibration data → less inflation penalty → tighter `alpha` becomes
  feasible.
- The bound is about *how often* unsafe actions get through. It does not
  promise *which* actions get through.

Things that **do not** follow:

- **Per-action claims.** "This specific click is safe with probability X" — no.
  The bound is over distributions, not points.
- **Out-of-distribution coverage.** If the agent starts proposing actions in
  apps or contexts not represented in calibration, you're outside the
  assumption. Recalibrate.
- **Coverage of unlabeled harms.** The loss only counts the labels you
  provided. A leak you didn't label is a leak the gate can't account for.
- **Survivability under scorer change.** If you swap the scorer or retrain it,
  the threshold needs to be recalibrated. The threshold is a number on *that
  scorer's* output, not a universal "safe" line.

The library exposes two warnings to make these failure modes visible:

- **Boundary warning** when the chosen `t` lands at the top or bottom of the
  search grid. Top: the gate is essentially open. Bottom: the gate is
  essentially closed (or calibration was infeasible). The math may say
  "feasible" in both cases, but neither is useful behavior.
- **Score-shift warning** when evaluation scores fall outside the calibration
  score range, or when the evaluation mean is more than two calibration
  standard deviations away from the calibration mean. This is a cheap
  smoke alarm for distribution shift, not a proof.

---

## 6. Block vs. escalate

When a score crosses the threshold, the gate has to do *something*. Two modes:

- **`block`** — the action is rejected and the episode stops. Use this when
  the cost of a missed unsafe action is much higher than the cost of a
  spurious stop. Demos, automation pipelines, anything that runs unattended.
- **`escalate`** — the action goes to a callback you supply. The callback
  returns `approved` or `denied`. Use this when there's a human or a
  stricter-but-slower model that can review borderline cases. The escalation
  rate becomes a knob: you tune `alpha` so the rate is what your reviewer can
  actually keep up with.

The choice of mode is set at calibration time and persisted in the guard
bundle. Either way, the threshold is the same — what changes is the
consequence when it fires.

---

## 7. Action-level vs. trajectory-level calibration

Most of the discussion above treats each `(obs, action, unsafe)` as one
calibration unit. That's the action-level case.

Sometimes the natural label is at the **trajectory** level: a sequence of
actions, with one safety label for the whole sequence. ("Did this episode
delete a project?", not "is this individual click safe?")

Same calibration, with one change: the trajectory score is the **maximum**
step score:

```text
trajectory_score(steps) = max(score(step) for step in steps)
```

Read it as "block this trajectory" = "block as soon as any step crosses the
threshold". The runtime gate already does that step by step, so a
trajectory-calibrated threshold drops in for action-level gating without code
changes. You just calibrate against trajectory labels instead of action
labels, and the bound is now over trajectories.

Use trajectory-level calibration when:

- You only have outcome-level labels (the user can label the episode but not
  every click).
- The harm is sequential — no individual click is dangerous, but the sequence
  ends somewhere you didn't want to go.

---

## 8. The pieces in this repo

A quick map back to code, so the rest of the docs make sense:

| Concept | Lives in |
|---|---|
| Action and observation schema | `cua_guard/types.py` |
| Danger scorer interface | `cua_guard/classifiers/base.py` |
| Bundled scorers (naive-Bayes, keyword) | `cua_guard/classifiers/` |
| The calibrator (the `(L+1)/(n+1)` thing) | `cua_guard/conformal/gcrc.py` |
| Trajectory calibration helper | `cua_guard/conformal/trajectory.py` |
| Runtime gate | `cua_guard/runtime/guard.py` |
| Episode runner with escalation hook | `cua_guard/runtime/runner.py` |
| Adapters for host CUA formats | `cua_guard/adapters/` |
| Held-out evaluation metrics | `cua_guard/evaluation.py` |
| JSONL audit traces | `cua_guard/audit.py` |
| CLI surface for the whole pipeline | `cua_guard/cli.py` |

The boundary between the library and your stack is `ActionProposal`. Your
agent produces one (directly, or via an adapter); the gate scores it; you
either execute it or don't.

---

## 9. Glossary

- **CUA** — computer-use agent. A model that produces GUI actions.
- **Action proposal** — `(observation, action)` pair that hasn't run yet.
- **Scorer / danger classifier** — function from a proposal to a number in
  `[0, 1]` where larger means more dangerous.
- **Threshold `t`** — the cut-off. `score < t` → allow; otherwise block or
  escalate.
- **`alpha`** — your target risk. "Miss at most this fraction of unsafe
  actions in expectation."
- **Calibration set** — labeled examples held out from training, used only to
  pick `t`.
- **Conservative / inflated risk** — the `(L + 1) / (n + 1)` value that the
  calibrator actually compares to `alpha`.
- **Feasible** — there exists at least one threshold whose inflated risk fits
  inside `alpha`. If not, `alpha` is too tight for `n`.
- **Boundary threshold** — the chosen `t` is at the top or bottom of the
  search grid; the gate is effectively open or closed regardless of what the
  math says.
- **Escalation callback** — function you provide that decides whether to run
  an above-threshold action (a human, a stricter model, a Slack approval).
- **Audit trace** — JSONL of every guard decision, used for debugging and
  post-hoc review.

---

## 10. What to read next

- [`GETTING_STARTED.md`](GETTING_STARTED.md) — train, calibrate, run, and wire
  the gate into a real agent. This is where the concepts meet code.
- [`DESIGN.md`](DESIGN.md) — the calibrator's math, why the search runs in one
  direction, what each warning means under the hood.
- [`DATA_FORMAT.md`](DATA_FORMAT.md) — schema for labeled actions, labeled
  trajectories, adapter inputs, and audit records.
