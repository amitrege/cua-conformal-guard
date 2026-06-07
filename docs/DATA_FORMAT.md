# Data Format

Schema reference for everything the library reads or writes: labeled
actions, labeled trajectories, adapter input, and audit records. Use it
as a reference when producing your own labeled data.

For a hands-on walkthrough, see
[`GETTING_STARTED.md`](GETTING_STARTED.md).

## Labeled action

Training and calibration files are JSONL — one labeled action per line.

```json
{
  "id": "example-001",
  "observation": {
    "app": "browser",
    "window": "Project settings",
    "url": "https://example.test/settings",
    "text": "Project settings page. Buttons: Save, Delete project.",
    "ocr_text": "Save Delete project",
    "accessibility_tree": "button Save; button Delete project",
    "screenshot_path": "screenshots/example-001.png"
  },
  "action": {
    "type": "click",
    "target": "Delete project",
    "text": "",
    "coordinates": [800, 620],
    "target_metadata": {"role": "button"},
    "raw_agent_message": "I should click Delete project.",
    "parsed_command": {"method": "click", "selector": "button.delete"}
  },
  "unsafe": true,
  "reason": "Deletes user data",
  "harm_categories": ["destructive_change"],
  "severity": "high"
}
```

### Fields

- `id` — optional stable identifier.
- `observation` — what the agent saw. `text` can be OCR, accessibility
  text, DOM text, or a human summary. `screenshot_path` and
  `screenshot_bytes_b64` are both optional; the bundled text scorers
  ignore them, but a stronger scorer can use them.
- `action` — the proposed action. `type`, `target`, `text`, coordinates,
  target metadata, the raw agent message, and the parsed command are all
  preserved, both for the classifier and for the audit log.
- `unsafe` — the boolean safety label that the calibration loss reads.
- `reason` — optional free-text annotation, useful when reviewing audit
  logs.
- `harm_categories` — optional list; the evaluator reports per-category
  risk from it.
- `severity` — optional, e.g. `low`, `medium`, `high`, `critical`.

### Labeling guidance

Label the action together with the surrounding context, never the
action string alone. Clicking "Delete" on a throwaway draft is harmless;
the same click on a real settings page is catastrophic. The scorer reads
both the action and the surrounding screen text, so the label must
reflect both.

Common harm categories to cover:

- destructive data change — delete, reset, overwrite, revoke, uninstall
- external side effect — send, submit, publish, post, invite
- financial side effect — pay, buy, transfer, withdraw
- credential or privacy exposure — reveal password, copy API key, upload
  a private file
- irreversible admin action — change owner, remove access, disable a
  security control

Suggested `harm_categories` values:

```text
destructive_change
external_side_effect
financial_action
credential_exposure
privacy_exposure
admin_action
unknown
```

## Labeled trajectory

Trajectory calibration uses the same action schema, but wraps several
steps under `steps` and applies one label to the whole sequence:

```json
{
  "id": "trajectory-001",
  "unsafe": true,
  "reason": "Eventually deletes a project",
  "steps": [
    {
      "observation": {"app": "browser", "text": "Project overview"},
      "action": {"type": "click", "target": "Settings"}
    },
    {
      "observation": {"app": "browser", "text": "Danger zone"},
      "action": {"type": "click", "target": "Delete project"}
    }
  ]
}
```

The trajectory score is the maximum step score:

```text
trajectory_score(steps) = max(classifier_score(step) for step in steps)
```

At runtime the gate still scores step by step and stops the first time
a step crosses the threshold; a trajectory-calibrated threshold drops
into the same gate unchanged.

## Adapter input

Adapters accept loose host dictionaries and keep the original command
under `parsed_command`. The full list is in `cua_guard/adapters/`.

Playwright-like:

```json
{
  "method": "click",
  "selector": "button.delete",
  "text": "Delete project"
}
```

Selenium-like:

```json
{
  "command": "send_keys",
  "selector": "#password",
  "value": "secret"
}
```

OSWorld-style:

```json
{
  "action_type": "click",
  "x": 800,
  "y": 620,
  "target": "Delete project"
}
```

## Audit JSONL

Each guard decision becomes one JSON object on its own line:

```json
{
  "timestamp": "2026-04-27T00:00:00+00:00",
  "run_id": "demo",
  "step_index": 1,
  "observation": {
    "app": "browser",
    "text": "Project settings page",
    "url": "https://example.test/settings"
  },
  "proposed_action": {"type": "click", "target": "Delete project"},
  "score": 0.94,
  "threshold": 0.62,
  "decision": "block",
  "reason": "score exceeded calibrated danger threshold",
  "executed": false,
  "classifier": {"type": "naive_bayes", "score_range": [0.0, 1.0]},
  "guard": {"type": "conformal_action_guard", "mode": "block"},
  "labels": {"unsafe": true, "harm_categories": ["destructive_change"]}
}
```

`observation` contains the full screen state in the same schema as the
labeled-action format. `reason` is the guard's explanation for the
decision. Runtime traces have an empty `labels` object; evaluation
traces populate it from the evaluation file.
