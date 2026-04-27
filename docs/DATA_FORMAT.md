# Data Format

Schema reference for everything the library reads or writes: labeled actions,
labeled trajectories, adapter inputs, and audit records. If you're trying to
make sense of how to *use* the library, start with
[`GETTING_STARTED.md`](GETTING_STARTED.md). This is the doc you keep open when
you start producing your own labeled data.

The training and calibration commands read JSONL files. Each line is one labeled
action.

## Labeled Action

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

Fields:

- `id`: Optional stable identifier.
- `observation`: What the agent saw. `text` can be OCR, accessibility text, DOM
  text, or a human summary. `screenshot_path` and `screenshot_bytes_b64` are
  optional; the built-in text scorers do not read images, but stronger scorers
  can.
- `action`: The proposed GUI action. `type`, `target`, `text`, coordinates,
  target metadata, raw agent message, and parsed command are rendered or
  preserved for classifiers and audit logs.
- `unsafe`: Boolean label for the calibrated loss.
- `reason`: Optional annotation for audit/debugging.
- `harm_categories`: Optional list used by the evaluator for per-category risk.
- `severity`: Optional label such as `low`, `medium`, `high`, or `critical`.

## Labeling guidance

Categories worth covering early:

- destructive data change: delete, reset, overwrite, revoke, uninstall
- external side effect: send, submit, publish, post, invite
- financial side effect: pay, buy, transfer, withdraw
- credential or privacy exposure: reveal password, copy API key, upload a private file
- irreversible admin action: change owner, remove access, disable a security control

Label the action in context, not the action string alone. `click "Delete"` on a
throwaway draft is fine. The same click on a real project settings page is not.
The classifier sees both the action and the surrounding screen text, so the
label needs to reflect both.

Suggested `harm_categories` values:

- `destructive_change`
- `external_side_effect`
- `financial_action`
- `credential_exposure`
- `privacy_exposure`
- `admin_action`
- `unknown`

## Labeled Trajectory

Trajectory-level calibration uses the same action schema, but wraps several
steps under `steps` and gives one label to the whole trajectory:

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
max(classifier_score(step) for step in trajectory)
```

The action-level guard at runtime uses the same threshold and stops as soon as
any single proposed step crosses it. A trajectory-calibrated threshold drops in
for action-level gating without any change.

## Adapter input

Adapters accept loose host dictionaries and preserve the original command under
`parsed_command`.

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

Audit logs are one JSON object per guard decision:

```json
{
  "timestamp": "2026-04-27T00:00:00+00:00",
  "run_id": "demo",
  "step_index": 1,
  "score": 0.94,
  "threshold": 0.62,
  "decision": "block",
  "executed": false,
  "proposed_action": {"type": "click", "target": "Delete project"},
  "classifier": {"type": "naive_bayes", "score_range": [0.0, 1.0]},
  "guard": {"type": "conformal_action_guard", "mode": "block"},
  "labels": {"unsafe": true, "harm_categories": ["destructive_change"]}
}
```

Runtime traces may have empty `labels`. Evaluation traces include labels from
the evaluation file.
