# Data Format

The training and calibration commands read JSONL files. Each line is one labeled
action.

## Labeled Action

```json
{
  "id": "example-001",
  "observation": {
    "app": "browser",
    "url": "https://example.test/settings",
    "text": "Project settings page. Buttons: Save, Delete project."
  },
  "action": {
    "type": "click",
    "target": "Delete project",
    "text": ""
  },
  "unsafe": true,
  "reason": "Deletes user data"
}
```

Fields:

- `id`: Optional stable identifier.
- `observation`: What the agent saw. `text` can be OCR, accessibility text, DOM
  text, or a human summary.
- `action`: The proposed GUI action. `type`, `target`, and `text` are rendered
  into classifier input.
- `unsafe`: Boolean label for the calibrated loss.
- `reason`: Optional annotation for audit/debugging.

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
