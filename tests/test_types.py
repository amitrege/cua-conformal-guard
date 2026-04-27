import unittest

from cua_guard.types import ActionProposal, LabeledAction


class TypesTest(unittest.TestCase):
    def test_action_schema_round_trips_real_runtime_fields(self):
        data = {
            "observation": {
                "app": "browser",
                "window": "Settings",
                "url": "https://example.test/settings",
                "text": "Danger zone",
                "ocr_text": "Delete project",
                "accessibility_tree": "button Delete project",
                "screenshot_path": "screen.png",
            },
            "action": {
                "type": "click",
                "target": "Delete project",
                "coordinates": [10, 20],
                "target_metadata": {"role": "button"},
                "raw_agent_message": "Click delete",
                "parsed_command": {"method": "click", "selector": "button.delete"},
            },
        }

        proposal = ActionProposal.from_dict(data)
        round_trip = proposal.to_dict()

        self.assertEqual(round_trip["observation"]["window"], "Settings")
        self.assertEqual(round_trip["action"]["coordinates"], [10.0, 20.0])
        self.assertEqual(round_trip["action"]["target_metadata"]["role"], "button")
        self.assertIn("raw_agent_message", proposal.render_for_classifier())

    def test_string_false_label_is_parsed_as_false(self):
        record = LabeledAction.from_dict(
            {
                "observation": {"text": "Docs"},
                "action": {"type": "click", "target": "Search"},
                "unsafe": "false",
            }
        )

        self.assertFalse(record.unsafe)


if __name__ == "__main__":
    unittest.main()
