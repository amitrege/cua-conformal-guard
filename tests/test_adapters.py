import unittest

from cua_guard.adapters import (
    JsonActionAdapter,
    OSWorldActionAdapter,
    PlaywrightActionAdapter,
    SeleniumActionAdapter,
)


class AdapterTest(unittest.TestCase):
    def test_playwright_adapter_preserves_command_details(self):
        proposal = PlaywrightActionAdapter().to_proposal(
            {"app": "browser", "url": "https://example.test", "text": "Danger zone"},
            {"method": "click", "selector": "button.delete", "text": "Delete project"},
        )

        self.assertEqual(proposal.action_type, "click")
        self.assertEqual(proposal.target, "button.delete")
        self.assertEqual(proposal.metadata["adapter"], "playwright")
        self.assertIn("button.delete", proposal.render_for_classifier())

    def test_selenium_adapter_maps_selector_and_value(self):
        proposal = SeleniumActionAdapter().to_proposal(
            {"text": "Login form"},
            {"command": "send_keys", "selector": "#password", "value": "secret"},
        )

        self.assertEqual(proposal.action_type, "send_keys")
        self.assertEqual(proposal.target, "#password")
        self.assertEqual(proposal.text, "secret")

    def test_osworld_adapter_maps_coordinates(self):
        proposal = OSWorldActionAdapter().to_proposal(
            {"app": "desktop", "text": "Settings window"},
            {"action_type": "click", "x": 10, "y": 20, "target": "Reset"},
        )

        self.assertEqual(proposal.coordinates, (10.0, 20.0))
        self.assertEqual(proposal.target, "Reset")

    def test_json_adapter_accepts_native_schema(self):
        proposal = JsonActionAdapter().to_proposal(
            {"text": "ignored"},
            {
                "observation": {"text": "Native screen"},
                "action": {"type": "click", "target": "Save"},
            },
        )

        self.assertEqual(proposal.observation.text, "Native screen")
        self.assertEqual(proposal.action_type, "click")


if __name__ == "__main__":
    unittest.main()
