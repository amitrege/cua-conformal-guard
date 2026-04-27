import tempfile
import unittest
from pathlib import Path

from cua_guard.classifiers.naive_bayes import NaiveBayesDangerClassifier, proposal_from_text
from cua_guard.types import LabeledAction


class NaiveBayesDangerClassifierTest(unittest.TestCase):
    def test_trained_model_scores_obvious_danger_higher_than_safe_action(self):
        records = [
            LabeledAction(proposal_from_text("docs page", "click search docs"), unsafe=False),
            LabeledAction(proposal_from_text("notes page", "save draft"), unsafe=False),
            LabeledAction(proposal_from_text("danger zone", "delete project"), unsafe=True),
            LabeledAction(proposal_from_text("bank transfer", "confirm transfer"), unsafe=True),
        ]
        model = NaiveBayesDangerClassifier().fit(records)

        safe = model.score(proposal_from_text("docs page", "click search"))
        unsafe = model.score(proposal_from_text("project danger zone", "delete project"))

        self.assertGreater(unsafe, safe)
        self.assertGreaterEqual(safe, 0.0)
        self.assertLessEqual(unsafe, 1.0)

    def test_save_and_load_preserves_scores(self):
        records = [
            LabeledAction(proposal_from_text("help page", "search"), unsafe=False),
            LabeledAction(proposal_from_text("settings", "delete account"), unsafe=True),
        ]
        model = NaiveBayesDangerClassifier().fit(records)
        proposal = proposal_from_text("settings", "delete account")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            model.save(path)
            loaded = NaiveBayesDangerClassifier.load(path)

        self.assertAlmostEqual(model.score(proposal), loaded.score(proposal), places=12)


if __name__ == "__main__":
    unittest.main()
