import tempfile
import unittest
from pathlib import Path

from cua_guard.classifiers.loader import load_classifier
from cua_guard.classifiers.rules import KeywordDangerScorer
from cua_guard.classifiers.naive_bayes import proposal_from_text


class KeywordDangerScorerTest(unittest.TestCase):
    def test_scores_destructive_action_above_benign(self):
        scorer = KeywordDangerScorer()

        benign = scorer.score(proposal_from_text("docs page", "search docs"))
        destructive = scorer.score(proposal_from_text("settings", "delete project"))

        self.assertGreater(destructive, benign)
        self.assertGreaterEqual(benign, 0.0)
        self.assertLessEqual(destructive, 1.0)

    def test_score_is_bounded_in_unit_interval(self):
        scorer = KeywordDangerScorer()
        # Hit every default pattern at once to exercise the saturation behavior.
        proposal = proposal_from_text(
            "production admin owner security confirm final irreversible permanent",
            "delete remove reset format send pay transfer password api key",
        )
        score = scorer.score(proposal)
        self.assertGreater(score, 0.9)
        self.assertLess(score, 1.0)

    def test_save_load_roundtrips_via_loader(self):
        scorer = KeywordDangerScorer()
        proposal = proposal_from_text("settings", "delete project")
        original = scorer.score(proposal)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rules.json"
            scorer.save(path)
            loaded = load_classifier(path)

        self.assertIsInstance(loaded, KeywordDangerScorer)
        self.assertAlmostEqual(loaded.score(proposal), original, places=12)


if __name__ == "__main__":
    unittest.main()
