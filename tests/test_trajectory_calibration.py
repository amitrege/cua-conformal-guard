import unittest

from cua_guard.classifiers.naive_bayes import NaiveBayesDangerClassifier, proposal_from_text
from cua_guard.conformal.gcrc import GCRCThresholdCalibrator
from cua_guard.conformal.trajectory import calibrate_trajectories, trajectory_score
from cua_guard.types import LabeledAction, LabeledTrajectory


class TrajectoryCalibrationTest(unittest.TestCase):
    def test_trajectory_score_uses_riskiest_step(self):
        records = [
            LabeledAction(proposal_from_text("docs", "search docs"), unsafe=False),
            LabeledAction(proposal_from_text("settings", "delete project"), unsafe=True),
        ]
        classifier = NaiveBayesDangerClassifier().fit(records)
        low = proposal_from_text("docs", "search docs")
        high = proposal_from_text("settings", "delete project")

        self.assertEqual(
            trajectory_score(classifier, [low, high]),
            max(classifier.score(low), classifier.score(high)),
        )

    def test_calibrate_trajectories_returns_result(self):
        records = [
            LabeledAction(proposal_from_text("docs", "search docs"), unsafe=False),
            LabeledAction(proposal_from_text("notes", "save draft"), unsafe=False),
            LabeledAction(proposal_from_text("settings", "delete project"), unsafe=True),
            LabeledAction(proposal_from_text("bank", "confirm transfer"), unsafe=True),
        ]
        classifier = NaiveBayesDangerClassifier().fit(records)
        trajectories = [
            LabeledTrajectory((proposal_from_text("docs", "search docs"),), unsafe=False),
            LabeledTrajectory((proposal_from_text("settings", "delete project"),), unsafe=True),
            LabeledTrajectory((proposal_from_text("bank", "confirm transfer"),), unsafe=True),
        ]
        result = calibrate_trajectories(
            classifier,
            trajectories,
            GCRCThresholdCalibrator(alpha=0.5),
        )

        self.assertTrue(result.feasible)
        self.assertGreaterEqual(result.threshold, 0.0)
        self.assertLessEqual(result.threshold, 1.0)


if __name__ == "__main__":
    unittest.main()
