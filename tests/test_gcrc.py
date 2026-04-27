import unittest

from cua_guard.conformal.gcrc import (
    GCRCThresholdCalibrator,
    conservative_risk_at_threshold,
)


class GCRCThresholdCalibratorTest(unittest.TestCase):
    def test_calibrator_picks_largest_threshold_before_missed_unsafe_risk_exceeds_alpha(self):
        scores = [0.05, 0.10, 0.20, 0.80, 0.90]
        unsafe = [False, False, False, True, True]
        calibrator = GCRCThresholdCalibrator(alpha=0.25, grid_size=11)

        result = calibrator.calibrate(scores, unsafe, grid=[0.0, 0.5, 0.8, 0.800000001, 1.0])

        self.assertTrue(result.feasible)
        self.assertEqual(result.threshold, 0.8)
        self.assertLessEqual(result.conservative_risk, 0.25)
        self.assertGreater(
            conservative_risk_at_threshold(scores, unsafe, 0.800000001),
            0.25,
        )

    def test_reports_infeasible_when_alpha_is_smaller_than_finite_sample_floor(self):
        scores = [0.9, 0.95]
        unsafe = [True, True]
        calibrator = GCRCThresholdCalibrator(alpha=0.1)

        result = calibrator.calibrate(scores, unsafe, grid=[0.0, 1.0])

        self.assertFalse(result.feasible)
        self.assertEqual(result.threshold, 0.0)
        self.assertEqual(result.boundary, "min")
        self.assertIn("minimum grid value", result.boundary_warning())

    def test_reports_boundary_warning_when_threshold_is_max_grid_value(self):
        scores = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        unsafe = [False, False, False, False, False, False]
        calibrator = GCRCThresholdCalibrator(alpha=0.5)

        result = calibrator.calibrate(scores, unsafe, grid=[0.0, 0.5, 1.0])

        self.assertTrue(result.feasible)
        self.assertEqual(result.threshold, 1.0)
        self.assertEqual(result.boundary, "max")
        self.assertIn("maximum grid value", result.boundary_warning())


if __name__ == "__main__":
    unittest.main()
