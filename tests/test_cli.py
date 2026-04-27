import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from cua_guard.cli import main
from cua_guard.io import read_json


ROOT = Path(__file__).resolve().parents[1]


class CLITest(unittest.TestCase):
    def test_train_calibrate_and_demo_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / "model.json"
            guard_path = tmp / "guard.json"
            trajectory_guard_path = tmp / "trajectory_guard.json"
            report_path = tmp / "report.json"
            audit_path = tmp / "audit.jsonl"

            with contextlib.redirect_stdout(io.StringIO()):
                train_code = main(
                    [
                        "train-classifier",
                        "--data",
                        str(ROOT / "examples" / "train_actions.jsonl"),
                        "--model",
                        str(model_path),
                    ]
                )
            self.assertEqual(train_code, 0)
            self.assertTrue(model_path.exists())

            with contextlib.redirect_stdout(io.StringIO()):
                calibrate_code = main(
                    [
                        "calibrate",
                        "--data",
                        str(ROOT / "examples" / "calibration_actions.jsonl"),
                        "--model",
                        str(model_path),
                        "--guard",
                        str(guard_path),
                        "--alpha",
                        "0.10",
                        "--mode",
                        "block",
                    ]
                )
            self.assertEqual(calibrate_code, 0)
            guard = read_json(guard_path)
            self.assertEqual(guard["type"], "conformal_action_guard")
            self.assertIn("threshold", guard)

            warning_stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()):
                with contextlib.redirect_stderr(warning_stderr):
                    trajectory_code = main(
                        [
                            "calibrate-trajectories",
                            "--data",
                            str(ROOT / "examples" / "calibration_trajectories.jsonl"),
                            "--model",
                            str(model_path),
                            "--guard",
                            str(trajectory_guard_path),
                            "--alpha",
                            "0.50",
                            "--mode",
                            "block",
                        ]
                    )
            self.assertEqual(trajectory_code, 0)
            self.assertIn("maximum grid value", warning_stderr.getvalue())

            demo_stdout = io.StringIO()
            with contextlib.redirect_stdout(demo_stdout):
                demo_code = main(
                    ["run-demo", "--guard", str(guard_path), "--audit-log", str(audit_path)]
                )
            self.assertEqual(demo_code, 0)
            self.assertIn('"unsafe_event": false', demo_stdout.getvalue())
            self.assertTrue(audit_path.exists())

            evaluate_stdout = io.StringIO()
            with contextlib.redirect_stdout(evaluate_stdout):
                evaluate_code = main(
                    [
                        "evaluate",
                        "--guard",
                        str(guard_path),
                        "--data",
                        str(ROOT / "examples" / "test_actions.jsonl"),
                        "--output",
                        str(report_path),
                    ]
                )
            self.assertEqual(evaluate_code, 0)
            report = read_json(report_path)
            self.assertIn("missed_unsafe_rate", report)
            self.assertIn("risk_by_harm_category", report)
            self.assertIn('"block_rate"', evaluate_stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
