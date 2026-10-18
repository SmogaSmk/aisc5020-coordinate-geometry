"""Module-2 tests for replay isolation and diagnostic artifact semantics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import QuadraticProblem, RunConfig
from coordinate_geometry.adam_diagnostics import (
    canonical_gradient_history,
    diagnostic_records,
    main,
)


class AdamDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.settings = {
            "study_status": "module2_adam_diagnostic",
            "protocol_version": 1,
            "seed": 22,
            "eigenvalues": [0.1, 1.0],
            "initial_loss": 1.0,
            "angles_degrees": [0, 45],
            "max_steps": 4,
            "checkpoints": [1, 2, 4],
            "gd_learning_rate": 0.5,
            "adam_learning_rate": 0.1,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }

    def test_canonical_history_matches_exact_diagonal_gd_recurrence(self):
        eigenvalues = np.array([0.1, 1.0])
        problem = QuadraticProblem(eigenvalues, np.eye(2))
        initial = np.array([2.0, -3.0])
        config = RunConfig("gd", 0.5, 4)
        history = canonical_gradient_history(problem, initial, config)
        expected = np.asarray(
            [eigenvalues * ((1 - config.learning_rate * eigenvalues) ** step) * initial
             for step in range(config.max_steps)]
        )
        assert_allclose(history, expected, atol=1e-15)

    def test_identity_control_is_zero_and_general_rotation_has_a_defect(self):
        rows, problem, summary = diagnostic_records(self.settings)
        identity = [row for row in rows if row["angle_degrees"] == 0]
        rotated = [row for row in rows if row["angle_degrees"] == 45]
        assert_allclose([row["replay_relative_defect"] for row in identity], 0.0, atol=1e-14)
        assert_allclose([row["adam_normalized_position_error"] for row in identity], 0.0, atol=1e-14)
        self.assertGreater(max(row["replay_relative_defect"] for row in rotated), 0.01)
        self.assertGreater(max(row["adam_normalized_position_error"] for row in rotated), 0.01)
        self.assertEqual(len(rows), 8)
        self.assertEqual(problem["gradient_history_source"],
                         "canonical GD, gradient recorded before each update")
        self.assertFalse(summary["protocol_frozen"])

    def test_entrypoint_writes_reproducible_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            output = root / "output"
            config.write_text(json.dumps(self.settings), encoding="utf-8")
            self.assertEqual(main(["--config", str(config), "--output", str(output)]), 0)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"diagnostics.csv", "problem.json", "run.log", "run_config.json", "summary.json"},
            )
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["study_status"], "module2_adam_diagnostic")
            self.assertEqual(len(summary["angles"]), 2)

    def test_invalid_protocol_and_checkpoints_fail_loudly(self):
        invalid = dict(self.settings, study_status="main_study")
        with self.assertRaisesRegex(ValueError, "study_status"):
            diagnostic_records(invalid)
        invalid = dict(self.settings, checkpoints=[0, 4])
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            diagnostic_records(invalid)


if __name__ == "__main__":
    unittest.main()
