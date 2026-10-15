"""Mathematical checks protecting the shared interface and coordinate conventions."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import QuadraticProblem, RunConfig, matched_pair, random_orthogonal, rotation_2d, run
from coordinate_geometry.adam import replay_defect, replay_directions
from coordinate_geometry.preconditioning import effective_hessian, p5_condition_number


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(731)
        self.d = np.geomspace(0.01, 1.0, 5)
        self.q = random_orthogonal(5, self.rng)
        self.base, self.rotated, self.z0, self.w0 = matched_pair(self.d, self.q, self.rng)

    def test_matched_initialization_preserves_loss_norm_and_gradient(self):
        assert_allclose(self.base.loss(self.z0), 1.0, atol=1e-13)
        assert_allclose(self.rotated.loss(self.w0), 1.0, atol=1e-13)
        assert_allclose(np.linalg.norm(self.z0), np.linalg.norm(self.w0), atol=1e-13)
        assert_allclose(self.rotated.gradient(self.w0), self.q @ self.base.gradient(self.z0), atol=1e-13)
        assert_allclose(np.linalg.eigvalsh(self.rotated.hessian), self.d, atol=1e-13)

    def test_gd_and_heavy_ball_correspondence(self):
        for algorithm in ("gd", "heavy_ball"):
            for steps in (1, 15, 100):
                with self.subTest(algorithm=algorithm, steps=steps):
                    config = RunConfig(algorithm, learning_rate=0.5, max_steps=steps)
                    base = run(self.base, self.z0, config)
                    rotated = run(self.rotated, self.w0, config)
                    assert_allclose(rotated.final_position, self.q @ base.final_position, atol=2e-12, rtol=2e-12)
                    assert_allclose([x.loss for x in rotated.trace], [x.loss for x in base.trace], atol=2e-12, rtol=2e-12)

    def test_gd_exact_eigen_coordinate_recurrence(self):
        steps, rate = 17, 1.2
        result = run(self.rotated, self.w0, RunConfig("gd", rate, steps))
        exact = self.q @ ((1 - rate * self.d)**steps * self.z0)
        assert_allclose(result.final_position, exact, atol=2e-12)

    def test_heavy_ball_exact_second_order_recurrence(self):
        rate, beta, steps = 0.7, 0.8, 25
        previous = self.z0.copy()  # Zero initial velocity means x_(-1) = x_0.
        current = self.z0.copy()
        for _ in range(steps):
            next_position = (1 + beta - rate * self.d) * current - beta * previous
            previous, current = current, next_position
        result = run(self.rotated, self.w0, RunConfig("heavy_ball", rate, steps, beta=beta))
        assert_allclose(result.final_position, self.q @ current, atol=2e-12)

    def test_adam_actual_epsilon_counterexample(self):
        q = rotation_2d(np.pi / 4)
        d = np.array([0.1, 1.0])
        z = np.array([10.0, 2.0])
        g = d * z
        config = RunConfig("adam", 0.1, 1)
        base_direction = replay_directions(g[None, :], config)[0]
        rotated_direction = replay_directions((q @ g)[None, :], config)[0]
        assert_allclose(base_direction, g / (np.abs(g) + config.epsilon), atol=1e-14)
        self.assertGreater(np.linalg.norm(rotated_direction - q @ base_direction), 0.5)
        base = run(QuadraticProblem(d, np.eye(2)), z, config)
        rotated = run(QuadraticProblem(d, q), q @ z, config)
        self.assertGreater(np.linalg.norm(rotated.final_position - q @ base.final_position), 0.05)

    def test_signed_permutation_adam_and_replay_correspondence(self):
        q = np.eye(5)[:, [2, 4, 1, 0, 3]] * np.array([-1, 1, -1, 1, -1])
        rotated_problem = QuadraticProblem(self.d, q)
        config = RunConfig("adam", 0.03, 80)
        base = run(self.base, self.z0, config)
        rotated = run(rotated_problem, q @ self.z0, config)
        assert_allclose(rotated.final_position, q @ base.final_position, atol=1e-12, rtol=1e-12)
        history = self.rng.normal(size=(100, 5))
        defect = replay_defect(history, q, config)
        assert_allclose(defect["relative_defect"], 0.0, atol=1e-12)

    def test_common_eigenbasis_adam_correspondence(self):
        base = run(self.base, self.z0, RunConfig("adam", 0.03, 80))
        rotated = run(self.rotated, self.w0, RunConfig("eigenbasis_adam", 0.03, 80))
        assert_allclose(rotated.final_position, self.q @ base.final_position, atol=2e-11, rtol=2e-11)
        assert_allclose([x.loss for x in rotated.trace], [x.loss for x in base.trace], atol=2e-12, rtol=2e-12)

    def test_regression_loss_and_gradient_equivalence(self):
        u, _ = np.linalg.qr(self.rng.normal(size=(13, 5)), mode="reduced")
        x = self.rotated.least_squares_design(u)
        w = self.rng.normal(size=5)
        assert_allclose(x.T @ x / 13, self.rotated.hessian, atol=1e-13)
        assert_allclose(np.linalg.norm(x @ w)**2 / 26, self.rotated.loss(w), atol=1e-13)
        assert_allclose(x.T @ (x @ w) / 13, self.rotated.gradient(w), atol=1e-13)

    def test_p5_formula_and_extreme_angles(self):
        for angle in np.linspace(0, np.pi / 2, 13):
            problem = QuadraticProblem(np.array([0.01, 1.0]), rotation_2d(angle))
            eigenvalues = np.linalg.eigvalsh(effective_hessian(problem.hessian))
            assert_allclose(p5_condition_number(problem.hessian), eigenvalues[-1] / eigenvalues[0], rtol=1e-12)
        aligned = QuadraticProblem(np.array([0.01, 1.0]), rotation_2d(0))
        diagonal = QuadraticProblem(np.array([0.01, 1.0]), rotation_2d(np.pi / 4))
        assert_allclose(p5_condition_number(aligned.hessian), 1.0)
        assert_allclose(p5_condition_number(diagonal.hessian), 100.0, rtol=1e-12)
        result = run(aligned, np.array([2.0, -1.0]), RunConfig("diagonal_gd", 1.0, 1))
        assert_allclose(result.final_position, np.zeros(2), atol=1e-14)

    def test_threshold_is_recorded_without_early_stop(self):
        problem = QuadraticProblem(np.ones(2), np.eye(2))
        initial = np.array([1.0, 2.0])
        result = run(problem, initial, RunConfig("gd", 1.0, 7))
        self.assertEqual(result.first_attainment, 1)
        self.assertEqual(result.gradient_calls, 7)
        self.assertEqual(result.status, "finite_cap")
        self.assertTrue(result.threshold_retained)
        assert_allclose(initial, [1.0, 2.0])

    def test_explosive_run_is_retained_not_imputed(self):
        problem = QuadraticProblem(np.ones(2), np.eye(2))
        result = run(problem, np.ones(2), RunConfig("gd", 10.0, 20, explosive_threshold=100))
        self.assertEqual(result.status, "explosive")
        self.assertIsNone(result.first_attainment)
        self.assertEqual(result.gradient_calls, 2)
        self.assertFalse(result.threshold_retained)
        self.assertTrue(np.all(np.isfinite(result.final_position)))
        self.assertGreater(result.trace[-1].relative_loss, 100)

    def test_nonfinite_update_keeps_last_finite_position(self):
        problem = QuadraticProblem(np.ones(1), np.eye(1))
        initial = np.array([2.0])
        result = run(problem, initial, RunConfig("gd", 1e308, 2))
        self.assertEqual(result.status, "nonfinite")
        assert_allclose(result.final_position, initial)
        self.assertIsNone(result.trace[-1].relative_loss)

    def test_nonfinite_adam_moments_are_a_failure_even_with_finite_position(self):
        # g is finite, but g**2 overflows; silently dividing by infinity would mask it.
        problem = QuadraticProblem(np.array([1e308]), np.eye(1))
        initial = np.array([1e-150])
        result = run(problem, initial, RunConfig("adam", 0.01, 3))
        self.assertEqual(result.status, "nonfinite")
        self.assertEqual(result.gradient_calls, 1)
        assert_allclose(result.final_position, initial)

    def test_invalid_inputs_fail_loudly(self):
        with self.assertRaises(ValueError):
            QuadraticProblem(np.array([-1.0, 1.0]), np.eye(2))
        with self.assertRaises(ValueError):
            QuadraticProblem(np.ones(2), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            RunConfig("adam", 0.01, 5, beta2=1.0)
        with self.assertRaises(ValueError):
            run(self.base, np.zeros(5), RunConfig("gd", 0.1, 5))


if __name__ == "__main__":
    unittest.main()
