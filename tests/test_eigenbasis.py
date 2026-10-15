"""Role-5 tests for basis information, trajectory semantics and setup accounting."""

from types import SimpleNamespace
from dataclasses import replace
import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry import (
    OptimizerState, QuadraticProblem, RunConfig, eigenbasis_adam_step,
    matched_pair, prepare_eigenbasis, random_orthogonal, rotation_2d, run,
)
from coordinate_geometry.adam import adam_step


class EigenbasisTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5501)
        self.d = np.geomspace(0.001, 1.0, 6)
        self.q = random_orthogonal(6, self.rng)
        self.base, self.rotated, self.z0, self.w0 = matched_pair(self.d, self.q, self.rng)
        self.config = RunConfig("eigenbasis_adam", 0.02, 80)

    def test_eigh_reads_only_hessian_and_preserves_solver_signs(self):
        # This object has no oracle rotation or spectrum available to read.
        hessian_only = SimpleNamespace(hessian=self.rotated.hessian)
        raw_values, raw_vectors = np.linalg.eigh(self.rotated.hessian)
        changed_vectors = raw_vectors * np.array([-1, 1, -1, 1, -1, 1])
        with patch("coordinate_geometry.eigenbasis.np.linalg.eigh", return_value=(raw_values, changed_vectors)) as solver:
            prepared = prepare_eigenbasis(hessian_only, source="eigh")
        solver.assert_called_once()
        assert_allclose(prepared.vectors, changed_vectors, atol=0)
        assert_allclose(prepared.eigenvalues, self.d, atol=1e-13)
        self.assertEqual(prepared.source, "eigh")
        self.assertFalse(prepared.diagnostics["oracle_alignment_performed"])

    def test_shared_basis_correspondence_at_every_finite_iterate(self):
        for d in (self.d, np.array([0.1, 0.1, 0.3, 0.3, 1.0, 1.0])):
            with self.subTest(spectrum=d):
                base, rotated, z0, w0 = matched_pair(d, self.q, self.rng)
                prepared = prepare_eigenbasis(rotated)
                zstate, wstate = OptimizerState.initial(z0), OptimizerState.initial(w0)
                for _ in range(100):
                    adam_step(zstate, base.gradient(zstate.position), self.config)
                    eigenbasis_adam_step(prepared, wstate, rotated.gradient(wstate.position), self.config)
                    assert_allclose(wstate.position, self.q @ zstate.position, atol=3e-11, rtol=3e-11)
                    assert_allclose(rotated.loss(wstate.position), base.loss(zstate.position), atol=2e-12)

    def test_distinct_recovered_basis_restores_correspondence(self):
        config = RunConfig("eigenbasis_adam", 0.02, 80, basis_source="eigh")
        recovered = run(self.rotated, self.w0, config)
        baseline = run(self.base, self.z0, RunConfig("adam", 0.02, 80))
        assert_allclose(recovered.final_position, self.q @ baseline.final_position, atol=3e-10, rtol=3e-10)
        assert_allclose([t.loss for t in recovered.trace], [t.loss for t in baseline.trace], atol=1e-11)
        self.assertEqual(recovered.metadata["basis_source"], "eigh")

    def test_signed_permuted_supplied_basis_is_equivalent(self):
        permutation = [3, 0, 5, 1, 4, 2]
        supplied = self.q[:, permutation] * np.array([-1, 1, 1, -1, 1, -1])
        reference = run(self.rotated, self.w0, self.config)
        result = run(self.rotated, self.w0, self.config, basis=supplied)
        assert_allclose(result.final_position, reference.final_position, atol=2e-11, rtol=2e-11)
        self.assertEqual(result.metadata["basis_source"], "supplied")

    def test_repeated_eigenspaces_require_a_common_basis(self):
        problem = QuadraticProblem(np.array([0.1, 0.1, 1.0]), np.eye(3))
        alternative = np.eye(3)
        alternative[:2, :2] = rotation_2d(0.4)
        prepared = prepare_eigenbasis(problem, basis=alternative)
        self.assertTrue(prepared.diagnostics["has_clustered_eigenvalues"])
        self.assertEqual([cluster["size"] for cluster in prepared.diagnostics["eigenvalue_clusters"]], [2, 1])
        self.assertEqual(prepared.diagnostics["correspondence_scope"], "shared_basis_required_for_clustered_eigenspaces")
        initial = np.array([1.0, 2.0, 3.0])
        config = RunConfig("eigenbasis_adam", 0.1, 1)
        reference = run(problem, initial, config)
        changed = run(problem, initial, config, basis=prepared)
        # Both are valid eigenbases of the same H, yet Adam's updates can differ.
        self.assertGreater(np.linalg.norm(reference.final_position - changed.final_position), 0.01)

    def test_setup_runs_once_and_counts_two_transforms_per_step(self):
        original_eigh = np.linalg.eigh
        with patch("coordinate_geometry.eigenbasis.np.linalg.eigh", wraps=original_eigh) as solver:
            result = run(self.rotated, self.w0, RunConfig("eigenbasis_adam", 0.02, 9, basis_source="eigh"))
        solver.assert_called_once()
        metadata = result.metadata
        self.assertEqual(metadata["gradient_matrix_vector_products"], 9)
        self.assertEqual(metadata["basis_transform_calls"], 18)
        self.assertEqual(metadata["loss_matrix_vector_products"], 10)
        self.assertEqual(metadata["setup_eigendecomposition_calls"], 1)
        self.assertEqual(metadata["setup_validation_matrix_matrix_products"], 2)
        for key in ("setup_seconds", "evaluation_seconds", "total_seconds"):
            self.assertTrue(np.isfinite(metadata[key]))
            self.assertGreaterEqual(metadata[key], 0)
        self.assertGreaterEqual(metadata["total_seconds"], metadata["setup_seconds"] + metadata["evaluation_seconds"])
        self.assertFalse(metadata["warmup_performed"])

    def test_prepared_basis_reuse_does_not_repeat_eigendecomposition(self):
        prepared = prepare_eigenbasis(self.rotated, source="eigh")
        with patch("coordinate_geometry.eigenbasis.np.linalg.eigh", side_effect=AssertionError("unexpected re-factorization")):
            first = run(self.rotated, self.w0, self.config, basis=prepared)
            second = run(self.rotated, self.w0, self.config, basis=prepared)
        assert_allclose(first.final_position, second.final_position)
        self.assertTrue(second.metadata["basis_setup_reused"])
        self.assertEqual(second.metadata["setup_eigendecomposition_calls"], 0)
        self.assertEqual(second.metadata["setup_validation_matrix_matrix_products"], 0)
        self.assertEqual(second.metadata["basis_source"], "eigh")
        self.assertEqual(second.metadata["basis_original_preparation_seconds"], prepared.setup_seconds)

    def test_mismatched_prepared_hessian_is_rejected(self):
        prepared = prepare_eigenbasis(self.rotated)
        other = QuadraticProblem(self.d * 2, self.q)
        with self.assertRaisesRegex(ValueError, "different Hessian"):
            run(other, self.w0, self.config, basis=prepared)

    def test_invalid_basis_shapes_values_and_diagonalization_are_rejected(self):
        for candidate in (np.ones((6, 5)), np.full((6, 6), np.nan), np.ones((6, 6)), np.eye(6)):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    prepare_eigenbasis(self.rotated, basis=candidate)
        for h in (np.ones((2, 3)), np.array([[1.0, np.nan], [np.nan, 1.0]]),
                  np.array([[1.0, 0.1], [0, 1.0]]), np.diag([1.0, -1.0])):
            with self.subTest(hessian=h):
                with self.assertRaises(ValueError):
                    prepare_eigenbasis(SimpleNamespace(hessian=h), source="eigh")

    def test_other_optimizers_do_not_receive_or_prepare_a_basis(self):
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            run(self.base, self.z0, RunConfig("gd", 0.1, 5), basis=np.eye(6))
        with self.assertRaisesRegex(ValueError, "only meaningful"):
            RunConfig("adam", 0.1, 5, basis_source="eigh")
        with patch("coordinate_geometry.runner.prepare_eigenbasis", side_effect=AssertionError("unexpected basis setup")):
            result = run(self.base, self.z0, RunConfig("adam", 0.1, 5))
        self.assertEqual(result.metadata["setup_seconds"], 0.0)
        self.assertEqual(result.metadata["basis_transform_calls"], 0)
        self.assertEqual(result.metadata["basis_source"], "ambient")

    def test_preparation_copies_and_protects_input_arrays(self):
        supplied = self.q.copy()
        prepared = prepare_eigenbasis(self.rotated, basis=supplied)
        supplied[:] = 0
        assert_allclose(prepared.vectors, self.q)
        with self.assertRaises(ValueError):
            prepared.vectors[0, 0] = 0
        for array in (prepared.vectors, prepared.eigenvalues, prepared.reference_hessian):
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertIs(copy.copy(prepared), prepared)
        self.assertIs(copy.deepcopy(prepared), prepared)
        diagnostics = prepared.diagnostics
        diagnostics["eigenvalue_clusters"][0]["size"] = 999
        self.assertNotEqual(prepared.diagnostics["eigenvalue_clusters"][0]["size"], 999)

    def test_unchecked_construction_and_dataclass_replacement_are_rejected(self):
        prepared = prepare_eigenbasis(self.rotated)
        with self.assertRaisesRegex(TypeError, "factory-only"):
            type(prepared)()
        with self.assertRaisesRegex(TypeError, "factory-only"):
            type(prepared)(vectors=np.zeros((6, 6)), eigenvalues=self.d, source="supplied",
                           setup_seconds=0.0, diagnostics={}, reference_hessian=self.rotated.hessian)
        for replacement in (
            {"vectors": np.zeros((6, 6))},
            {"vectors": np.ones((6, 5))},
            {"eigenvalues": np.zeros(6)},
            {"eigenvalues": np.ones(7)},
        ):
            with self.subTest(replacement=replacement):
                with self.assertRaisesRegex(TypeError, "factory-only"):
                    replace(prepared, **replacement)

    def test_near_repeated_roots_are_numerically_uncertain_at_hessian_scale(self):
        rng = np.random.default_rng(55019)
        q = random_orthogonal(3, rng)
        problem = QuadraticProblem(np.array([1e-4, 1e-4 + 1e-11, 1.0]), q)
        prepared = prepare_eigenbasis(problem, source="eigh")
        diagnostics = prepared.diagnostics
        self.assertTrue(diagnostics["stored_eigenvalues_all_distinct"])
        self.assertTrue(diagnostics["has_clustered_eigenvalues"])
        self.assertTrue(diagnostics["eigenvectors_numerically_uncertain"])
        self.assertFalse(diagnostics["eigenbasis_numerically_identifiable"])
        self.assertGreater(diagnostics["unresolved_eigengap_threshold_absolute"],
                           diagnostics["minimum_absolute_eigengap"])
        self.assertEqual(diagnostics["correspondence_scope"], "shared_basis_required_for_clustered_eigenspaces")
        self.assertGreater(diagnostics["eigenvector_uncertainty_indicator"],
                           diagnostics["eigenvector_uncertainty_target"])

    def test_well_separated_spectrum_has_distinct_numerical_scope(self):
        prepared = prepare_eigenbasis(self.rotated, source="eigh")
        self.assertTrue(prepared.diagnostics["stored_eigenvalues_all_distinct"])
        self.assertFalse(prepared.diagnostics["eigenvectors_numerically_uncertain"])
        self.assertTrue(prepared.diagnostics["eigenbasis_numerically_identifiable"])
        self.assertEqual(prepared.diagnostics["correspondence_scope"], "distinct_eigenbasis_up_to_signed_permutation")


if __name__ == "__main__":
    unittest.main()
