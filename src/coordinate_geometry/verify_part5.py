"""Independently recompute saved Module 5 claims from raw position arrays.

No optimizer or basis-preparation helper is imported: updates are reconstructed
directly from the Adam equations. This checks the saved results, not just exit
codes, and detects stale source/configuration manifests.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose


def expected_problems(config):
    """Rebuild the seed schedule independently of the experiment generator."""
    stream = np.random.SeedSequence(config["seed"])
    def make(case_id, d, q, rng, scope, angle):
        u = rng.normal(size=len(d))
        z = np.sqrt(2.0 / float(u @ (d * u))) * u
        return case_id, d, q, z, scope, angle
    for kappa in config["condition_numbers"]:
        d = np.array([1.0 / kappa, 1.0])
        for init_id in range(config["two_dimensional_initializations"]):
            child = stream.spawn(1)[0]
            for angle in config["angles_degrees"]:
                theta = np.deg2rad(angle)
                q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                yield make(f"p2_k{kappa}_i{init_id}_a{angle}", d, q, np.random.default_rng(child), "well_separated", angle)
    for p in config["higher_dimensions"]:
        for kappa in config["condition_numbers"]:
            d = np.geomspace(1.0 / kappa, 1.0, p)
            for rotation_id in range(config["higher_dimensional_rotations"][str(p)]):
                rng = np.random.default_rng(stream.spawn(1)[0])
                q, r = np.linalg.qr(rng.normal(size=(p, p)))
                q *= np.where(np.diag(r) < 0, -1.0, 1.0)
                for init_id in range(config["higher_dimensional_initializations"]):
                    yield make(f"p{p}_k{kappa}_r{rotation_id}_i{init_id}", d, q,
                               np.random.default_rng(stream.spawn(1)[0]), "well_separated", None)
    theta = np.pi / 4
    q = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    yield make("repeated_identity", np.ones(2), q, np.random.default_rng(901), "repeated", 45)
    rng = np.random.default_rng(55019)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)
    yield make("near_repeated", np.array([1e-4, 1e-4 + 1e-11, 1.0]), q,
               np.random.default_rng(902), "near_repeated", None)


def check_updates(positions, losses, hessian, basis, config):
    assert np.all(np.isfinite(positions)) and np.all(np.isfinite(losses))
    assert positions.shape[0] == config["max_steps"] + 1
    recomputed_losses = .5 * np.einsum("ti,ij,tj->t", positions, hessian, positions)
    assert_allclose(losses, recomputed_losses, rtol=3e-10, atol=2e-13)
    m, s = np.zeros(positions.shape[1]), np.zeros(positions.shape[1])
    for t in range(1, positions.shape[0]):
        g = hessian @ positions[t - 1]
        if basis is not None:
            g = basis.T @ g
        m = config["beta1"] * m + (1 - config["beta1"]) * g
        s = config["beta2"] * s + (1 - config["beta2"]) * g**2
        direction = (m / (1 - config["beta1"]**t)) / (np.sqrt(s / (1 - config["beta2"]**t)) + config["epsilon"])
        if basis is not None:
            direction = basis @ direction
        expected = positions[t - 1] - config["learning_rate"] * direction
        assert_allclose(positions[t], expected, rtol=2e-11, atol=2e-13)


def verify(output, *, check_sources=True):
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    with (output / "correspondence.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == summary["trajectory_count"]
    case_ids = sorted({row["case_id"] for row in rows})
    expected = {item[0]: item[1:] for item in expected_problems(config)}
    assert set(case_ids) == set(expected), "Saved case schedule does not match the declared configuration"
    assert len(case_ids) == summary["case_count"]
    assert set(case_ids) == {path.stem for path in (output / "raw").glob("*.npz")}, "Stale or missing raw cases"
    required = []
    maxima = {"oracle_basis": 0.0, "recovered_basis": 0.0}
    for case_id in case_ids:
        with np.load(output / "raw" / f"{case_id}.npz", allow_pickle=False) as raw:
            d, q = raw["eigenvalues"], raw["rotation"]
            expected_d, expected_q, expected_z0, expected_scope, expected_angle = expected[case_id]
            assert_allclose(d, expected_d, rtol=1e-14, atol=1e-15)
            assert_allclose(q, expected_q, rtol=2e-12, atol=2e-12)
            assert_allclose(raw["canonical_initial"], expected_z0, rtol=2e-12, atol=2e-12)
            h = (q * d) @ q.T
            assert_allclose(q.T @ q, np.eye(len(d)), atol=2e-12, rtol=2e-12)
            canonical = raw["canonical_positions"]
            assert_allclose(canonical[0], raw["canonical_initial"], atol=1e-14)
            check_updates(canonical, raw["canonical_losses"], np.diag(d), None, config)
            expected_path = canonical @ q.T
            denominator = max(1.0, float(np.linalg.norm(expected_path[0])))
            case_rows = [r for r in rows if r["case_id"] == case_id]
            assert {r["method"] for r in case_rows} == {"standard_adam", "oracle_basis", "recovered_basis"}
            for row in case_rows:
                method = row["method"]
                assert row["scope"] == expected_scope
                assert row["angle_degrees"] == ("" if expected_angle is None else str(expected_angle))
                assert int(row["dimension"]) == len(d)
                assert_allclose(float(row["condition_number"]), np.max(d) / np.min(d), rtol=1e-13)
                assert float(row["learning_rate"]) == config["learning_rate"]
                basis = None if method == "standard_adam" else raw[method + "_basis"]
                if basis is not None:
                    assert_allclose(basis.T @ basis, np.eye(len(d)), rtol=2e-12, atol=2e-12)
                    transformed = basis.T @ h @ basis
                    assert_allclose(transformed, np.diag(np.diag(transformed)), rtol=1e-10, atol=1e-10)
                positions, losses = raw[method + "_positions"], raw[method + "_losses"]
                assert_allclose(positions[0], expected_path[0], rtol=2e-12, atol=2e-12)
                check_updates(positions, losses, h, basis, config)
                errors = np.linalg.norm(positions - expected_path, axis=1) / denominator
                assert_allclose(errors, raw[method + "_errors"], rtol=1e-9, atol=2e-15)
                maximum = float(np.max(errors))
                assert_allclose(maximum, float(row["maximum_normalized_position_error"]), rtol=1e-9, atol=2e-15)
                assert_allclose(float(np.max(np.abs(losses - raw["canonical_losses"]))),
                                float(row["maximum_absolute_objective_difference"]), rtol=1e-9, atol=2e-15)
                assert_allclose(losses[-1] / losses[0], float(row["final_relative_loss"]), rtol=1e-9, atol=2e-15)
                assert int(row["gradient_calls"]) == config["max_steps"]
                expected_required = method == "oracle_basis" or (method == "recovered_basis" and row["scope"] == "well_separated")
                assert (row["correspondence_required"] == "True") == expected_required
                if expected_required:
                    assert maximum <= config["position_error_tolerance"]
                    assert row["correspondence_passed"] == "True"
                    required.append(row)
                    maxima[method] = max(maxima[method], maximum)
    assert len(required) == summary["required_correspondence_checks"]
    assert summary["all_required_checks_passed"] is True
    for method, maximum in maxima.items():
        assert_allclose(maximum, summary["maximum_required_path_error_by_method"][method], rtol=1e-9, atol=2e-15)
    scope_rows = [row for row in rows if row["scope"] != "well_separated"]
    assert len(summary["scope_controls"]) == len(scope_rows)
    for original, saved in zip(scope_rows, summary["scope_controls"]):
        for key, value in saved.items():
            expected_text = "" if value is None else str(value)
            assert original[key] == expected_text, "Scope-control summary mismatch: " + key
    with (output / "timing.csv").open(encoding="utf-8", newline="") as handle:
        timing = list(csv.DictReader(handle))
    assert len(timing) == summary["timing_repetitions"]
    for row in timing:
        calls = int(row["gradient_calls"])
        assert calls == config["benchmark_steps"] and row["status"] == "finite_cap"
        assert int(row["gradient_matrix_vector_products"]) == calls
        assert int(row["loss_matrix_vector_products"]) == calls + 1
        assert int(row["basis_transform_calls"]) == (0 if row["method"] == "standard_adam" else 2 * calls)
        assert int(row["setup_eigendecomposition_calls"]) == int(row["method"] == "recovered_basis")
        for key in ("setup_seconds", "evaluation_seconds", "total_seconds"):
            assert np.isfinite(float(row[key])) and float(row[key]) >= 0
        assert float(row["total_seconds"]) >= float(row["evaluation_seconds"])
        assert row["warmup_performed_by_harness"] == "True"
        assert int(row["setup_validation_matrix_matrix_products"]) == (0 if row["method"] == "standard_adam" else 2)
    expected_timing_keys = {(p, method, rep)
                            for p in config["benchmark_dimensions"]
                            for method in ("standard_adam", "oracle_basis", "recovered_basis")
                            for rep in range(config["benchmark_repetitions"])}
    actual_timing_keys = {(int(r["dimension"]), r["method"], int(r["repetition"])) for r in timing}
    assert actual_timing_keys == expected_timing_keys and len(timing) == len(expected_timing_keys)
    expected_median_keys = {(p, method) for p in config["benchmark_dimensions"]
                            for method in ("standard_adam", "oracle_basis", "recovered_basis")}
    assert {(r["dimension"], r["method"]) for r in summary["timing_medians"]} == expected_median_keys
    assert len(summary["timing_medians"]) == len(expected_median_keys)
    for row in summary["timing_medians"]:
        selected = [r for r in timing if int(r["dimension"]) == row["dimension"] and r["method"] == row["method"]]
        for key in ("setup_seconds", "evaluation_seconds", "total_seconds"):
            assert_allclose(row[key], float(np.median([float(r[key]) for r in selected])), rtol=1e-12, atol=1e-15)
    diagnoses = json.loads((output / "basis_diagnostics.json").read_text(encoding="utf-8"))
    assert len(diagnoses) == 2 * len(case_ids)
    assert {(r["case_id"], r["source"]) for r in diagnoses} == {(c, s) for c in case_ids for s in ("oracle", "eigh")}
    for diagnostic in diagnoses:
        method = "oracle_basis" if diagnostic["source"] == "oracle" else "recovered_basis"
        with np.load(output / "raw" / (diagnostic["case_id"] + ".npz"), allow_pickle=False) as raw:
            basis, d, q = raw[method + "_basis"], raw["eigenvalues"], raw["rotation"]
            h = (q * d) @ q.T
            orthogonal_error = np.linalg.norm(basis.T @ basis - np.eye(len(d)), ord="fro") / np.sqrt(len(d))
            scaled = h / np.max(np.abs(h))
            product = scaled @ basis
            values = np.sum(basis * product, axis=0)
            residual = np.linalg.norm(product - basis * values, ord="fro") / np.linalg.norm(scaled, ord="fro")
            assert_allclose(diagnostic["orthogonality_error"], orthogonal_error, rtol=1e-8, atol=1e-15)
            assert_allclose(diagnostic["relative_diagonalization_residual"], residual, rtol=1e-8, atol=1e-15)
            assert diagnostic["oracle_alignment_performed"] is False
            assert diagnostic["eigendecomposition_calls"] == int(method == "recovered_basis")
            row = next(r for r in rows if r["case_id"] == diagnostic["case_id"] and r["method"] == method)
            assert row["eigenvectors_numerically_uncertain"] == str(diagnostic["eigenvectors_numerically_uncertain"])
    if check_sources:
        root = Path(__file__).resolve().parents[2]
        for relative, expected in summary["source"]["sha256"].items():
            assert hashlib.sha256((root / relative).read_bytes().replace(b"\r\n", b"\n")).hexdigest() == expected, "Changed source: " + relative
    evidence = {"passed": True, "case_count": len(case_ids), "trajectories_recomputed": len(rows) + len(case_ids),
                "required_correspondence_checks": len(required), "timing_rows_checked": len(timing),
                "source_hashes_checked": len(summary["source"]["sha256"]) if check_sources else 0,
                "checks": "Seeded problem schedule and initializations, raw Adam updates at every iterate, losses, CSV/JSON metrics, timing medians, basis residuals, operation counts and normalized-LF source hashes."}
    (output / "artifact_verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/part5"))
    args = parser.parse_args(argv)
    print(json.dumps(verify(args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
