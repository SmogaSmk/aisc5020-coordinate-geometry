"""Reproduce Module 5 correspondence checks and setup-inclusive timing.

The fixed rates below test an identity, not optimizer superiority. Raw problems
and position traces are retained so a separate checker can recompute claims.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from .adam import adam_step
from .contracts import OptimizerState, RunConfig
from .eigenbasis import eigenbasis_adam_step, prepare_eigenbasis
from .problems import QuadraticProblem, matched_pair, random_orthogonal, rotation_2d
from .runner import run


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_trajectory(problem, initial, config, prepared=None):
    """Instrument every iterate for this verification suite, not the main runner."""
    state = OptimizerState.initial(initial)
    positions = [state.position.copy()]
    losses = [problem.loss(initial)]
    for _ in range(config.max_steps):
        gradient = problem.gradient(state.position)
        if prepared is None:
            adam_step(state, gradient, config)
        else:
            eigenbasis_adam_step(prepared, state, gradient, config)
        if not all(np.all(np.isfinite(a)) for a in (state.position, state.first_moment, state.second_moment)):
            raise RuntimeError("Nonfinite verification trajectory: inspect raw problem and configuration")
        positions.append(state.position.copy())
        losses.append(problem.loss(state.position))
    return np.asarray(positions), np.asarray(losses)


def case_records(settings):
    stream = np.random.SeedSequence(settings["seed"])
    for kappa in settings["condition_numbers"]:
        d = np.array([1.0 / kappa, 1.0])
        for init_id in range(settings["two_dimensional_initializations"]):
            child = stream.spawn(1)[0]
            for angle in settings["angles_degrees"]:
                yield (f"p2_k{kappa}_i{init_id}_a{angle}", d, rotation_2d(np.deg2rad(angle)),
                       np.random.default_rng(child), "well_separated", angle)
    for p in settings["higher_dimensions"]:
        for kappa in settings["condition_numbers"]:
            d = np.geomspace(1.0 / kappa, 1.0, p)
            for rotation_id in range(settings["higher_dimensional_rotations"][str(p)]):
                q = random_orthogonal(p, np.random.default_rng(stream.spawn(1)[0]))
                for init_id in range(settings["higher_dimensional_initializations"]):
                    yield (f"p{p}_k{kappa}_r{rotation_id}_i{init_id}", d, q,
                           np.random.default_rng(stream.spawn(1)[0]), "well_separated", None)
    yield ("repeated_identity", np.ones(2), rotation_2d(np.pi / 4),
           np.random.default_rng(901), "repeated", 45)
    yield ("near_repeated", np.array([1e-4, 1e-4 + 1e-11, 1.0]),
           random_orthogonal(3, np.random.default_rng(55019)),
           np.random.default_rng(902), "near_repeated", None)


def verification_suite(settings, output):
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows, diagnoses = [], []
    config = RunConfig("adam", settings["learning_rate"], settings["max_steps"], tolerance=settings["tolerance"],
                       beta1=settings["beta1"], beta2=settings["beta2"], epsilon=settings["epsilon"])
    for case_id, d, q, rng, scope, angle in case_records(settings):
        base, rotated, z0, w0 = matched_pair(d, q, rng)
        canonical_positions, canonical_losses = full_trajectory(base, z0, config)
        expected_positions = canonical_positions @ q.T
        denominator = max(1.0, float(np.linalg.norm(w0)))
        arrays = {"eigenvalues": d, "rotation": q, "canonical_initial": z0,
                  "canonical_positions": canonical_positions, "canonical_losses": canonical_losses}
        for name, source in (("standard_adam", None), ("oracle_basis", "oracle"), ("recovered_basis", "eigh")):
            prepared = None if source is None else prepare_eigenbasis(rotated, source)
            positions, losses = full_trajectory(rotated, w0, config, prepared)
            errors = np.linalg.norm(positions - expected_positions, axis=1) / denominator
            required = source == "oracle" or (source == "eigh" and scope == "well_separated")
            maximum_error = float(np.max(errors))
            passed = bool(maximum_error <= settings["position_error_tolerance"]) if required else None
            if required and not passed:
                raise RuntimeError(f"Correspondence gate failed: {case_id}/{name}, error={maximum_error:g}")
            diagnostic = prepared.diagnostics if prepared is not None else {}
            if scope in ("repeated", "near_repeated") and source == "eigh":
                if not diagnostic["eigenvectors_numerically_uncertain"]:
                    raise RuntimeError(f"Missing eigengap uncertainty flag: {case_id}")
            rows.append({"case_id": case_id, "scope": scope, "dimension": len(d),
                         "condition_number": float(np.max(d) / np.min(d)), "angle_degrees": angle,
                         "method": name, "learning_rate": config.learning_rate,
                         "gradient_calls": config.max_steps, "maximum_normalized_position_error": maximum_error,
                         "maximum_absolute_objective_difference": float(np.max(np.abs(losses - canonical_losses))),
                         "final_relative_loss": float(losses[-1] / losses[0]),
                         "correspondence_required": required, "correspondence_passed": passed,
                         "eigenvectors_numerically_uncertain": diagnostic.get("eigenvectors_numerically_uncertain")})
            arrays[name + "_positions"] = positions
            arrays[name + "_losses"] = losses
            arrays[name + "_errors"] = errors
            if prepared is not None:
                arrays[name + "_basis"] = prepared.vectors
                diagnoses.append({"case_id": case_id, "source": prepared.source, **diagnostic})
        np.savez_compressed(raw / f"{case_id}.npz", **arrays)
    write_csv(output / "correspondence.csv", rows)
    write_json(output / "basis_diagnostics.json", diagnoses)
    return rows


def timing_suite(settings, output):
    rows = []
    for p in settings["benchmark_dimensions"]:
        rng = np.random.default_rng(settings["seed"] + p)
        d = np.geomspace(.01, 1.0, p)
        q = random_orthogonal(p, rng)
        _, problem, _, w0 = matched_pair(d, q, rng)
        for name, source in (("standard_adam", "oracle"), ("oracle_basis", "oracle"), ("recovered_basis", "eigh")):
            algorithm = "adam" if name == "standard_adam" else "eigenbasis_adam"
            config = RunConfig(algorithm, settings["learning_rate"], settings["benchmark_steps"], basis_source=source,
                               beta1=settings["beta1"], beta2=settings["beta2"], epsilon=settings["epsilon"])
            run(problem, w0, config)  # Deliberate warm-up, not a measured repetition.
            for repetition in range(settings["benchmark_repetitions"]):
                result = run(problem, w0, config)
                meta = result.metadata
                expected_transforms = 0 if algorithm == "adam" else 2 * result.gradient_calls
                if meta["basis_transform_calls"] != expected_transforms:
                    raise RuntimeError("Incorrect basis-transform accounting")
                if result.status != "finite_cap" or result.gradient_calls != config.max_steps:
                    raise RuntimeError("Timing must compare full finite budgets")
                rows.append({"dimension": p, "method": name, "repetition": repetition,
                             "warmup_performed_by_harness": True,
                             "gradient_calls": result.gradient_calls, "status": result.status,
                             **{key: meta[key] for key in (
                                 "setup_seconds", "evaluation_seconds", "total_seconds",
                                 "basis_transform_calls", "gradient_matrix_vector_products",
                                 "loss_matrix_vector_products", "setup_eigendecomposition_calls",
                                 "setup_validation_matrix_matrix_products")}})
    write_csv(output / "timing.csv", rows)
    return rows


def make_figures(output, verification_rows, timing_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Plotting needs the plots extra: python -m pip install -e '.[plots]'") from error
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    angles = [0, 15, 30, 45, 60, 75, 90]
    for method, marker, style in (("standard_adam", "o", "-"), ("oracle_basis", "s", "--"), ("recovered_basis", "^", ":")):
        values = []
        for angle in angles:
            selected = [r["maximum_normalized_position_error"] for r in verification_rows
                        if r["dimension"] == 2 and r["scope"] == "well_separated" and
                        r["condition_number"] == 100 and r["angle_degrees"] == angle and r["method"] == method]
            values.append(max(float(np.median(selected)), 1e-17))
        axes[0].semilogy(angles, values, marker=marker, linestyle=style, label=method.replace("_", " "))
    axes[0].set(xlabel="Rotation angle (degrees)", ylabel="Maximum normalized path discrepancy",
                title="Fixed spectrum, $p=2$, $\\kappa=100$")
    axes[0].legend(fontsize=8)
    special = np.load(output / "raw" / "near_repeated.npz")
    for name, style in (("oracle_basis", "--"), ("recovered_basis", "-")):
        axes[1].semilogy(np.maximum(special[name + "_errors"], 1e-17), linestyle=style, label=name.replace("_", " "))
    axes[1].set(xlabel="Gradient evaluations", ylabel="Normalized path discrepancy",
                title="Near-repeated eigengap control")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "correspondence.pdf", bbox_inches="tight")
    figure.savefig(output / "correspondence.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6.5, 4))
    dimensions = sorted({row["dimension"] for row in timing_rows})
    for name, marker in (("standard_adam", "o"), ("oracle_basis", "s"), ("recovered_basis", "^")):
        median = [np.median([row["total_seconds"] for row in timing_rows if row["method"] == name and row["dimension"] == p]) for p in dimensions]
        axis.plot(dimensions, median, marker=marker, label=name.replace("_", " "))
    axis.set(xlabel="Dimension", ylabel="Median seconds (setup + complete run)",
             title="Local cost check; fixed rates, no tuning")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "cost.pdf", bbox_inches="tight")
    figure.savefig(output / "cost.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def source_manifest(root):
    hashes = {}
    for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("configs", "*.json"), ("scripts", "*.ps1")):
        for path in sorted((root / folder).rglob(pattern)):
            hashes[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, True
    return {"git_revision": revision, "working_tree_dirty": dirty,
            "hash_text_newline_normalization": "CRLF to LF for cross-platform Git checkouts", "sha256": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/part5.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/part5"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    if settings.get("study_status") != "module5_structural_verification":
        parser.error("This command is only for Module 5 structural verification")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "config.json", settings)
    verification_rows = verification_suite(settings, args.output)
    timing_rows = timing_suite(settings, args.output)
    required = [r for r in verification_rows if r["correspondence_required"]]
    by_method = {method: max(r["maximum_normalized_position_error"] for r in required if r["method"] == method)
                 for method in ("oracle_basis", "recovered_basis")}
    summary = {
        "study_status": settings["study_status"], "main_study_completed": False,
        "case_count": len(verification_rows) // 3, "trajectory_count": len(verification_rows),
        "required_correspondence_checks": len(required), "all_required_checks_passed": all(r["correspondence_passed"] for r in required),
        "maximum_required_path_error_by_method": by_method,
        "timing_repetitions": len(timing_rows), "scope_controls": [r for r in verification_rows if r["scope"] != "well_separated"],
        "timing_medians": [{"dimension": p, "method": method,
                            **{key: float(np.median([r[key] for r in timing_rows if r["dimension"] == p and r["method"] == method]))
                               for key in ("setup_seconds", "evaluation_seconds", "total_seconds")}}
                           for p in settings["benchmark_dimensions"]
                           for method in ("standard_adam", "oracle_basis", "recovered_basis")],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                        "blas_thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")}},
        "utc_timestamp": datetime.now(timezone.utc).isoformat(), "source": source_manifest(Path(__file__).resolve().parents[2]),
        "interpretation": "Fixed-rate structural verification only. Numerical correspondence is not a proof of faster optimization; timings are local and include validation overhead."
    }
    write_json(args.output / "summary.json", summary)
    if not args.no_plots:
        make_figures(args.output, verification_rows, timing_rows)
    print(json.dumps({key: summary[key] for key in ("case_count", "trajectory_count", "required_correspondence_checks", "all_required_checks_passed", "maximum_required_path_error_by_method")}, indent=2))
    print("Module 5 artifacts:", args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
