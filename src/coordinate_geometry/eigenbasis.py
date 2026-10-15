"""Role 5 (HOU, Kaiwen): validated, reusable bases and common-basis Adam.

Recovered bases use only H. No oracle sign, permutation or eigenspace alignment is
performed. Distinct eigenbases differ by signed permutations; bases within a
repeated eigenspace can differ by general rotations, for which Adam is not
equivariant. Trajectory correspondence then requires a specified common basis.
"""

from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Any

import numpy as np

from .adam import adam_direction
from .contracts import OptimizerState, RunConfig
from .problems import QuadraticProblem


@dataclass(frozen=True, init=False)
class PreparedEigenbasis:
    """Factory-only value object, with deeply immutable mathematical arrays.

    Use prepare_eigenbasis(). Rejecting construction also prevents
    dataclasses.replace() from swapping validated arrays for unchecked ones.
    """

    vectors: np.ndarray
    eigenvalues: np.ndarray
    source: str
    setup_seconds: float
    _diagnostics_json: str = field(repr=False)
    reference_hessian: np.ndarray = field(repr=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("PreparedEigenbasis is factory-only; use prepare_eigenbasis()")

    def __copy__(self) -> "PreparedEigenbasis":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "PreparedEigenbasis":
        # A value-object copy must not recreate NumPy buffers as writable arrays.
        memo[id(self)] = self
        return self

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("PreparedEigenbasis is not pickled; save its arrays and prepare them again")

    @property
    def diagnostics(self) -> dict[str, Any]:
        # Callers may annotate their copy without mutating this basis's audit data.
        return json.loads(self._diagnostics_json)

    def validate_for(self, problem: QuadraticProblem) -> None:
        h = np.asarray(problem.hessian, dtype=np.float64)
        scale = max(float(np.max(np.abs(self.reference_hessian))), np.finfo(float).tiny)
        if h.shape != self.reference_hessian.shape or not np.all(np.isfinite(h)):
            raise ValueError("prepared basis does not match this finite square Hessian")
        if not np.allclose(h / scale, self.reference_hessian / scale, atol=1e-12, rtol=1e-12):
            raise ValueError("prepared basis belongs to a different Hessian; prepare it again")


def _cluster_diagnostics(
    values: np.ndarray,
    rtol: float,
    atol: float,
    *,
    hessian_scale: float,
    backward_error_absolute: float,
    validation_tolerance: float,
) -> dict[str, Any]:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    gaps = np.diff(sorted_values)
    eps = float(np.finfo(np.float64).eps)
    # Residuals alone can be tiny even when cancellation in H obscures a small
    # eigengap. Include both an H-scale roundoff allowance and the accepted
    # validation residual. These are conservative diagnostics, not a theorem
    # certifying the forward error of a particular LAPACK implementation.
    roundoff_allowance = (32 * eps * values.size) * hessian_scale
    numerical_error_scale = max(backward_error_absolute, roundoff_allowance)
    validation_allowance = validation_tolerance * hessian_scale
    unresolved_gap_threshold = 2 * max(numerical_error_scale, validation_allowance)
    minimum_gap = float(np.min(gaps)) if gaps.size else None
    uncertainty_indicator = (
        min(1.0, numerical_error_scale / minimum_gap) if minimum_gap is not None and minimum_gap > 0
        else 1.0 if minimum_gap == 0 else 0.0
    )
    uncertainty_target = max(validation_tolerance, float(np.sqrt(eps)))
    groups: list[list[int]] = []
    for index in order:
        i = int(index)
        if not groups:
            groups.append([i])
            continue
        previous = groups[-1][-1]
        gap = abs(float(values[i] - values[previous]))
        threshold = max(atol + rtol * max(abs(float(values[i])), abs(float(values[previous]))),
                        unresolved_gap_threshold)
        if gap <= threshold:
            groups[-1].append(i)
        else:
            groups.append([i])
    clusters = [
        {"indices": group, "size": len(group),
         "minimum": float(np.min(values[group])), "maximum": float(np.max(values[group]))}
        for group in groups
    ]
    clustered = any(len(group) > 1 for group in groups)
    uncertain = bool(clustered or uncertainty_indicator > uncertainty_target)
    return {
        "eigenvalue_clusters": clusters,
        "cluster_rtol": rtol,
        "cluster_atol": atol,
        "has_clustered_eigenvalues": clustered,
        "has_repeated_eigenspace": clustered,
        "cluster_interpretation": "conservative H-scale numerical clusters; possible repeated or unresolved eigenspaces, not proof of exact multiplicity",
        "stored_eigenvalues_all_distinct": bool(np.all(gaps > 0)),
        "minimum_absolute_eigengap": minimum_gap,
        "minimum_relative_eigengap": minimum_gap / hessian_scale if minimum_gap is not None else None,
        "hessian_spectral_scale": hessian_scale,
        "backward_error_absolute": backward_error_absolute,
        "roundoff_allowance_absolute": roundoff_allowance,
        "validation_allowance_absolute": validation_allowance,
        "unresolved_eigengap_threshold_absolute": unresolved_gap_threshold,
        "eigenvector_uncertainty_indicator": uncertainty_indicator,
        "eigenvector_uncertainty_target": uncertainty_target,
        "eigenvectors_numerically_uncertain": uncertain,
        "eigenbasis_numerically_identifiable": not uncertain,
        "uncertainty_interpretation": "H-scale residual/roundoff divided by the smallest eigengap; conservative indicator, not a certified forward-error bound",
        "mathematical_statement": "exactly distinct eigenvalues imply signed-permutation equivalence of exact eigenbases; floating-point accuracy additionally depends on eigengaps",
        "correspondence_scope": (
            "shared_basis_required_for_clustered_eigenspaces" if clustered
            else "signed_permutation_equivalence_requires_numerical_verification" if uncertain
            else "distinct_eigenbasis_up_to_signed_permutation"
        ),
    }


def prepare_eigenbasis(
    problem: QuadraticProblem,
    source: str = "oracle",
    *,
    basis: np.ndarray | None = None,
    validation_tolerance: float = 1e-10,
    cluster_rtol: float = 1e-8,
    cluster_atol: float = 1e-12,
) -> PreparedEigenbasis:
    """Prepare once before stepping, validating shape, finiteness and H B = B D.

    An explicit matrix overrides source and is labelled ``supplied``. ``eigh``
    never reads the generating Q or eigenvalues. Setup includes validation work
    as well as factorization, and is kept separate from the optimization timer.
    """
    started = perf_counter()
    if source not in ("oracle", "eigh"):
        raise ValueError("source must be 'oracle' or 'eigh'")
    tolerances = (validation_tolerance, cluster_rtol, cluster_atol)
    if not all(np.isfinite(value) and value >= 0 for value in tolerances) or validation_tolerance == 0:
        raise ValueError("tolerances must be finite and nonnegative; validation_tolerance must be positive")
    h = np.array(problem.hessian, dtype=np.float64, copy=True)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] == 0 or not np.all(np.isfinite(h)):
        raise ValueError("Hessian must be a finite nonempty square matrix")
    scale = max(float(np.max(np.abs(h))), np.finfo(float).tiny)
    scaled_h = h / scale
    if not np.allclose(scaled_h, scaled_h.T, atol=validation_tolerance, rtol=validation_tolerance):
        raise ValueError("Hessian must be symmetric")
    if basis is not None:
        vectors = np.array(basis, dtype=np.float64, copy=True)
        actual_source = "supplied"
    elif source == "eigh":
        # Keep the eigensolver's signs and ordering; no oracle alignment here.
        _, vectors = np.linalg.eigh(h)
        actual_source = "eigh"
    else:
        vectors = np.array(problem.rotation, dtype=np.float64, copy=True)
        actual_source = "oracle"
    if vectors.shape != h.shape or not np.all(np.isfinite(vectors)):
        raise ValueError("basis must be a finite square matrix matching the Hessian")
    orthogonality_error = float(np.linalg.norm(vectors.T @ vectors - np.eye(h.shape[0]), ord="fro") / np.sqrt(h.shape[0]))
    if orthogonality_error > validation_tolerance:
        raise ValueError("basis must be orthogonal")
    scaled_h_vectors = scaled_h @ vectors
    scaled_values = np.sum(vectors * scaled_h_vectors, axis=0)
    residual = scaled_h_vectors - vectors * scaled_values
    diagonalization_error = float(np.linalg.norm(residual, ord="fro") / np.linalg.norm(scaled_h, ord="fro"))
    if diagonalization_error > validation_tolerance:
        raise ValueError("basis must diagonalize this Hessian")
    values = scaled_values * scale
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("basis must give finite strictly positive eigenvalues")
    diagnostics = {
        "orthogonality_error": orthogonality_error,
        "relative_diagonalization_residual": diagonalization_error,
        "validation_tolerance": validation_tolerance,
        "oracle_alignment_performed": False,
        "eigendecomposition_calls": int(actual_source == "eigh"),
        "validation_matrix_matrix_products": 2,
        **_cluster_diagnostics(
            values, cluster_rtol, cluster_atol,
            hessian_scale=float(np.max(values)),
            backward_error_absolute=(diagonalization_error * float(np.linalg.norm(scaled_h, ord="fro"))) * scale,
            validation_tolerance=validation_tolerance,
        ),
    }
    # A normal ndarray with WRITEABLE=False can have write access re-enabled
    # when it owns the buffer. Arrays backed by immutable bytes cannot.
    immutable_arrays = [
        np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)
        for array in (vectors, values, h)
    ]
    prepared = object.__new__(PreparedEigenbasis)
    for name, value in {
        "vectors": immutable_arrays[0], "eigenvalues": immutable_arrays[1],
        "reference_hessian": immutable_arrays[2], "source": actual_source,
        "_diagnostics_json": json.dumps(diagnostics, allow_nan=False),
        "setup_seconds": perf_counter() - started,
    }.items():
        object.__setattr__(prepared, name, value)
    return prepared


def eigenbasis_adam_step(
    prepared: PreparedEigenbasis, state: OptimizerState, gradient: np.ndarray, config: RunConfig
) -> None:
    """One update using a prepared basis; exactly two dense matrix-vector transforms.

    Moments always live in prepared-basis coordinates. States cannot be reused
    with a different basis mid-run without transforming their interpretation.
    """
    basis_gradient = prepared.vectors.T @ gradient
    basis_direction = adam_direction(state, basis_gradient, config)
    state.position -= config.learning_rate * (prepared.vectors @ basis_direction)
