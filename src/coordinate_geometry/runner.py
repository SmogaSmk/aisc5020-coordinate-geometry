"""Shared fixed-budget runner. Loss bookkeeping does not spend gradient calls."""

from time import perf_counter

import numpy as np

from .adam import adam_step
from .contracts import OptimizerState, RunConfig, RunResult, TraceEntry
from .eigenbasis import PreparedEigenbasis, eigenbasis_adam_step, prepare_eigenbasis
from .gd_momentum import gd_step, heavy_ball_step
from .preconditioning import diagonal_gd_step
from .problems import QuadraticProblem


def run(
    problem: QuadraticProblem,
    initial: np.ndarray,
    config: RunConfig,
    *,
    basis: np.ndarray | PreparedEigenbasis | None = None,
) -> RunResult:
    """Run a finite budget; basis preparation occurs once, outside the loop.

    Supplying ``basis`` is meaningful only for eigenbasis_adam. A prepared object
    can be reused with the same Hessian; its original preparation time is not
    charged again, and per-run validation remains part of setup_seconds.
    """
    total_started = perf_counter()
    if basis is not None and config.algorithm != "eigenbasis_adam":
        raise ValueError("an explicit basis is only meaningful for eigenbasis_adam")
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (problem.dimension,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial must be a finite vector of the problem dimension")
    initial_loss = problem.loss(initial)
    if not np.isfinite(initial_loss) or initial_loss <= 0:
        raise ValueError("initial loss must be finite and strictly positive")
    prepared = None
    setup_seconds = 0.0
    setup_reused = False
    if config.algorithm == "eigenbasis_adam":
        setup_started = perf_counter()
        if isinstance(basis, PreparedEigenbasis):
            basis.validate_for(problem)
            prepared = basis
            setup_reused = True
        else:
            prepared = prepare_eigenbasis(problem, config.basis_source, basis=basis)
        setup_seconds = perf_counter() - setup_started
    state = OptimizerState.initial(initial)
    trace = [TraceEntry(0, initial_loss, 1.0, None)]
    first_attainment = None
    status = "finite_cap"
    last_finite_position = initial.copy()
    last_finite_loss = initial_loss
    calls = 0
    started = perf_counter()
    for step in range(1, config.max_steps + 1):
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            gradient = problem.gradient(state.position)
            calls += 1
            gradient_norm = float(np.linalg.norm(gradient))
            if config.algorithm == "gd":
                gd_step(state, gradient, config)
            elif config.algorithm == "heavy_ball":
                heavy_ball_step(state, gradient, config)
            elif config.algorithm == "adam":
                adam_step(state, gradient, config)
            elif config.algorithm == "eigenbasis_adam":
                eigenbasis_adam_step(prepared, state, gradient, config)
            else:
                diagonal_gd_step(problem, state, gradient, config)
            loss = problem.loss(state.position)
            relative_loss = loss / initial_loss
        finite = all(np.all(np.isfinite(array)) for array in (
            gradient, state.position, state.velocity, state.first_moment, state.second_moment
        )) and np.isfinite(loss)
        if not finite:
            status = "nonfinite"
            trace.append(TraceEntry(step, None, None,
                                    gradient_norm if np.isfinite(gradient_norm) else None))
            break
        last_finite_position = state.position.copy()
        last_finite_loss = loss
        if first_attainment is None and relative_loss <= config.tolerance:
            first_attainment = step
        explosive = relative_loss > config.explosive_threshold
        if step % config.record_every == 0 or step == config.max_steps or explosive:
            trace.append(TraceEntry(step, loss, relative_loss,
                                    gradient_norm if np.isfinite(gradient_norm) else None))
        if explosive:
            status = "explosive"
            break
    evaluation_seconds = perf_counter() - started
    total_seconds = perf_counter() - total_started
    return RunResult(
        algorithm=config.algorithm,
        final_position=last_finite_position,
        final_loss=last_finite_loss,
        initial_loss=initial_loss,
        status=status,
        gradient_calls=calls,
        first_attainment=first_attainment,
        threshold_retained=(status == "finite_cap" and last_finite_loss / initial_loss <= config.tolerance),
        trace=trace,
        metadata={"elapsed_seconds": evaluation_seconds,
                  "evaluation_seconds": evaluation_seconds,
                  "setup_seconds": setup_seconds,
                  "total_seconds": total_seconds,
                  "basis_source": prepared.source if prepared is not None else "ambient",
                  "basis_setup_reused": setup_reused,
                  "basis_original_preparation_seconds": prepared.setup_seconds if prepared is not None else 0.0,
                  "basis_diagnostics": prepared.diagnostics if prepared is not None else None,
                  "gradient_matrix_vector_products": calls,
                  "basis_transform_calls": 2 * calls if prepared is not None else 0,
                  "loss_matrix_vector_products": calls + 1,
                  "setup_eigendecomposition_calls": int(prepared is not None and prepared.source == "eigh" and not setup_reused),
                  "setup_validation_matrix_matrix_products": 0 if prepared is None or setup_reused else 2,
                  "timing_scope": "single run, no warm-up; evaluation includes gradients, transforms, losses, checks and trace bookkeeping",
                  "warmup_performed": False,
                  "stop_at_threshold": False,
                  "final_position_is_last_finite": True},
    )
