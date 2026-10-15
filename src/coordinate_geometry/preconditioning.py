"""Role 4 (HUANG, Ze): diagonal preconditioning and P5 reference calculations."""

import numpy as np

from .contracts import OptimizerState, RunConfig
from .problems import QuadraticProblem


def diagonal_gd_step(
    problem: QuadraticProblem, state: OptimizerState, gradient: np.ndarray, config: RunConfig
) -> None:
    state.position -= config.learning_rate * gradient / np.diag(problem.hessian)
    state.step += 1


def effective_hessian(hessian: np.ndarray) -> np.ndarray:
    h = np.asarray(hessian, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or not np.all(np.isfinite(h)):
        raise ValueError("hessian must be a finite square matrix")
    if not np.allclose(h, h.T) or np.min(np.linalg.eigvalsh(h)) <= 0:
        raise ValueError("hessian must be symmetric positive definite")
    scale = 1 / np.sqrt(np.diag(h))
    return scale[:, None] * h * scale[None, :]


def p5_condition_number(hessian: np.ndarray) -> float:
    """Exact two-dimensional kappa_eff = (1+r)/(1-r)."""
    h = np.asarray(hessian, dtype=np.float64)
    if h.shape != (2, 2):
        raise ValueError("P5 requires a 2 by 2 hessian")
    effective_hessian(h)  # Validate the SPD assumption.
    r = abs(h[0, 1]) / np.sqrt(h[0, 0] * h[1, 1])
    return float((1 + r) / (1 - r))
