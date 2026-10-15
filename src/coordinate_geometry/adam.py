"""Role 2 (坤): Adam and gradient-history replay."""

import numpy as np

from .contracts import OptimizerState, RunConfig


def adam_direction(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> np.ndarray:
    """Update moments/time and return bias-corrected direction; do not move position."""
    state.step += 1
    state.first_moment = config.beta1 * state.first_moment + (1 - config.beta1) * gradient
    state.second_moment = config.beta2 * state.second_moment + (1 - config.beta2) * gradient**2
    m_hat = state.first_moment / (1 - config.beta1**state.step)
    s_hat = state.second_moment / (1 - config.beta2**state.step)
    return m_hat / (np.sqrt(s_hat) + config.epsilon)


def adam_step(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> None:
    state.position -= config.learning_rate * adam_direction(state, gradient, config)


def replay_directions(gradients: np.ndarray, config: RunConfig) -> np.ndarray:
    """Reconstruct A_t from a supplied (time, coordinate) history, without optimization."""
    history = np.asarray(gradients, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] == 0 or not np.all(np.isfinite(history)):
        raise ValueError("gradients must be a nonempty finite time-by-coordinate matrix")
    state = OptimizerState.initial(np.zeros(history.shape[1]))
    return np.stack([adam_direction(state, g, config) for g in history])


def replay_defect(gradients: np.ndarray, rotation: np.ndarray, config: RunConfig) -> dict[str, np.ndarray]:
    history = np.asarray(gradients, dtype=np.float64)
    base = replay_directions(history, config)
    transformed = replay_directions(history @ rotation.T, config)
    numerator = np.linalg.norm(transformed - base @ rotation.T, axis=1)
    denominator = np.linalg.norm(base, axis=1)
    return {"numerator": numerator, "denominator": denominator,
            "relative_defect": numerator / (denominator + 1e-12)}
