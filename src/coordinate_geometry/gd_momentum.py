"""Role 1 (CHENG, Zixuan): GD and heavy-ball reference updates, ready to refine."""

import numpy as np

from .contracts import OptimizerState, RunConfig


def gd_step(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> None:
    state.position -= config.learning_rate * gradient
    state.step += 1


def heavy_ball_step(state: OptimizerState, gradient: np.ndarray, config: RunConfig) -> None:
    # v_(t+1) = beta v_t + gradient(w_t); w_(t+1) = w_t - eta v_(t+1).
    state.velocity = config.beta * state.velocity + gradient
    state.position -= config.learning_rate * state.velocity
    state.step += 1
