"""Shared optimizer and result contract. Coordinate this file before changing it."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

ALGORITHMS = ("gd", "heavy_ball", "adam", "eigenbasis_adam", "diagonal_gd")


@dataclass(frozen=True)
class RunConfig:
    algorithm: str
    learning_rate: float
    max_steps: int
    beta: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    tolerance: float = 1e-6
    explosive_threshold: float = 1e12
    record_every: int = 1
    basis_source: str = "oracle"

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {self.algorithm!r}; choose {ALGORITHMS}")
        if self.basis_source not in ("oracle", "eigh"):
            raise ValueError("basis_source must be 'oracle' or 'eigh'")
        if self.algorithm != "eigenbasis_adam" and self.basis_source != "oracle":
            raise ValueError("basis_source='eigh' is only meaningful for eigenbasis_adam")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not isinstance(self.record_every, int) or self.record_every < 1:
            raise ValueError("record_every must be a positive integer")
        if not all(0 <= b < 1 for b in (self.beta, self.beta1, self.beta2)):
            raise ValueError("momentum coefficients must lie in [0, 1)")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not 0 < self.tolerance < 1:
            raise ValueError("tolerance must lie in (0, 1)")
        if not np.isfinite(self.explosive_threshold) or self.explosive_threshold <= 1:
            raise ValueError("explosive_threshold must be finite and greater than 1")


@dataclass
class OptimizerState:
    """Moments use ambient coordinates, except eigenbasis Adam uses eigen-coordinates."""

    position: np.ndarray
    velocity: np.ndarray
    first_moment: np.ndarray
    second_moment: np.ndarray
    step: int = 0

    @classmethod
    def initial(cls, position: np.ndarray) -> "OptimizerState":
        w = np.array(position, dtype=np.float64, copy=True)
        return cls(w, np.zeros_like(w), np.zeros_like(w), np.zeros_like(w))


@dataclass(frozen=True)
class TraceEntry:
    step: int
    loss: float | None
    relative_loss: float | None
    # This is the gradient used for the update, evaluated BEFORE that step.
    # Recording does not consume an extra gradient evaluation.
    gradient_norm_before_step: float | None


@dataclass
class RunResult:
    algorithm: str
    final_position: np.ndarray
    final_loss: float
    initial_loss: float
    status: str
    gradient_calls: int
    first_attainment: int | None
    threshold_retained: bool
    trace: list[TraceEntry]
    metadata: dict[str, Any] = field(default_factory=dict)
