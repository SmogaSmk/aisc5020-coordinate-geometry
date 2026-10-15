"""Startup infrastructure; reference algorithms remain owned by their teammates."""

from .contracts import OptimizerState, RunConfig, RunResult, TraceEntry
from .eigenbasis import PreparedEigenbasis, eigenbasis_adam_step, prepare_eigenbasis
from .problems import QuadraticProblem, matched_pair, random_orthogonal, rotation_2d
from .runner import run

__all__ = [
    "OptimizerState", "RunConfig", "RunResult", "TraceEntry", "QuadraticProblem",
    "matched_pair", "random_orthogonal", "rotation_2d", "run",
    "PreparedEigenbasis", "prepare_eigenbasis", "eigenbasis_adam_step",
]
