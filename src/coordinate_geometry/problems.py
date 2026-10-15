"""Role 3 (@Sans): generators and matched initialization."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class QuadraticProblem:
    """f(w)=w.T H w / 2, H=Q diag(eigenvalues) Q.T, minimizer zero."""

    eigenvalues: np.ndarray
    rotation: np.ndarray
    hessian: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        d = np.array(self.eigenvalues, dtype=np.float64, copy=True)
        q = np.array(self.rotation, dtype=np.float64, copy=True)
        if d.ndim != 1 or d.size == 0 or not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError("eigenvalues must be a nonempty vector of finite positive values")
        if q.shape != (d.size, d.size) or not np.all(np.isfinite(q)):
            raise ValueError("rotation must be a finite p by p matrix")
        if not np.allclose(q.T @ q, np.eye(d.size), atol=1e-10, rtol=1e-10):
            raise ValueError("rotation must be orthogonal")
        h = (q * d) @ q.T
        for array in (d, q, h):
            array.setflags(write=False)
        object.__setattr__(self, "eigenvalues", d)
        object.__setattr__(self, "rotation", q)
        object.__setattr__(self, "hessian", h)

    @property
    def dimension(self) -> int:
        return self.eigenvalues.size

    @property
    def condition_number(self) -> float:
        return float(np.max(self.eigenvalues) / np.min(self.eigenvalues))

    def loss(self, position: np.ndarray) -> float:
        return float(0.5 * position @ self.hessian @ position)

    def gradient(self, position: np.ndarray) -> np.ndarray:
        return self.hessian @ position

    def least_squares_design(self, left_basis: np.ndarray) -> np.ndarray:
        """Return X with X.T X / n = H, given orthonormal n-by-p columns U."""
        u = np.asarray(left_basis, dtype=np.float64)
        if u.ndim != 2 or u.shape[1] != self.dimension or u.shape[0] < self.dimension:
            raise ValueError("left_basis must have shape (n, p), n >= p")
        if not np.allclose(u.T @ u, np.eye(self.dimension), atol=1e-10, rtol=1e-10):
            raise ValueError("left_basis must have orthonormal columns")
        return np.sqrt(u.shape[0]) * (u * np.sqrt(self.eigenvalues)) @ self.rotation.T


def rotation_2d(angle_radians: float) -> np.ndarray:
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian QR with diagonal-sign correction; caller owns the random stream."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    return q * signs


def matched_pair(
    eigenvalues: np.ndarray,
    rotation: np.ndarray,
    rng: np.random.Generator,
    initial_loss: float = 1.0,
) -> tuple[QuadraticProblem, QuadraticProblem, np.ndarray, np.ndarray]:
    """Return diagonal/rotated problems and z0,w0=Qz0 at a shared initial loss."""
    if not np.isfinite(initial_loss) or initial_loss <= 0:
        raise ValueError("initial_loss must be finite and positive")
    rotated = QuadraticProblem(eigenvalues, rotation)
    canonical = QuadraticProblem(rotated.eigenvalues, np.eye(rotated.dimension))
    u = rng.normal(size=rotated.dimension)
    energy = float(u @ (rotated.eigenvalues * u))
    if energy == 0:
        raise ValueError("sampled zero initial direction; use a different seed")
    z0 = np.sqrt(2 * initial_loss / energy) * u
    return canonical, rotated, z0, rotated.rotation @ z0
