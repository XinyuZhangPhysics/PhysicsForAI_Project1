from __future__ import annotations
import hashlib, math
from pathlib import Path
import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def array_sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def central_norm_np(X: np.ndarray, U: np.ndarray, mu: float) -> float:
    return math.sqrt(mu * (float(np.vdot(X, X).real) + float(np.vdot(U, U).real)))


def raw_norm_np(X: np.ndarray, U: np.ndarray) -> float:
    return math.sqrt(float(np.vdot(X, X).real) + float(np.vdot(U, U).real))


def make_generic_direction(seed: int, x_shape: tuple[int, ...], u_shape: tuple[int, ...], mu: float):
    rng = np.random.default_rng(seed)
    gx = rng.standard_normal(x_shape, dtype=np.float64)
    gu = rng.standard_normal(u_shape, dtype=np.float64)
    gx /= np.linalg.norm(gx.ravel())
    gu /= np.linalg.norm(gu.ravel())
    X = gx / math.sqrt(2.0 * mu)
    U = gu / math.sqrt(2.0 * mu)
    return X, U


def aligned_central_amplitude(a0: float, mu: float) -> float:
    return a0 * math.sqrt(mu)


def linear_step_numpy(X, U, mu, lam, eta_x, eta_u, ct_action, c_action):
    # Archived matrix-free operator C equals -H_UX. Therefore
    # L has off-diagonal -eta H = +eta C.
    return (
        (1.0 - eta_x * mu) * X + eta_x * ct_action(U),
        (1.0 - eta_u * lam) * U + eta_u * c_action(X),
    )
