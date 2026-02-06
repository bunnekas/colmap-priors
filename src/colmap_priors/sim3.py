from __future__ import annotations

import numpy as np


def umeyama(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit similarity transform mapping Y -> X:  X ≈ s R Y + t.  X,Y are (N,3)."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected X,Y shape (N,3) matching, got {X.shape} and {Y.shape}")
    n = X.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for Sim(3) alignment")

    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    Xc = X - muX
    Yc = Y - muY

    C = (Yc.T @ Xc) / n
    U, S, Vt = np.linalg.svd(C)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    varY = (Yc**2).sum() / n
    s = (S.sum() / varY) if varY > 0 else 1.0
    t = muX - s * (R @ muY)
    return float(s), R.astype(np.float64), t.astype(np.float64)


def apply_sim3(P: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return (s * (R @ P.T)).T + t.reshape(1, 3)


def apply_to_c2w(T_c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply Sim(3) to a c2w pose (rotation unaffected by scale; translation scaled)."""
    T = np.asarray(T_c2w, dtype=np.float64).reshape(4, 4)
    R_c2w = T[:3, :3]
    t_c2w = T[:3, 3]
    R_new = R @ R_c2w
    t_new = s * (R @ t_c2w) + t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_new
    out[:3, 3] = t_new
    return out
