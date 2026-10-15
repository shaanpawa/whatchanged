"""Align session B onto session A using only structure that did not change.

Both sessions are already levelled (floor at z = 0, +z up), so what is left is a 4-DoF rigid
motion: a turn about the vertical axis and a shift. Point-to-plane ICP with a robust (Tukey)
weight down-weights anything that moved, so walls, floor edges and large furniture drive the fit.
Running it again after the changed regions are masked removes their pull completely.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def normals(points: np.ndarray, k: int = 12) -> np.ndarray:
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k)
    nb = points[idx] - points[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    _, vecs = np.linalg.eigh(cov)
    return vecs[:, :, 0]                      # smallest-eigenvalue direction


def _rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def icp_4dof(src: np.ndarray, dst: np.ndarray, iters: int = 40, max_dist: float = 0.10,
             tukey: float = 0.03, keep: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Return the 4x4 transform moving src onto dst (yaw + xyz only) and fit statistics."""
    dst_n = normals(dst)
    tree = cKDTree(dst)
    if keep is not None:
        src = src[keep]
    T = np.eye(4)
    cur = src.copy()
    for it in range(iters):
        d, j = tree.query(cur, k=1, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        p, q, n = cur[ok], dst[j[ok]], dst_n[j[ok]]
        r = np.einsum("ij,ij->i", n, p - q)                    # signed point-to-plane residual
        w = np.where(np.abs(r) < tukey, (1 - (r / tukey) ** 2) ** 2, 0.0)
        # d(residual)/d(theta, tx, ty, tz) for a small turn about z.
        J = np.column_stack([n[:, 0] * -p[:, 1] + n[:, 1] * p[:, 0], n[:, 0], n[:, 1], n[:, 2]])
        A = J.T @ (J * w[:, None]) + 1e-6 * np.eye(4)
        b = -J.T @ (w * r)
        x = np.linalg.solve(A, b)
        dT = np.eye(4)
        dT[:3, :3] = _rz(x[0])
        dT[:3, 3] = x[1:]
        cur = cur @ dT[:3, :3].T + dT[:3, 3]
        T = dT @ T
        if np.abs(x).max() < 1e-5:
            break
    d, _ = tree.query(cur, k=1)
    stats = {"iterations": it + 1, "yaw_deg": round(float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))), 3),
             "shift_m": np.round(T[:3, 3], 4).tolist(),
             "residual_p50_m": round(float(np.percentile(d, 50)), 4),
             "residual_p90_m": round(float(np.percentile(d, 90)), 4),
             "inlier_share_3cm": round(float((d < 0.03).mean()), 3)}
    return T, stats


def nn_stats(a: np.ndarray, b: np.ndarray, n: int = 50_000) -> dict:
    rng = np.random.default_rng(0)
    sb = b[rng.choice(len(b), min(n, len(b)), replace=False)]
    d, _ = cKDTree(a).query(sb, k=1)
    return {"p50": round(float(np.percentile(d, 50)), 4), "p90": round(float(np.percentile(d, 90)), 4)}


def subsample(p: np.ndarray, n: int = 60_000) -> np.ndarray:
    if len(p) <= n:
        return p
    return p[np.random.default_rng(0).choice(len(p), n, replace=False)]
