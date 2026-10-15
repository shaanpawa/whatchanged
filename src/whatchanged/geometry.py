"""Small geometry helpers: similarity alignment, floor levelling, voxels, projection, PLY output."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with dst ≈ s * R @ src + t (Umeyama 1991). src, dst: (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return s * pts @ R.T + t


def transform_poses(s: float, R: np.ndarray, t: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Apply a similarity to cam2world poses (N, 4, 4). Rotation part stays orthonormal."""
    out = poses.copy()
    out[:, :3, :3] = R @ poses[:, :3, :3]
    out[:, :3, 3] = s * poses[:, :3, 3] @ R.T + t
    return out


def fit_floor(points: np.ndarray, up_hint: np.ndarray, cam_centres: np.ndarray,
              iters: int = 400, thresh: float = 0.02, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC the largest plane roughly perpendicular to up_hint that lies below the cameras.

    Returns (unit normal pointing up, offset d) with plane n·x + d = 0.
    """
    rng = np.random.default_rng(seed)
    up_hint = up_hint / np.linalg.norm(up_hint)
    pts = points if len(points) < 200_000 else points[rng.choice(len(points), 200_000, replace=False)]
    best, best_count = None, -1
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n @ up_hint < 0:
            n = -n
        if n @ up_hint < np.cos(np.radians(30)):
            continue
        d = -n @ p[0]
        if np.median(cam_centres @ n + d) < 0.3:   # cameras must be well above the floor
            continue
        count = int((np.abs(pts @ n + d) < thresh).sum())
        if count > best_count:
            best, best_count = (n, d), count
    if best is None:
        raise RuntimeError("could not find a floor plane; film more of the floor")
    n, d = best
    inl = pts[np.abs(pts @ n + d) < thresh]
    c = inl.mean(0)
    _, vecs = np.linalg.eigh(np.cov((inl - c).T))   # smallest-eigenvalue axis = plane normal
    n2 = vecs[:, 0] if vecs[:, 0] @ n > 0 else -vecs[:, 0]
    return n2, float(-n2 @ c)


def levelling_transform(n: np.ndarray, d: float, forward_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that put the floor at z = 0 with +z up."""
    z = n / np.linalg.norm(n)
    x = forward_hint - (forward_hint @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])          # rows: new axes in old coords
    t = np.array([0.0, 0.0, d])      # new_z = n·p + d
    return R, t


def voxelize(points: np.ndarray, colors: np.ndarray | None, size: float) -> dict:
    """Average points (and colours) per voxel. Returns centres, colours, counts and integer keys."""
    keys = np.floor(points / size).astype(np.int64)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    centres = np.zeros((len(uniq), 3))
    np.add.at(centres, inv, points)
    centres /= counts[:, None]
    out = {"keys": uniq, "centres": centres, "counts": counts}
    if colors is not None:
        col = np.zeros((len(uniq), 3))
        np.add.at(col, inv, colors.astype(np.float64))
        out["colors"] = (col / counts[:, None]).astype(np.uint8)
    return out


def project(points: np.ndarray, cam2world: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points → pixel coords (N, 2) and camera-frame depth z (N,). OpenCV convention."""
    w2c = np.linalg.inv(cam2world)
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc[:, :2] / z[:, None]) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    return uv, z


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    path = Path(path)
    n = len(points)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        dtype += [("r", "u1"), ("g", "u1"), ("b", "u1")]
    header.append("end_header")
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    if colors is not None:
        arr["r"], arr["g"], arr["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode())
        f.write(arr.tobytes())
