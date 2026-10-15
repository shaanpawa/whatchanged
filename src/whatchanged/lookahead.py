"""Where should the robot stand to settle what it can't tell?

For every region the second video could not check, try standing spots on a 25 cm floor grid
(only floor both videos saw as free), at the second video's camera height, aimed at the
region. Score each spot by the share of the region's voxels whose line of sight is clear
through the CURRENT occupancy (session B, so a new obstacle still blocks the view), within
0.5-3 m. Compare with a naive choice: the nearest camera of B turned toward the region.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import project


class Occupancy:
    def __init__(self, points: np.ndarray, size: float = 0.06):
        self.size = size
        keys = np.unique(np.floor(points / size).astype(np.int64), axis=0)
        self.lo = keys.min(0)
        k = keys - self.lo
        self.grid = np.zeros(k.max(0) + 1, bool)
        self.grid[tuple(k.T)] = True

    def blocked(self, pts: np.ndarray) -> np.ndarray:
        k = np.floor(pts / self.size).astype(np.int64) - self.lo
        ok = np.all((k >= 0) & (k < self.grid.shape), axis=-1)
        out = np.zeros(pts.shape[:-1], bool)
        out[ok] = self.grid[tuple(k[ok].T)]
        return out


def visible_fraction(cam: np.ndarray, targets: np.ndarray, occ: Occupancy, step: float = 0.04,
                     clearance: float = 0.12, fov_deg: float = 70.0, aim: np.ndarray | None = None) -> float:
    d = targets - cam
    dist = np.linalg.norm(d, axis=1)
    if aim is not None:
        cosang = (d @ aim) / np.maximum(dist, 1e-6) / np.linalg.norm(aim)
        in_fov = cosang > np.cos(np.radians(fov_deg / 2))
    else:
        in_fov = np.ones(len(d), bool)
    n = int(np.ceil(dist.max() / step))
    ts = np.linspace(0, 1, n)[:, None, None]
    samples = cam + ts * d[None]                                  # (n, M, 3)
    along = ts[:, :, 0] * dist[None]
    hit = occ.blocked(samples) & (along > 0.25) & (along < dist[None] - clearance)
    clear = ~hit.any(0)
    return float((clear & in_fov & (dist > 0.5) & (dist < 3.0)).mean())


def plan(A, B, regions: list[dict], floor_free: np.ndarray, floor_origin: np.ndarray, res: float,
         out_dir: Path, spacing: float = 0.25, max_regions: int = 4, log=print) -> list[dict]:
    pb, _ = B.points(conf_pct=30, stride=2)
    occ = Occupancy(pb[(pb[:, 2] > 0.06) & (pb[:, 2] < 2.2)])
    height = float(np.median(B.poses[:, 2, 3]))
    # Candidate standing spots: centres of free floor cells on a coarser grid.
    step = max(1, int(round(spacing / res)))
    ys, xs = np.nonzero(floor_free[::step, ::step])
    spots = np.column_stack([floor_origin[0] + (xs * step + 0.5) * res, floor_origin[1] + (ys * step + 0.5) * res,
                             np.full(len(xs), height)])
    plans = []
    rng = np.random.default_rng(0)
    picked = [r for r in regions if r["reason"] == "hidden"][:2] + [r for r in regions if r["reason"] == "unfilmed"][:2]
    for ri, r in enumerate(picked[:max_regions]):
        V = r["voxels"]
        targets = V[rng.choice(len(V), min(150, len(V)), replace=False)]
        centre = targets.mean(0)
        scores = np.array([visible_fraction(s, targets, occ, aim=centre - s) for s in spots]) if len(spots) else np.array([])
        cams_b = B.poses[:, :3, 3]
        naive_cam = cams_b[np.argmin(np.linalg.norm(cams_b - centre, axis=1))]
        naive = visible_fraction(naive_cam, targets, occ, aim=centre - naive_cam)
        if not len(scores) or scores.max() <= 0:
            continue
        best = int(np.argmax(scores))
        cam = spots[best]
        aim = centre - cam
        yaw = float(np.degrees(np.arctan2(aim[1], aim[0])))
        pitch = float(np.degrees(np.arctan2(aim[2], np.linalg.norm(aim[:2]))))
        prev = out_dir / "lookahead" / f"view{ri + 1}.jpg"
        render_preview(A, B, cam, centre, prev)
        plans.append({"id": f"v{ri + 1}", "reason": r["reason"], "region_centre_m": np.round(centre, 2).tolist(),
                      "stand_at_m": np.round(cam[:2], 2).tolist(), "camera_height_m": round(height, 2),
                      "face_yaw_deg": round(yaw), "tilt_deg": round(pitch),
                      "predicted_visible": round(float(scores[best]), 2), "naive_visible": round(naive, 2),
                      "distance_m": round(float(np.linalg.norm(aim)), 2), "preview": str(prev)})
        log(f"Look next {ri + 1}: stand at ({cam[0]:.2f}, {cam[1]:.2f}) m, would see {scores[best]:.0%} "
            f"of the {'hidden' if r['reason'] == 'hidden' else 'unfilmed'} region (nearest B camera: {naive:.0%})")
    (out_dir / "viewpoints.json").write_text(json.dumps(plans, indent=1))
    return plans


def render_preview(A, B, cam: np.ndarray, target: np.ndarray, path: Path, W: int = 480, H: int = 320) -> None:
    """What the robot would see from there, rendered from the fused point clouds (B over A)."""
    pa, ca = A.points(conf_pct=30, stride=2)
    pb, cb = B.points(conf_pct=30, stride=2)
    pts, cols = np.concatenate([pb, pa]), np.concatenate([cb, (ca * 0.6).astype(np.uint8)])
    z = target - cam
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, cam
    f = W / (2 * np.tan(np.radians(35)))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    uv, d = project(pts, P, K)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (d > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d, c = u[ok], v[ok], d[ok], cols[ok]
    order = np.argsort(-d)                      # far first, near overwrite
    img = np.full((H, W, 3), 18, np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = np.clip(u[order] + du, 0, W - 1), np.clip(v[order] + dv, 0, H - 1)
            img[vv, uu] = c[order]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, quality=88)
