"""Decide, for every bit of space, whether the other video saw it change.

For each voxel seen in session X, every camera of session Y that can see it compares its
own measured depth at that pixel with the voxel's distance:

    Y's depth further than the voxel  -> "free"      (Y saw through where it was)
    Y's depth about equal             -> "seen"      (still there)
    Y's depth nearer                  -> "occluded"  (something in front)

A voxel only counts as changed with enough "free" votes. Too few free-or-seen votes means
"can't tell": hidden if Y's cameras were blocked, not filmed otherwise. Running X=A, Y=B
finds removals; X=B, Y=A finds additions. A removal and an addition that look alike and
sit apart are one moved object.

The depth tolerance is set from the room's own noise: most of what both videos saw did not
change, so the spread of depth disagreements over co-visible voxels is the noise floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .geometry import project, voxelize
from .recon import Session


@dataclass
class Config:
    voxel: float = 0.03            # metres
    floor_cut: float = 0.04        # voxels below this height are floor
    min_views: int = 3             # votes needed before deciding anything
    min_support: int = 3           # own-video frames that must agree a surface was there
    free_share: float = 0.6        # share of free votes needed for "changed"
    k_sigma: float = 3.0           # tolerance = k x robust spread of depth disagreement
    tau_min: float = 0.03          # metres; never trust differences below this
    min_blob_voxels: int = 25
    min_blob_extent: float = 0.10  # metres
    max_blob_extent: float = 2.5   # metres; bigger means misalignment, not an object
    conf_pct: float = 30.0         # drop the least confident pixels
    edge_px: int = 5               # neighbourhood that must all be "further" for a free vote
    match_max_cost: float = 0.55   # removed/added pairing threshold for "moved"
    min_move: float = 0.15         # metres


@dataclass
class Blob:
    kind: str                      # "removed" or "added"
    voxels: np.ndarray             # (M, 3) centres
    colors: np.ndarray             # (M, 3) uint8
    score: float                   # mean free share
    margin: float                  # mean free-vote excess over the noise tolerance, in tolerances
    views: int                     # cameras that voted
    centre: np.ndarray = field(default=None)
    size: np.ndarray = field(default=None)      # oriented extent (length, width, height)
    yaw: float = 0.0                             # radians, footprint orientation
    bottom: float = 0.0
    top: float = 0.0

    def __post_init__(self):
        lo, hi = self.voxels.min(0), self.voxels.max(0)
        self.bottom, self.top = float(lo[2]), float(hi[2])
        xy = self.voxels[:, :2] - self.voxels[:, :2].mean(0)
        if len(xy) >= 3:
            w, v = np.linalg.eigh(np.cov(xy.T))
            axis = v[:, 1]
            self.yaw = float(np.arctan2(axis[1], axis[0]))
            rot = np.array([[np.cos(-self.yaw), -np.sin(-self.yaw)], [np.sin(-self.yaw), np.cos(-self.yaw)]])
            r = xy @ rot.T
            ext = r.max(0) - r.min(0)
            mid = (r.max(0) + r.min(0)) / 2 @ np.linalg.inv(rot).T + self.voxels[:, :2].mean(0)
        else:
            ext, mid = hi[:2] - lo[:2], (hi[:2] + lo[:2]) / 2
        self.size = np.array([ext[0], ext[1], hi[2] - lo[2]]) + 0.03
        self.centre = np.array([mid[0], mid[1], (hi[2] + lo[2]) / 2])


def _votes(X: Session, Y: Session, cfg: Config, log=print) -> dict:
    pts, cols, fids = X.points(conf_pct=cfg.conf_pct, stride=2, with_frames=True)
    vox = voxelize(pts, cols, cfg.voxel)
    C = vox["centres"]
    # How many of X's own frames put a surface in each voxel. One-frame specks are depth noise,
    # and the other video will naturally "see through" them.
    keys = np.floor(pts / cfg.voxel).astype(np.int64)
    pairs = np.unique(np.column_stack([keys, fids]), axis=0)
    _, pos = np.unique(pairs[:, :3], axis=0, return_inverse=True)
    support = np.bincount(pos.reshape(-1), minlength=len(C))
    n = len(C)
    free = np.zeros(n, np.int32)
    seen = np.zeros(n, np.int32)
    occ = np.zeros(n, np.int32)
    excess = np.zeros(n)                      # sum of (r - tau)/tau over free votes
    conf_thr = np.percentile(Y.conf[Y.mask], cfg.conf_pct)
    H, W = Y.depth.shape[1:]
    # "Free" must hold for a small pixel neighbourhood, not one pixel: points on an object's
    # outline otherwise look like they vanished from a slightly different angle.
    dmin = np.where(Y.mask & (Y.depth > 0), Y.depth, np.inf)
    dmin = np.stack([ndimage.grey_erosion(d, size=(cfg.edge_px, cfg.edge_px)) for d in dmin])

    # Pass 1: relative depth disagreement over all valid (voxel, camera) pairs → noise floor.
    samples = []
    per_cam = []
    for j in range(len(Y.depth)):
        uv, z = project(C, Y.poses[j], Y.K[j])
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = np.nonzero(ok)[0]
        uu, vv = u[idx], v[idx]
        good = Y.mask[j, vv, uu] & (Y.conf[j, vv, uu] >= conf_thr) & (Y.depth[j, vv, uu] > 0)
        idx, uu, vv = idx[good], uu[good], vv[good]
        e = (Y.depth[j, vv, uu] - z[idx]) / z[idx]
        emin = (dmin[j, vv, uu] - z[idx]) / z[idx]
        per_cam.append((idx, e, emin, z[idx]))
        samples.append(e[np.abs(e) < 0.25])
    s = np.concatenate(samples) if samples else np.zeros(1)
    med = float(np.median(s))
    sigma = float(1.4826 * np.median(np.abs(s - med)))

    # Pass 2: vote with tolerance tau(z) = max(tau_min, k * sigma * z).
    for idx, e, emin, z in per_cam:
        e, emin = e - med, emin - med
        tau = np.maximum(cfg.tau_min / z, cfg.k_sigma * sigma)
        f = emin > tau
        o = e < -tau
        sn = ~f & ~o
        np.add.at(free, idx[f], 1)
        np.add.at(seen, idx[sn], 1)
        np.add.at(occ, idx[o], 1)
        np.add.at(excess, idx[f], (e[f] - tau[f]) / tau[f])
    log(f"  noise floor: depth disagreement spread {sigma * 100:.1f}% of distance "
        f"(bias {med * 100:+.1f}%), tolerance {cfg.k_sigma:.0f}x")
    return {"centres": C, "colors": vox["colors"], "free": free, "seen": seen, "occ": occ, "support": support,
            "excess": excess, "sigma": sigma, "bias": med}


def _components(mask_pts: np.ndarray, voxel: float) -> list[np.ndarray]:
    """26-connected components of voxel centres, bridging 1-voxel gaps."""
    if len(mask_pts) == 0:
        return []
    keys = np.floor(mask_pts / voxel).astype(np.int64)
    lo = keys.min(0)
    k = keys - lo
    grid = np.zeros(k.max(0) + 1, bool)
    grid[tuple(k.T)] = True
    grown = ndimage.binary_dilation(grid, structure=np.ones((3, 3, 3)))
    labels, nlab = ndimage.label(grown, structure=np.ones((3, 3, 3)))
    lab = labels[tuple(k.T)]
    return [np.nonzero(lab == i)[0] for i in range(1, nlab + 1)]


def classify(V: dict, cfg: Config) -> dict:
    free, seen, occ = V["free"], V["seen"], V["occ"]
    voted = free + seen
    share = np.where(voted > 0, free / np.maximum(voted, 1), 0.0)
    changed = (free >= cfg.min_views) & (share >= cfg.free_share) & (V["support"] >= cfg.min_support)
    cant = voted < cfg.min_views
    hidden = cant & (occ >= cfg.min_views)
    unfilmed = cant & ~hidden
    return {"changed": changed, "hidden": hidden, "unfilmed": unfilmed, "share": share,
            "unchanged": ~changed & ~cant}


def blobs(V: dict, cls: dict, kind: str, cfg: Config) -> list[Blob]:
    C = V["centres"]
    sel = np.nonzero(cls["changed"] & (C[:, 2] > cfg.floor_cut))[0]
    out = []
    for comp in _components(C[sel], cfg.voxel):
        ids = sel[comp]
        if len(ids) < cfg.min_blob_voxels:
            continue
        ext = C[ids].max(0) - C[ids].min(0)
        if ext.max() < cfg.min_blob_extent or ext.max() > cfg.max_blob_extent:
            continue
        freev = V["free"][ids]
        out.append(Blob(kind=kind, voxels=C[ids], colors=V["colors"][ids],
                        score=float(cls["share"][ids].mean()),
                        margin=float(V["excess"][ids].sum() / max(freev.sum(), 1)),
                        views=int(np.median(V["free"][ids] + V["seen"][ids]))))
    return sorted(out, key=lambda b: -len(b.voxels))


def regions(V: dict, cls: dict, cfg: Config, min_voxels: int = 60) -> list[dict]:
    """Clusters of space the other video could not check, above the floor."""
    C = V["centres"]
    out = []
    for name in ("hidden", "unfilmed"):
        sel = np.nonzero(cls[name] & (C[:, 2] > cfg.floor_cut) & (C[:, 2] < 2.0))[0]
        for comp in _components(C[sel], cfg.voxel):
            ids = sel[comp]
            if len(ids) < min_voxels:
                continue
            P = C[ids]
            out.append({"reason": name, "centre": P.mean(0), "min": P.min(0), "max": P.max(0),
                        "voxels": P, "n": len(ids)})
    return sorted(out, key=lambda r: -r["n"])


def _hist(colors: np.ndarray) -> np.ndarray:
    h, _ = np.histogramdd(colors.astype(float), bins=(4, 4, 4), range=((0, 256),) * 3)
    h = h.ravel()
    return h / max(h.sum(), 1)


def match_moves(removed: list[Blob], added: list[Blob], cfg: Config) -> list[tuple[int, int, float]]:
    """Pair removals with additions that look like the same object (size + colour)."""
    if not removed or not added:
        return []
    cost = np.ones((len(removed), len(added)))
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            sr, sa = np.sort(r.size), np.sort(a.size)
            size_cost = float(np.mean(np.abs(sr - sa) / np.maximum(sr, sa)))
            color_cost = 1 - float(np.minimum(_hist(r.colors), _hist(a.colors)).sum())
            dist = float(np.linalg.norm(r.centre[:2] - a.centre[:2]))
            cost[i, j] = 0.5 * size_cost + 0.5 * color_cost if dist >= cfg.min_move else 1.0
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols) if cost[i, j] <= cfg.match_max_cost]


def detect(A: Session, B: Session, cfg: Config | None = None, log=print) -> dict:
    cfg = cfg or Config()
    log("Changes: did B see what A saw?")
    VA = _votes(A, B, cfg, log)
    CA = classify(VA, cfg)
    log("Changes: did A see what B saw?")
    VB = _votes(B, A, cfg, log)
    CB = classify(VB, cfg)
    removed = blobs(VA, CA, "removed", cfg)
    added = blobs(VB, CB, "added", cfg)
    moves = match_moves(removed, added, cfg)
    log(f"Changes: {len(removed)} removed, {len(added)} added, {len(moves)} of them paired as moves")
    return {"cfg": cfg, "VA": VA, "CA": CA, "VB": VB, "CB": CB, "removed": removed, "added": added,
            "moves": moves, "regions_A": regions(VA, CA, cfg), "regions_B": regions(VB, CB, cfg)}
