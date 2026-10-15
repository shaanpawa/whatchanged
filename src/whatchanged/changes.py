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
    max_range: float = 3.5         # metres; depth further from the camera is too uncertain to vote
    noise_bins: int = 6            # distance bands for the noise floor
    min_fit: float = 0.5           # share of a moved object's shape that must match after the turn and shift
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
    # Noise floor per distance band: depth disagreement over (mostly unchanged) co-visible space.
    allz = np.concatenate([z for _, _, _, z in per_cam]) if per_cam else np.ones(1)
    alle = np.concatenate([e for _, e, _, _ in per_cam]) if per_cam else np.zeros(1)
    keep = (np.abs(alle) < 0.5) & (allz <= cfg.max_range)
    med = float(np.median(alle[keep])) if keep.any() else 0.0
    edges = np.quantile(allz[keep], np.linspace(0, 1, cfg.noise_bins + 1)) if keep.sum() > 100 else np.array([0, 99.0])
    sig = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = keep & (allz >= lo) & (allz <= hi)
        e = alle[m] - med
        sig.append(float(1.4826 * np.median(np.abs(e))) if m.sum() > 50 else np.nan)
    sig = np.maximum.accumulate(np.nan_to_num(np.array(sig), nan=np.nanmax(sig) if np.isfinite(sig).any() else 0.03))
    sigma = float(np.median(sig))

    def sigma_at(z):
        return sig[np.clip(np.searchsorted(edges[1:-1], z), 0, len(sig) - 1)]

    # Pass 2: vote with tolerance tau(z) = max(tau_min, k * sigma(z) * z), only within max_range.
    for idx, e, emin, z in per_cam:
        near = z <= cfg.max_range
        idx, e, emin, z = idx[near], e[near] - med, emin[near] - med, z[near]
        tau = np.maximum(cfg.tau_min / z, cfg.k_sigma * sigma_at(z))
        f = emin > tau
        o = e < -tau
        sn = ~f & ~o
        np.add.at(free, idx[f], 1)
        np.add.at(seen, idx[sn], 1)
        np.add.at(occ, idx[o], 1)
        np.add.at(excess, idx[f], (emin[f] - tau[f]) / tau[f])
    budget = [{"range_m": [round(float(lo), 2), round(float(hi), 2)], "sigma_pct": round(float(sg) * 100, 2),
               "min_detectable_cm": round(float(max(cfg.tau_min, cfg.k_sigma * sg * hi)) * 100, 1)}
              for lo, hi, sg in zip(edges[:-1], edges[1:], sig)]
    log(f"  noise floor: {sig.min() * 100:.1f}-{sig.max() * 100:.1f}% of distance by range (bias {med * 100:+.1f}%), "
        f"tolerance {cfg.k_sigma:g}x, votes within {cfg.max_range:g} m")
    return {"centres": C, "colors": vox["colors"], "free": free, "seen": seen, "occ": occ, "support": support,
            "excess": excess, "sigma": sigma, "bias": med, "error_budget": budget}


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


def _rot(theta: float) -> np.ndarray:
    c, s_ = np.cos(theta), np.sin(theta)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])


def _kabsch_xy(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Best turn about z and shift taking points p onto q (least squares, closed form)."""
    mp, mq = p.mean(0), q.mean(0)
    P, Q = p[:, :2] - mp[:2], q[:, :2] - mq[:2]
    th = float(np.arctan2((P[:, 0] * Q[:, 1] - P[:, 1] * Q[:, 0]).sum(), (P * Q).sum()))
    t = mq - _rot(th) @ mp
    return th, t


def fit_move(r: Blob, a: Blob, voxel: float) -> dict:
    """Register the removed shape onto the added shape: turn about z + shift, from 12 starting turns."""
    from scipy.spatial import cKDTree
    src = r.voxels - r.voxels.mean(0)
    dst = a.voxels - a.voxels.mean(0)
    tree = cKDTree(dst)
    tol = 1.5 * voxel
    results = []
    for k in range(12):
        th, t = k * np.pi / 6, np.zeros(3)
        for _ in range(20):
            cur = src @ _rot(th).T + t
            d, j = tree.query(cur, k=1, distance_upper_bound=4 * voxel)
            ok = np.isfinite(d)
            if ok.sum() < 5:
                break
            dth, dt = _kabsch_xy(cur[ok], dst[j[ok]])
            th, t = th + dth, _rot(dth) @ t + dt
            if abs(dth) < 1e-4 and np.linalg.norm(dt) < 1e-4:
                break
        cur = src @ _rot(th).T + t
        d1, _ = tree.query(cur, k=1)
        d2, _ = cKDTree(cur).query(dst, k=1)
        fit = float(min((d1 < tol).mean(), (d2 < tol).mean()))
        rmse = float(np.sqrt(np.mean(np.minimum(d1, 4 * voxel) ** 2)))
        results.append((fit, rmse, (th + np.pi) % (2 * np.pi) - np.pi))
    results.sort(key=lambda x: (-x[0], x[1]))
    best = results[0]
    rival = max((f for f, _, th in results[1:] if abs((th - best[2] + np.pi) % (2 * np.pi) - np.pi) > np.radians(45)), default=0.0)
    return {"fitness": round(best[0], 3), "rmse_m": round(best[1], 4), "turn_deg": round(float(np.degrees(best[2])), 1),
            "turn_determined": bool(best[0] - rival > 0.08)}


def match_moves(removed: list[Blob], added: list[Blob], cfg: Config) -> list[tuple]:
    """Pair removals with additions whose SHAPES match after a turn and shift (colour breaks ties)."""
    if not removed or not added:
        return []
    cost = np.ones((len(removed), len(added)))
    fits = {}
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            sr, sa = np.sort(r.size), np.sort(a.size)
            if np.mean(np.abs(sr - sa) / np.maximum(sr, sa)) > 0.5:
                continue
            dist = float(np.linalg.norm(r.centre[:2] - a.centre[:2]))
            f = fit_move(r, a, cfg.voxel)
            turned = f["turn_determined"] and abs(f["turn_deg"]) > 20
            if dist < cfg.min_move and not turned:
                continue
            colour = 1 - float(np.minimum(_hist(r.colors), _hist(a.colors)).sum())
            fits[(i, j)] = f
            cost[i, j] = (1 - f["fitness"]) + 0.1 * colour
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j), float(cost[i, j]), fits[(i, j)]) for i, j in zip(rows, cols)
            if (i, j) in fits and fits[(i, j)]["fitness"] >= cfg.min_fit]


def naive_diff(A: Session, B: Session, cfg: Config, thresh: float = 0.05) -> dict:
    """The obvious baseline: a point is changed if the other cloud has nothing within 5 cm. No visibility."""
    from scipy.spatial import cKDTree
    out = {}
    for kind, X, Y in (("removed", A, B), ("added", B, A)):
        cx = voxelize(X.points(conf_pct=cfg.conf_pct, stride=2)[0], None, cfg.voxel)["centres"]
        cy = voxelize(Y.points(conf_pct=cfg.conf_pct, stride=2)[0], None, cfg.voxel)["centres"]
        d, _ = cKDTree(cy).query(cx, k=1)
        sel = np.nonzero((d > thresh) & (cx[:, 2] > cfg.floor_cut))[0]
        found = []
        for comp in _components(cx[sel], cfg.voxel):
            ids = sel[comp]
            ext = cx[ids].max(0) - cx[ids].min(0)
            if len(ids) >= cfg.min_blob_voxels and cfg.min_blob_extent <= ext.max() <= cfg.max_blob_extent:
                found.append({"centre_m": np.round(cx[ids].mean(0), 2).tolist(), "size_m": np.round(ext, 2).tolist()})
        out[kind] = found
    return out


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
