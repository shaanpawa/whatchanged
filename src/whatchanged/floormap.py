"""Robot floor maps in ROS / Nav2 map_server format.

Each session gives a top-down grid: occupied (something between 5 cm and the robot's
height), free (floor seen, nothing above it), unknown (never seen). The updated map
patches the "before" map only where the second video actually looked: cells B never saw
keep A's value and are flagged stale, instead of being overwritten as free or unknown.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

UNKNOWN, FREE, OCC = -1, 0, 100     # nav_msgs/OccupancyGrid convention


def grid_of(points: np.ndarray, origin: np.ndarray, shape: tuple[int, int], res: float,
            robot_height: float = 2.2, floor_band: float = 0.04, obstacle_min: float = 0.06) -> np.ndarray:
    ij = np.floor((points[:, :2] - origin) / res).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    ij, z = ij[ok], points[ok, 2]
    occ = np.zeros(shape, np.int32)
    flo = np.zeros(shape, np.int32)
    o = (z > obstacle_min) & (z < robot_height)
    f = np.abs(z) < floor_band
    np.add.at(occ, (ij[o, 1], ij[o, 0]), 1)
    np.add.at(flo, (ij[f, 1], ij[f, 0]), 1)
    g = np.full(shape, UNKNOWN, np.int8)
    g[flo >= 2] = FREE
    g[occ >= 3] = OCC
    # Drop single-cell occupied speckle (thin chair legs survive: they span several cells in height).
    lab, n = ndimage.label(g == OCC)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    for i, s in enumerate(sizes, start=1):
        if s < 2:
            g[lab == i] = FREE if flo[lab == i].any() else UNKNOWN
    return g


def inflate(g: np.ndarray, res: float, robot_radius: float = 0.35, inflation_radius: float = 0.55,
            k: float = 10.0) -> np.ndarray:
    """Nav2 inflation-layer costs: 254 lethal, 253 inscribed, then 252*exp(-k*(d-r))."""
    d = ndimage.distance_transform_edt(g != OCC) * res
    cost = np.where(d <= 0, 254, np.where(d <= robot_radius, 253,
                    np.where(d <= inflation_radius, 252 * np.exp(-k * (d - robot_radius)), 0)))
    cost = cost.astype(np.uint8)
    cost[g == UNKNOWN] = 255
    return cost


def write_map(g: np.ndarray, path_stem: Path, origin: np.ndarray, res: float) -> None:
    """map_server trinary PGM + YAML. Row 0 of the PGM is the top (max y)."""
    img = np.full(g.shape, 205, np.uint8)
    img[g == FREE] = 254
    img[g == OCC] = 0
    Image.fromarray(img[::-1]).save(path_stem.with_suffix(".pgm"))
    path_stem.with_suffix(".yaml").write_text(
        f"image: {path_stem.with_suffix('.pgm').name}\nmode: trinary\nresolution: {res}\n"
        f"origin: [{origin[0]:.3f}, {origin[1]:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")


def build(A, B, changes: list[dict], out_dir: Path, res: float = 0.05, robot_radius: float = 0.35) -> dict:
    pa, _ = A.points(conf_pct=30, stride=2)
    pb, _ = B.points(conf_pct=30, stride=2)
    allp = np.concatenate([pa, pb])
    lo = np.percentile(allp[:, :2], 1, axis=0) - 0.25
    hi = np.percentile(allp[:, :2], 99, axis=0) + 0.25
    shape = (int(np.ceil((hi[1] - lo[1]) / res)), int(np.ceil((hi[0] - lo[0]) / res)))
    ga, gb = grid_of(pa, lo, shape, res), grid_of(pb, lo, shape, res)

    seen_b = gb != UNKNOWN
    updated = np.where(seen_b, gb, ga)
    stale = ~seen_b & (ga != UNKNOWN)

    # Change footprints from the detector (not from raw grid noise).
    def footprint(c, key="after"):
        f = c.get(key, c) if c["type"] == "moved" else c
        cx, cy = f["centre_m"][:2]
        sx, sy = f["size_m"][:2]
        yaw = np.radians(f["yaw_deg"])
        ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
        X, Y = lo[0] + (xs + 0.5) * res - cx, lo[1] + (ys + 0.5) * res - cy
        u = X * np.cos(yaw) + Y * np.sin(yaw)
        v = -X * np.sin(yaw) + Y * np.cos(yaw)
        return (np.abs(u) <= sx / 2) & (np.abs(v) <= sy / 2)

    blocked = np.zeros(shape, bool)
    freed = np.zeros(shape, bool)
    for c in changes:
        if c["type"] in ("added", "moved") and c["bottom_m"] < 0.15:
            blocked |= footprint(c, "after")
        if c["type"] in ("removed", "moved"):
            f = footprint(c, "before")
            if (c.get("before", c))["bottom_m"] < 0.15:
                freed |= f
    freed &= ~blocked

    out_dir.mkdir(parents=True, exist_ok=True)
    write_map(ga, out_dir / "map_before", lo, res)
    write_map(gb, out_dir / "map_after", lo, res)
    write_map(updated, out_dir / "map_updated", lo, res)
    Image.fromarray(inflate(updated, res, robot_radius)[::-1]).save(out_dir / "costmap_updated.png")

    # A colour picture for people: grey free, dark occupied, green newly blocked, orange newly free, hatched stale.
    rgb = np.full(shape + (3,), 236, np.uint8)
    rgb[updated == FREE] = (250, 250, 250)
    rgb[updated == OCC] = (60, 66, 74)
    rgb[updated == UNKNOWN] = (214, 219, 224)
    hatch = ((np.add.outer(np.arange(shape[0]), np.arange(shape[1])) // 2) % 3 == 0)
    rgb[stale & hatch] = (230, 159, 0)
    rgb[freed] = (213, 94, 0)
    rgb[blocked] = (0, 158, 115)
    img = Image.fromarray(rgb[::-1]).resize((shape[1] * 6, shape[0] * 6), Image.NEAREST)
    img.save(out_dir / "map_update.png")

    summary = {
        "resolution_m": res, "origin_m": [round(float(lo[0]), 3), round(float(lo[1]), 3)],
        "robot_radius_m": robot_radius, "obstacle_band_m": [0.06, 2.2],
        "newly_blocked_m2": round(float(blocked.sum() * res * res), 2),
        "newly_free_m2": round(float(freed.sum() * res * res), 2),
        "stale_m2": round(float(stale.sum() * res * res), 2),
        "known_before_m2": round(float((ga != UNKNOWN).sum() * res * res), 2),
        "known_after_m2": round(float((gb != UNKNOWN).sum() * res * res), 2),
        "files": ["map_before.yaml", "map_after.yaml", "map_updated.yaml", "costmap_updated.png", "map_update.png"],
    }
    (out_dir / "map_update.json").write_text(json.dumps(summary, indent=1))
    np.savez_compressed(out_dir / "grids.npz", before=ga, after=gb, updated=updated, origin=lo, res=res,
                        free_both=(ga == FREE) & (gb == FREE))
    return summary
