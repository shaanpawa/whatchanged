"""The brief's traps on a synthetic room, rendered by ray casting. No model, runs in seconds.

A 4 x 3 m room with a floor and walls. Session A circles the room at 1.5 m; session B takes a
different route at 1.1 m. Then we check the four things the brief cares about:
  - a different viewpoint alone reports nothing,
  - a removed box is "removed",
  - a box hidden behind a new obstacle is not called removed,
  - a moved box is "moved".
"""

import numpy as np
import pytest

from whatchanged.changes import Config, detect
from whatchanged.recon import Session

W, H, F = 160, 120, 110.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
ROOM = [  # (min xyz, max xyz, colour)
    ((-0.1, -0.1, -0.1), (4.1, 3.1, 0.0), (180, 170, 160)),   # floor
    ((-0.1, -0.1, 0.0), (0.0, 3.1, 2.5), (200, 200, 210)),    # walls
    ((4.0, -0.1, 0.0), (4.1, 3.1, 2.5), (205, 200, 200)),
    ((-0.1, -0.1, 0.0), (4.1, 0.0, 2.5), (210, 205, 195)),
    ((-0.1, 3.0, 0.0), (4.1, 3.1, 2.5), (195, 205, 205)),
    ((3.0, 2.0, 0.0), (3.9, 2.9, 0.5), (90, 60, 40)),          # bed (static)
]
BOX = ((1.0, 1.0, 0.0), (1.4, 1.3, 0.4), (30, 90, 200))
SUITCASE = ((1.7, 1.0, 0.0), (2.2, 1.4, 1.0), (40, 160, 90))  # tall, between B's cameras and BOX


def look_at(eye, target):
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, eye
    return P


def render(boxes, pose):
    v, u = np.mgrid[0:H, 0:W]
    d_cam = np.stack([(u + 0.5 - K[0, 2]) / F, (v + 0.5 - K[1, 2]) / F, np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d_cam @ pose[:3, :3].T
    o = pose[:3, 3]
    best = np.full(len(d), np.inf)
    col = np.zeros((len(d), 3), np.uint8)
    for lo, hi, c in boxes:
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (np.array(lo) - o) / d
            t2 = (np.array(hi) - o) / d
        tn = np.nanmax(np.minimum(t1, t2), 1)
        tf = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tf >= tn) & (tn > 0.05) & (tn < best)
        best[hit] = tn[hit]
        col[hit] = c
    depth = (best * d_cam[:, 2]).reshape(H, W)
    depth[~np.isfinite(depth)] = 0
    return depth, col.reshape(H, W, 3)


def session(name, boxes, eyes, target=(2.0, 1.5, 0.3)):
    poses = np.stack([look_at(np.array(e, float), target) for e in eyes])
    rend = [render(boxes, P) for P in poses]
    depth = np.stack([r[0] for r in rend])
    return Session(name, np.stack([r[1] for r in rend]), depth, np.ones_like(depth), depth > 0,
                   np.repeat(K[None], len(eyes), 0), poses, np.arange(len(eyes), dtype=float), [""] * len(eyes))


def ring(cx, cy, r, z, n, start=0.0, span=2 * np.pi):
    a = start + np.linspace(0, span, n, endpoint=False)
    return [(cx + r * np.cos(t), cy + r * 0.8 * np.sin(t), z) for t in a]


A_EYES = ring(2.0, 1.5, 1.6, 1.5, 16)
B_EYES = ring(2.0, 1.5, 1.2, 1.1, 14, start=0.4)
B_FRONT = [(x, y, 1.1) for x, y in [(2.9, 1.0), (3.0, 1.3), (2.9, 1.6), (2.8, 0.7), (3.1, 0.9), (3.0, 1.9)]]
CFG = Config(voxel=0.04, min_blob_voxels=10)


def run(boxes_a, boxes_b, eyes_b=B_EYES):
    A = session("A", boxes_a, A_EYES)
    B = session("B", boxes_b, eyes_b)
    return detect(A, B, CFG, log=lambda *_: None)


def test_viewpoint_change_alone_reports_nothing():
    r = run(ROOM + [BOX], ROOM + [BOX])
    assert r["removed"] == [] and r["added"] == []


def test_removed_box_is_removed():
    r = run(ROOM + [BOX], ROOM)
    assert len(r["removed"]) == 1 and r["added"] == []
    c = r["removed"][0].centre
    assert abs(c[0] - 1.2) < 0.1 and abs(c[1] - 1.15) < 0.1


def test_added_box_is_added():
    r = run(ROOM, ROOM + [BOX])
    assert len(r["added"]) == 1 and r["removed"] == []


def test_hidden_box_is_not_called_removed():
    # B only films from behind the new suitcase, so it never sees where BOX is.
    r = run(ROOM + [BOX], ROOM + [BOX, SUITCASE], eyes_b=B_FRONT)
    near_box = [b for b in r["removed"] if np.linalg.norm(b.centre[:2] - [1.2, 1.15]) < 0.3]
    assert near_box == []


def test_moved_box_is_moved():
    moved = ((2.0, 2.2, 0.0), (2.4, 2.5, 0.4), BOX[2])
    r = run(ROOM + [BOX], ROOM + [moved])
    assert len(r["moves"]) == 1
    i, j = r["moves"][0][:2]
    dist = np.linalg.norm(r["added"][j].centre[:2] - r["removed"][i].centre[:2])
    assert dist == pytest.approx(np.hypot(1.0, 1.2), abs=0.1)


def test_moved_and_turned_box_reports_the_turn():
    long_box = ((1.0, 1.0, 0.0), (1.6, 1.25, 0.45), (200, 60, 40))
    # same box moved ~0.9 m and turned 90 degrees (its long side now runs along y)
    turned = ((2.2, 1.6, 0.0), (2.45, 2.2, 0.45), (200, 60, 40))
    r = run(ROOM + [long_box], ROOM + [turned])
    assert len(r["moves"]) == 1
    fit = r["moves"][0][3]
    assert fit["fitness"] > 0.5
    assert abs(abs(fit["turn_deg"]) - 90) < 15 or abs(abs(fit["turn_deg"]) - 90) > 165  # 90 deg, either sense


def test_icp_recovers_a_small_misalignment():
    from whatchanged.align import icp_4dof
    A = session("A", ROOM + [BOX], A_EYES)
    pts = A.points(conf_pct=0, stride=2)[0]
    th, shift = np.radians(2.0), np.array([0.05, -0.03, 0.01])
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = pts @ R.T + shift
    T, stats = icp_4dof(moved, pts)
    back = moved @ T[:3, :3].T + T[:3, 3]
    assert np.abs(back - pts).max() < 0.005
