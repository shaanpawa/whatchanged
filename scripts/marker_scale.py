#!/usr/bin/env python3
"""Metres per COLMAP unit from the printed floor marker (ArUco DICT_4X4_50, id 0).

  scripts/marker_scale.py out/pair --side-mm 150                 a `whatchanged room` folder
  scripts/marker_scale.py out/real --side-mm 150                 or a `run --poses colmap` folder
  scripts/marker_scale.py out/pair --side-mm 150 --json out/pair/marker_scale.json
  scripts/marker_scale.py out/pair --side-mm 150 --check desk=123.4cm
  scripts/marker_scale.py --selftest                             known square, synthetic cameras

The marker's 4 corners are found in every registered, undistorted frame (so COLMAP's pinhole
cameras hold exactly), each corner is triangulated from all its detections (DLT, then views that
miss by more than 3 px are dropped and it is redone), and each of the 4 sides gives a scale. The
median is printed with the spread across sides; the last line is METRES_PER_UNIT=... for scripts.
Exit 0 = scale found and consistent; 3 = marker not seen in enough frames; 4 = sides disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from whatchanged import room3d  # noqa: E402

DICT = cv2.aruco.DICT_4X4_50


def find_dense(path: Path) -> Path:
    """The undistorted COLMAP folder (images/ + sparse/0) behind a room, run or work folder."""
    path = Path(path)
    diag = path / "cache" / "diag.json"
    cands = [path / "work" / "dense", path / "dense", path]
    if diag.exists() and json.loads(diag.read_text()).get("colmap"):
        cands.insert(0, Path(json.loads(diag.read_text())["colmap"]["work"]) / "dense")
    for c in cands:
        if (c / "sparse" / "0" / "images.bin").exists() and (c / "images").exists():
            return c
    sys.exit(f"no undistorted COLMAP model under {path} (looked for work/dense, dense, cache/diag.json)")


def intrinsics(cam: dict) -> np.ndarray:
    p = cam["params"]
    if cam["model"] == 1:            # PINHOLE: fx fy cx cy
        fx, fy, cx, cy = p
    elif cam["model"] == 0:          # SIMPLE_PINHOLE: f cx cy
        fx, cx, cy = p
        fy = fx
    else:
        sys.exit(f"camera model {cam['model']} is not undistorted; use the dense/ folder")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)


def detect(det: cv2.aruco.ArucoDetector, img: np.ndarray, marker_id: int = 0) -> np.ndarray | None:
    """The marker's 4 corners (top-left, top-right, bottom-right, bottom-left of the printed square), or None."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    hit = [c for c, i in zip(corners, ids.ravel()) if i == marker_id]
    return hit[0].reshape(4, 2).astype(np.float64) if hit else None


def triangulate(Ps: list[np.ndarray], uv: np.ndarray) -> np.ndarray:
    """Linear (DLT) triangulation in normalised camera coordinates; Ps are 3x4 [R|t], uv already K^-1-mapped."""
    A = []
    for P, (x, y) in zip(Ps, uv):
        A += [x * P[2] - P[0], y * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def reproject(K: np.ndarray, P: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    pc = P[:, :3] @ X + P[:, 3]
    return (K @ (pc / pc[2]))[:2], float(pc[2])


def corner_point(obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], max_px: float = 3.0) -> dict | None:
    """One corner from all its detections (K, [R|t], uv), dropping views that disagree."""
    keep = list(range(len(obs)))
    for _ in range(5):
        if len(keep) < 2:
            return None
        Ps = [obs[i][1] for i in keep]
        uvn = [(np.linalg.inv(obs[i][0]) @ np.r_[obs[i][2], 1])[:2] for i in keep]
        X = triangulate(Ps, np.array(uvn))
        err = []
        for K, P, uv in obs:
            p, z = reproject(K, P, X)
            err.append(np.inf if z <= 0 else float(np.linalg.norm(p - uv)))
        err = np.array(err)
        lim = max(max_px, 3 * float(np.median(err[keep])))
        new = [i for i in range(len(obs)) if err[i] < lim]
        if new == keep:
            break
        keep = new
    if len(keep) < 2:
        return None
    return {"X": X, "views": len(keep), "of": len(obs), "rms_px": float(np.sqrt(np.mean(err[keep] ** 2)))}


def measure(dense: Path, side_mm: float, marker_id: int = 0, log=print) -> dict:
    """Detect, triangulate and scale. Returns the numbers printed by main()."""
    sp = dense / "sparse" / "0"
    cams = room3d.read_cameras(sp / "cameras.bin")
    imgs = room3d.read_images(sp / "images.bin")
    det = detector()
    obs: list[list] = [[], [], [], []]
    seen = []
    for im in sorted(imgs.values(), key=lambda v: v["name"]):
        img = cv2.imread(str(dense / "images" / im["name"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        c = detect(det, img, marker_id)
        if c is None:
            continue
        K = intrinsics(cams[im["camera_id"]])
        P = np.linalg.inv(room3d.cam_to_world(im))[:3]
        for k in range(4):
            obs[k].append((K, P, c[k]))
        seen.append(im["name"])
    res = {"dense": str(dense), "frames": len(imgs), "detections": len(seen), "seen_in": seen[:5] + (["..."] if len(seen) > 5 else [])}
    log(f"  marker id {marker_id} seen in {len(seen)}/{len(imgs)} registered frames")
    if len(seen) < 3:
        return {**res, "ok": False, "why": "marker seen in fewer than 3 registered frames"}
    pts = [corner_point(o) for o in obs]
    if any(p is None for p in pts):
        return {**res, "ok": False, "why": "a corner could not be triangulated consistently"}
    X = np.array([p["X"] for p in pts])
    sides = np.array([np.linalg.norm(X[(k + 1) % 4] - X[k]) for k in range(4)])
    diags = np.array([np.linalg.norm(X[2] - X[0]), np.linalg.norm(X[3] - X[1])])
    per_side = side_mm / 1000 / sides
    mpu = float(np.median(per_side))
    spread = float((per_side.max() - per_side.min()) / mpu)
    angles = [float(np.degrees(np.arccos(np.clip(np.dot(X[(k + 1) % 4] - X[k], X[k - 1] - X[k]) /
                                                  (sides[k] * sides[k - 1]), -1, 1)))) for k in range(4)]
    n = np.cross(X[1] - X[0], X[3] - X[0])
    flat = float(abs(np.dot(X[2] - X[0], n / np.linalg.norm(n))) * mpu * 1000)
    return {**res, "ok": True, "metres_per_unit": mpu, "spread": spread, "side_units": sides.round(6).tolist(),
            "scale_per_side": np.round(per_side, 6).tolist(),
            "diagonal_check": float(np.mean(diags) * mpu / (np.sqrt(2) * side_mm / 1000)),
            "corner_angles_deg": np.round(angles, 1).tolist(), "out_of_plane_mm": round(flat, 2),
            "corners": [{"views": p["views"], "of": p["of"], "rms_px": round(p["rms_px"], 2)} for p in pts],
            "corners_xyz": X.round(6).tolist(), "side_mm": side_mm}


def other_scales(path: Path) -> dict:
    """Scales the pipeline already has for this model, to compare against."""
    out = {}
    diag = path / "cache" / "diag.json"
    if diag.exists():
        c = json.loads(diag.read_text()).get("colmap") or {}
        if "metric_scale_m_per_unit" in c:
            out["MapAnything depth (run)"] = c["metric_scale_m_per_unit"]
    scene = path / "scene.json"
    if scene.exists():
        fr = json.loads(scene.read_text()).get("meta", {}).get("frame", {})
        if "scale" in fr:
            out[f"viewer ({fr.get('scale_from', '?')})"] = fr["scale"]
    return out


def parse_length(s: str) -> float:
    m = re.fullmatch(r"([\d.]+)\s*(mm|cm|m)", s.strip())
    if not m:
        sys.exit(f"cannot read the length {s!r}; write e.g. 123.4cm")
    return float(m[1]) * {"mm": 0.001, "cm": 0.01, "m": 1.0}[m[2]]


def check_line(spec: str, mpu: float) -> str:
    """NAME=LEN, optionally @x,y,z@x,y,z (two points in COLMAP units) to actually compare."""
    name, _, rest = spec.partition("=")
    length, *ends = rest.split("@")
    tape = parse_length(length)
    if len(ends) == 2:
        p, q = (np.array([float(v) for v in e.split(",")]) for e in ends)
        got = float(np.linalg.norm(q - p)) * mpu
        return f"  check {name}: tape {tape * 100:.1f} cm, model {got * 100:.1f} cm ({(got - tape) / tape:+.1%})"
    return (f"  check {name}: tape {tape * 100:.1f} cm = {tape / mpu:.4f} COLMAP units at this scale "
            f"(no 3D end points given, so not compared yet)")


def report(r: dict, path: Path, checks: list[str]) -> int:
    if not r["ok"]:
        print(f"No scale: {r['why']}.")
        return 3
    print("  corners: " + ", ".join(f"{c['views']}/{c['of']} views, {c['rms_px']} px" for c in r["corners"]))
    print(f"  sides: {r['side_units']} COLMAP units -> {r['scale_per_side']} m/unit")
    print(f"  square check: corner angles {r['corner_angles_deg']} deg, out of plane {r['out_of_plane_mm']} mm, "
          f"diagonals {r['diagonal_check']:.3f}x the expected length")
    print(f"Scale: {r['metres_per_unit']:.6f} m per COLMAP unit (spread across the 4 sides {r['spread']:.1%}, "
          f"worst corner reprojection {max(c['rms_px'] for c in r['corners'])} px)")
    for name, s in other_scales(path).items():
        print(f"  vs {name}: {s:.6f} ({(s - r['metres_per_unit']) / r['metres_per_unit']:+.1%})")
    for c in checks:
        print(check_line(c, r["metres_per_unit"]))
    if r["spread"] > 0.05:
        print("warning: the sides disagree by more than 5%: is the marker flat, and is --side-mm the black square's side?")
    print(f"METRES_PER_UNIT={r['metres_per_unit']:.6f}")
    return 0 if r["spread"] <= 0.10 else 4


# ---------------------------------------------------------------- synthetic test

def _quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return float(w), float(x), float(y), float(z)


def _write_cameras(path: Path, cams: dict[int, dict]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cid, c in cams.items():
            f.write(struct.pack("<iiQQ", cid, c["model"], c["width"], c["height"]))
            f.write(struct.pack(f"<{len(c['params'])}d", *c["params"]))


def selftest(tmp: Path, mpu_true: float = 0.37, side_mm: float = 150, seed: int = 1) -> int:
    """Project a known square through known cameras, write a fake dense model, recover the scale."""
    rng = np.random.default_rng(seed)
    dense = tmp / "dense"
    (dense / "images" / "A").mkdir(parents=True)
    (dense / "sparse" / "0").mkdir(parents=True)
    W, H, f = 1600, 900, 1100.0
    _write_cameras(dense / "sparse" / "0" / "cameras.bin", {1: {"model": 1, "width": W, "height": H, "params": (f, f, W / 2, H / 2)}})
    s = side_mm / 1000 / mpu_true                               # the square's side in COLMAP units
    # A tilted square somewhere in an arbitrary COLMAP world.
    R0 = cv2.Rodrigues(np.array([0.4, -0.7, 0.2]))[0]
    c0 = np.array([0.3, -0.2, 1.1])
    square = lambda a: c0 + (np.array([[-a, -a, 0], [a, -a, 0], [a, a, 0], [-a, a, 0]]) / 2) @ R0.T  # noqa: E731
    X_true = square(s)
    quiet = square(s * 1.5)
    n = R0[:, 2]
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), 0, 600)
    pad = np.full((900, 900), 255, np.uint8)
    imgs = {}
    for i in range(14):
        # Camera 1.5-6 units away, 15-60 deg off the marker's normal, aimed near its centre.
        th, ph, r = np.radians(rng.uniform(15, 60)), rng.uniform(0, 2 * np.pi), rng.uniform(1.5, 6.0)
        e1, e2 = R0[:, 0], R0[:, 1]
        C = c0 + r * (-np.cos(th) * n + np.sin(th) * (np.cos(ph) * e1 + np.sin(ph) * e2))
        z = c0 + rng.normal(0, 0.1 * r, 3) - C
        z /= np.linalg.norm(z)
        x = np.cross(z, e1 + 0.3 * e2)
        x /= np.linalg.norm(x)
        R = np.stack([x, np.cross(z, x), z])                    # world -> camera rows
        t = -R @ C
        img = cv2.resize(rng.integers(60, 200, (H // 8, W // 8), dtype=np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (0, 0), 3)
        for pts3, im_ in ((quiet, pad), (X_true, tag)):         # white quiet zone, then the marker on it
            pc = pts3 @ R.T + t
            dst = ((pc[:, :2] / pc[:, 2:]) * f + [W / 2, H / 2]).astype(np.float32)
            m = im_.shape[0] - 1
            Hm = cv2.getPerspectiveTransform(np.array([[0, 0], [m, 0], [m, m], [0, m]], np.float32), dst)
            warped = cv2.warpPerspective(im_, Hm, (W, H), flags=cv2.INTER_LINEAR)
            inside = cv2.warpPerspective(np.full(im_.shape, 255, np.uint8), Hm, (W, H)) > 127
            img[inside] = warped[inside]
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        name = f"A/A_{i:04d}.jpg"
        cv2.imwrite(str(dense / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i == 13:                                             # one camera stored turned 1.7 deg: must be dropped
            R = cv2.Rodrigues(np.array([0.0, 0.03, 0.0]))[0] @ R
            t = -R @ C
        imgs[i + 1] = {"name": name, "camera_id": 1, "q": _quat(R), "t": tuple(t)}
        assert np.allclose(room3d.cam_to_world(imgs[i + 1])[:3, :3], R.T, atol=1e-9)
    room3d.write_images(dense / "sparse" / "0" / "images.bin", imgs)
    r = measure(dense, side_mm)
    code = report(r, tmp, [])
    err = abs(r.get("metres_per_unit", 0) - mpu_true) / mpu_true
    bad = [c["views"] for c in r.get("corners", [])]
    print(f"selftest: true {mpu_true}, got {r.get('metres_per_unit', float('nan')):.6f} ({err:.2%} off); "
          f"views kept per corner {bad} of 14 (the camera with the bad pose should be dropped)")
    ok = code == 0 and err < 0.01 and all(v == 13 for v in bad) and max(c["rms_px"] for c in r["corners"]) < 1.0
    print("selftest PASS" if ok else "selftest FAIL")
    return 0 if ok else 1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", nargs="?", help="a `whatchanged room` or `run --poses colmap` folder (or its work/dense)")
    p.add_argument("--side-mm", dest="side_mm", type=float, default=150, help="measured side of the printed black square")
    p.add_argument("--id", type=int, default=0, help="marker id (DICT_4X4_50)")
    p.add_argument("--check", action="append", default=[], help='tape distance, e.g. "desk=123.4cm" (add @x,y,z@x,y,z to compare)')
    p.add_argument("--json", help="also write the numbers here")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.selftest:
        with tempfile.TemporaryDirectory() as tmp:
            sys.exit(selftest(Path(tmp)))
    if not a.folder:
        sys.exit(__doc__)
    dense = find_dense(Path(a.folder))
    print(f"Marker scale from {dense}")
    r = measure(dense, a.side_mm, a.id)
    code = report(r, Path(a.folder), a.check)
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    sys.exit(code)


if __name__ == "__main__":
    main()
