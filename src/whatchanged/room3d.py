"""A photoreal 3D room from phone video: sharp frames -> COLMAP poses -> Gaussian splat -> web viewer.

  whatchanged room ROOM.MOV --out out/room3d              one video, one splat
  whatchanged room BEFORE.MOV AFTER.MOV --out out/pair    shared poses, one splat per video

All videos are posed together in ONE COLMAP model, so every splat lives in the same frame and a
before/after pair lines up exactly. Each video still gets its own splat: training one splat on
both would turn every moved object into a ghost. The splats are for looking at the room; change
detection runs on the MapAnything geometry (recon.py, changes.py), not on these.
"""

from __future__ import annotations

import gzip
import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np


VIEWER = Path(__file__).with_name("viewer") / "room.html"


# ---------------------------------------------------------------- frames

def extract_frames(video: str | Path, out_dir: Path, prefix: str, fps: float = 4.0,
                   max_side: int = 1600, log=print) -> list[dict]:
    """Keep the sharpest frame in every 1/fps-second window, then drop the blurriest leftovers."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = max(1, round(vfps / fps))
    best: dict[int, tuple[float, np.ndarray, float]] = {}
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        ok, img = cap.retrieve()
        if ok:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            h, w = img.shape[:2]
            s = max_side / max(h, w)
            if s < 1:
                img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            b = i // window
            if b not in best or sharp > best[b][0]:
                best[b] = (sharp, img, t)
        i += 1
    cap.release()
    med = float(np.median([v[0] for v in best.values()]))
    frames = []
    for b in sorted(best):
        sharp, img, t = best[b]
        if sharp < 0.35 * med:  # a whole window of motion blur: leave it out rather than teach the splat blur
            continue
        name = f"{prefix}_{len(frames):04d}.jpg"
        cv2.imwrite(str(out_dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append({"name": f"{out_dir.name}/{name}", "video": str(video), "t": round(t, 3), "sharpness": round(sharp, 1)})
    log(f"  {prefix}: {len(frames)} frames from {i} ({len(best) - len(frames)} blurry windows dropped)")
    return frames


# ---------------------------------------------------------------- COLMAP

def _colmap(*args, log=print) -> None:
    cmd = ["colmap", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
        raise RuntimeError(f"{' '.join(cmd[:2])} failed:\n{tail}")


def match_pairs(groups: list[list[str]], window: int = 10, stride: int = 4) -> list[tuple[str, str]]:
    """Neighbours in time within each video, plus a strided all-pairs net within and across videos.

    The net closes loops (the walk ends back at the door) and ties the videos together, at a
    fraction of the cost of exhaustive matching.
    """
    pairs = set()
    for g in groups:
        for i in range(len(g)):
            for k in range(1, window + 1):
                if i + k < len(g):
                    pairs.add((g[i], g[i + k]))
    net = [n for g in groups for n in g[::stride]]
    for i in range(len(net)):
        for j in range(i + 1, len(net)):
            pairs.add((net[i], net[j]))
    return sorted(pairs)


def run_colmap(work: Path, groups: list[list[str]], log=print) -> Path:
    """Features, matching, global SfM, undistortion. Returns the undistorted dataset folder."""
    db, images, sparse = work / "colmap.db", work / "images", work / "sparse"
    db.unlink(missing_ok=True)
    shutil.rmtree(sparse, ignore_errors=True)
    sparse.mkdir(parents=True)
    t = time.time()
    _colmap("feature_extractor", "--database_path", db, "--image_path", images,
            "--ImageReader.camera_model", "OPENCV", "--ImageReader.single_camera_per_folder", 1,
            "--FeatureExtraction.use_gpu", 0, "--SiftExtraction.max_num_features", 8192, log=log)
    log(f"  features: {time.time() - t:.0f}s")
    pairs = match_pairs(groups)
    (work / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in pairs))
    t = time.time()
    _colmap("matches_importer", "--database_path", db, "--match_list_path", work / "pairs.txt",
            "--match_type", "pairs", "--FeatureMatching.use_gpu", 0, log=log)
    log(f"  matches: {len(pairs)} pairs in {time.time() - t:.0f}s")
    t = time.time()
    try:
        _colmap("global_mapper", "--database_path", db, "--image_path", images, "--output_path", sparse,
                "--GlobalMapper.gp_use_gpu", 0, log=log)
    except RuntimeError as e:
        log(f"  global mapper failed, trying incremental: {str(e).splitlines()[0]}")
    n_all = sum(len(g) for g in groups)
    model, n_reg = _largest_model(sparse)
    if n_reg < 0.8 * n_all:
        log(f"  global mapper posed {n_reg}/{n_all}; trying the incremental mapper")
        shutil.rmtree(sparse)
        sparse.mkdir()
        _colmap("mapper", "--database_path", db, "--image_path", images, "--output_path", sparse, log=log)
        model, n_reg = _largest_model(sparse)
    log(f"  poses: {n_reg}/{n_all} frames registered in {time.time() - t:.0f}s")
    dense = work / "dense"
    shutil.rmtree(dense, ignore_errors=True)
    _colmap("image_undistorter", "--image_path", images, "--input_path", model, "--output_path", dense,
            "--output_type", "COLMAP", log=log)
    (dense / "sparse" / "0").mkdir(exist_ok=True)
    for f in (dense / "sparse").glob("*.bin"):
        f.rename(dense / "sparse" / "0" / f.name)
    return dense


def _largest_model(sparse: Path) -> tuple[Path, int]:
    best, n_best = None, -1
    for m in sorted(p for p in sparse.iterdir() if (p / "images.bin").exists()):
        n = len(read_images(m / "images.bin"))
        if n > n_best:
            best, n_best = m, n
    if best is None:
        raise RuntimeError("COLMAP could not pose the video. Film slower, with more texture in view.")
    return best, n_best


# ---------------------------------------------------------------- COLMAP binary files (just what we need)

def read_cameras(path: Path) -> dict[int, dict]:
    nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8}
    cams = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
            params = struct.unpack(f"<{nparams[model]}d", f.read(8 * nparams[model]))
            cams[cid] = {"model": model, "width": w, "height": h, "params": params}
    return cams


def read_images(path: Path) -> dict[int, dict]:
    """image_id -> name, camera id, world-to-camera quaternion (w, x, y, z) and translation."""
    imgs = {}
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            iid, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<i7di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            n2d = struct.unpack("<Q", f.read(8))[0]
            f.seek(24 * n2d, 1)
            imgs[iid] = {"name": name.decode(), "camera_id": cid, "q": (qw, qx, qy, qz), "t": (tx, ty, tz)}
    return imgs


def write_images(path: Path, imgs: dict[int, dict]) -> None:
    """images.bin without 2D observations (the splat trainer only needs poses)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, im in imgs.items():
            f.write(struct.pack("<i7di", iid, *im["q"], *im["t"], im["camera_id"]))
            f.write(im["name"].encode() + b"\0" + struct.pack("<Q", 0))


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """xyz (N, 3), rgb (N, 3) uint8, and the image ids that saw each point."""
    xyz, rgb, seen = [], [], []
    with open(path, "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            rec = struct.unpack("<Q3d3BdQ", f.read(51))
            xyz.append(rec[1:4])
            rgb.append(rec[4:7])
            track = np.frombuffer(f.read(8 * rec[8]), dtype="<u4").reshape(-1, 2)
            seen.append(track[:, 0])
    return np.array(xyz), np.array(rgb, dtype=np.uint8), seen


def write_points_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz, rgb
    path.write_bytes(head.encode() + rec.tobytes())


def cam_to_world(im: dict) -> np.ndarray:
    """4x4 camera-to-world in COLMAP's convention (x right, y down, z forward)."""
    qw, qx, qy, qz = im["q"]
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3] = -R.T @ np.asarray(im["t"])
    return T


def session_dataset(dense: Path, prefix: str, out: Path) -> int:
    """A dataset folder holding only one video's poses and the points that video saw."""
    imgs = read_images(dense / "sparse" / "0" / "images.bin")
    mine = {k: v for k, v in imgs.items() if v["name"].startswith(prefix + "/")}
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    shutil.copy(dense / "sparse" / "0" / "cameras.bin", out / "sparse" / "0" / "cameras.bin")
    write_images(out / "sparse" / "0" / "images.bin", mine)
    xyz, rgb, seen = read_points(dense / "sparse" / "0" / "points3D.bin")
    ids = np.array(list(mine), dtype=np.uint32)
    keep = np.array([np.isin(s, ids).sum() >= 2 for s in seen])
    write_points_ply(out / "sparse" / "0" / "points3D.ply", xyz[keep], rgb[keep])
    link = out / "images"
    if not link.exists():
        link.symlink_to((dense / "images").resolve(), target_is_directory=True)
    return len(mine)


# ---------------------------------------------------------------- splat training

def train_splat(dataset: Path, out_ply: Path, iterations: int = 15000, log=print) -> dict:
    import msplat

    t = time.time()
    ds = msplat.Dataset(str(dataset))
    cfg = msplat.TrainingConfig(iterations=iterations, num_downscales=2, resolution_schedule=max(500, iterations // 6),
                                keep_crs=True, bg_color=[0.0, 0.0, 0.0])
    tr = msplat.GaussianTrainer(ds, cfg)
    every = max(100, iterations // 10)
    tr.train(lambda s: log(f"  step {s.iteration:>6}/{iterations}  {s.splat_count:>9,} splats  {s.ms_per_step:5.1f} ms/step"),
             callback_every=every)
    tr.export_ply(str(out_ply))
    stats = {"iterations": iterations, "splats": tr.splat_count, "frames": ds.num_train, "seconds": round(time.time() - t)}
    msplat.cleanup()
    return stats


# ---------------------------------------------------------------- tidy + compress

PLY_TYPES = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}


def read_splat_ply(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    fields, n = [], 0
    for line in raw[:end].decode().splitlines():
        p = line.split()
        if p[:2] == ["element", "vertex"]:
            n = int(p[2])
        elif p[0] == "property":
            fields.append((p[2], PLY_TYPES[p[1]]))
    return np.frombuffer(raw, dtype=np.dtype(fields), count=n, offset=end)


def tidy(g: np.ndarray, cams: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop near-invisible splats, giant blobs and floaters outside the room."""
    n0 = len(g)
    xyz = np.stack([g["x"], g["y"], g["z"]], 1)
    alpha = 1 / (1 + np.exp(-g["opacity"]))
    size = np.exp(np.max(np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1), 1))
    lo, hi = np.percentile(pts, 1, 0), np.percentile(pts, 99, 0)
    lo, hi = np.minimum(lo, cams.min(0)), np.maximum(hi, cams.max(0))
    pad = 0.15 * (hi - lo)
    inside = np.all((xyz > lo - pad) & (xyz < hi + pad), 1)
    extent = float(np.linalg.norm(hi - lo))
    keep = inside & (alpha > 0.02) & (size < 0.1 * extent)
    return g[keep], {"before": n0, "after": int(keep.sum()), "dropped_outside": int((~inside).sum()),
                     "dropped_faint": int((alpha <= 0.02).sum()), "dropped_huge": int((size >= 0.1 * extent).sum())}


def write_spz(path: Path, g: np.ndarray, sh_degree: int = 3) -> int:
    """Niantic SPZ v2 (gzip, about 10x smaller than PLY). Coordinates are stored as-is."""
    n = len(g)
    names = g.dtype.names
    have = sum(1 for k in names if k.startswith("f_rest_")) // 3
    dims = {0: 0, 1: 3, 2: 8, 3: 15}
    sh_degree = max(d for d, k in dims.items() if k <= have and d <= sh_degree)
    xyz = np.stack([g["x"], g["y"], g["z"]], 1).astype(np.float64)
    frac = 12
    while np.abs(xyz).max() * (1 << frac) >= (1 << 23) and frac > 4:
        frac -= 1
    fixed = np.round(xyz * (1 << frac)).astype(np.int32).reshape(-1)
    pos = np.stack([fixed & 0xFF, (fixed >> 8) & 0xFF, (fixed >> 16) & 0xFF], 1).astype(np.uint8)

    def u8(x):
        return np.clip(np.round(x), 0, 255).astype(np.uint8)

    alpha = u8(255 / (1 + np.exp(-g["opacity"].astype(np.float64))))
    dc = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1)
    color = u8(dc * (0.15 * 255) + 127.5)
    scale = u8((np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1) + 10) * 16)
    q = np.stack([g["rot_1"], g["rot_2"], g["rot_3"], g["rot_0"]], 1).astype(np.float64)  # x, y, z, w
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q *= np.where(q[:, 3:] < 0, -1, 1)
    rot = u8(q[:, :3] * 127.5 + 127.5)
    parts = [pos.tobytes(), alpha.tobytes(), color.tobytes(), scale.tobytes(), rot.tobytes()]
    k = dims[sh_degree]
    if k:
        rest = np.stack([g[f"f_rest_{i}"] for i in range(3 * have)], 1).reshape(n, 3, have)[:, :, :k]
        sh = np.round(rest.transpose(0, 2, 1) * 128 + 128)  # coefficient-major, rgb interleaved
        bucket = np.ones((k, 1)) * 16
        bucket[:3] = 8  # degree 1 keeps 5 bits, the rest 4
        sh = np.floor((sh + bucket / 2) / bucket) * bucket
        parts.append(u8(sh).tobytes())
    head = struct.pack("<IIIBBBB", 0x5053474E, 2, n, sh_degree, frac, 0, 0)
    data = gzip.compress(head + b"".join(parts), compresslevel=9)
    path.write_bytes(data)
    return len(data)


# ---------------------------------------------------------------- viewer frame

def level_up(cams_c2w: np.ndarray) -> np.ndarray:
    """World up: the one direction every camera's x axis is perpendicular to (a handheld phone has ~no roll).

    The cameras' own up vectors are tilted by however much the phone looked down, so averaging them
    is off by that pitch; their x axes are not. Falls back to the average up if the camera never turned.
    """
    X = cams_c2w[:, :3, 0]
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    hint = -cams_c2w[:, :3, 1].mean(0)
    up = vt[-1] if sv[1] > 0.2 * sv[0] else hint
    return up * np.sign(up @ hint) / np.linalg.norm(up)


def find_floor(pts: np.ndarray, up: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, float]:
    """The lowest substantial horizontal layer below the cameras, as a plane n·x + d = 0.

    A bed or desk top often holds more points than the floor, so this takes the lowest histogram
    peak with at least a fifth of the busiest one's points, then fits a plane to that layer.
    """
    h = pts @ up
    hc = float(np.median(centres @ up))
    below = h[h < hc]
    lo = float(np.percentile(below, 0.5))
    w = (hc - lo) / 40
    hist, edges = np.histogram(below, bins=np.arange(lo, hc + w, w))
    sm = np.convolve(hist, np.ones(3) / 3, mode="same")
    pad = np.r_[-1, sm, -1]
    peaks = [i for i in range(len(sm)) if sm[i] >= pad[i] and sm[i] >= pad[i + 2] and sm[i] >= 0.2 * sm.max()]
    i = peaks[0] if peaks else int(np.argmax(sm))
    level = (edges[i] + edges[i + 1]) / 2
    band = pts[np.abs(h - level) < 1.5 * w]
    c = band.mean(0)
    _, vecs = np.linalg.eigh(np.cov((band - c).T))
    n = vecs[:, 0] * np.sign(vecs[:, 0] @ up)
    if n @ up < np.cos(np.radians(10)):  # a sloping layer is not the floor: keep the camera-derived up
        n = up
    return n, float(-n @ c)


def viewer_frame(cams_c2w: np.ndarray, pts: np.ndarray, camera_height: float = 1.4, scale: float | None = None) -> dict:
    """Similarity from COLMAP's world to a Y-up, floor-at-zero, metric frame centred on the walk.

    COLMAP has no scale: `scale` (metres per COLMAP unit, e.g. from metric depth) sets it when known,
    otherwise the median camera height is taken to be `camera_height` metres.
    """
    centres = cams_c2w[:, :3, 3]
    y, d = find_floor(pts, level_up(cams_c2w), centres)
    fwd = cams_c2w[0, :3, 2] - (cams_c2w[0, :3, 2] @ y) * y  # the first camera looks along -z
    z = -fwd / np.linalg.norm(fwd)
    x = np.cross(y, z)
    R = np.stack([x, y, z])  # rows: new axes in COLMAP coordinates
    h = float(np.median(centres @ y + d))
    s = scale or camera_height / h
    t = -s * (R @ centres.mean(0))
    t[1] = s * d  # floor (n·p + d = 0) -> y = 0
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return {"matrix": M, "scale": s, "scale_from": "metric depth" if scale else f"camera height {camera_height} m",
            "median_camera_height_m": round(h * s, 3)}


# ---------------------------------------------------------------- tie-in with a detection run

def _run_sessions(run: Path, names: list[str], n: str) -> list:
    """Detection sessions filmed with video n: all of them for a single (self-split) video, else the same name."""
    from .recon import Session
    want = ["A", "B"] if names == ["room"] else [n]
    return [Session.load(s, run / "cache" / f"{s}.npz") for s in want if (run / "cache" / f"{s}.npz").exists()]


def metric_scale(run: Path, names: list[str], frames: dict, imgs: dict[int, dict], xyz: np.ndarray,
                 seen: list[np.ndarray], log=print) -> float | None:
    """Metres per COLMAP unit from MapAnything's metric depth, independent of either pipeline's poses.

    Where a detection keyframe and a COLMAP frame are the same video frame, every COLMAP point that
    frame saw has a depth in both; the median ratio is the scale.
    """
    obs: dict[int, list[int]] = {}
    for k, ids in enumerate(seen):
        for i in ids:
            obs.setdefault(int(i), []).append(k)
    tname = {f["name"]: f["t"] for n in names for f in frames[n]}
    ratios, used = [], 0
    for n in names:
        mine = sorted((tname[im["name"]], iid) for iid, im in imgs.items() if im["name"] in tname and im["name"].startswith(n + "/"))
        ts = np.array([t for t, _ in mine])
        for S in _run_sessions(run, names, n):
            for i, t in enumerate(S.times):
                j = int(np.argmin(np.abs(ts - t)))
                if abs(ts[j] - t) > 0.05 or len(obs.get(mine[j][1], [])) < 20:
                    continue
                T = np.linalg.inv(cam_to_world(imgs[mine[j][1]]))
                pc = xyz[obs[mine[j][1]]] @ T[:3, :3].T + T[:3, 3]
                pc = pc[pc[:, 2] > 1e-6]
                K, (H, W) = S.K[i], S.depth[i].shape
                u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
                v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
                ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                dm = S.depth[i][v[ok], u[ok]]
                good = S.mask[i][v[ok], u[ok]] & (dm > 0)
                ratios.append(dm[good] / pc[ok, 2][good])
                used += 1
    if used < 3:
        log(f"  metric scale: only {used} shared frames with {run}; falling back to camera height")
        return None
    r = np.concatenate(ratios)
    s = float(np.median(r))
    iqr = float(np.subtract(*np.percentile(r, [75, 25])) / s)
    log(f"  metric scale: {s:.4f} m per COLMAP unit from {len(r):,} depths in {used} shared frames (IQR {iqr:.0%})")
    return s


def detection_alignment(run: Path, names: list[str], frames: dict, imgs: dict[int, dict], M: np.ndarray,
                        log=print) -> dict | None:
    """Yaw + shift taking the detection run's levelled frame into the viewer frame.

    Both frames are already level, floor at 0 and metric, so only a turn about the vertical and a
    horizontal shift remain. Fitted with RANSAC on camera centres at the same video times; the
    inlier share says how far the two pipelines' camera paths agree.
    """
    by_name = {im["name"]: M[:3, :3] @ cam_to_world(im)[:3, 3] + M[:3, 3] for im in imgs.values()}
    src, dst = [], []
    for n in names:
        mine = sorted((f["t"], by_name[f["name"]]) for f in frames[n] if f["name"] in by_name)
        ts, cs = np.array([t for t, _ in mine]), np.array([c for _, c in mine])
        for S in _run_sessions(run, names, n):
            for t, P in zip(S.times, S.poses):
                j = int(np.searchsorted(ts, t))
                if not (0 < j < len(ts)) or ts[j] - ts[j - 1] > 0.6:
                    continue
                w = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
                dst.append((1 - w) * cs[j - 1] + w * cs[j])
                x, y, z = P[:3, 3]
                src.append([x, z, -y])
    if len(src) < 4:
        log("  boxes: no shared camera times with the detection run; boxes left out")
        return None
    src, dst = np.array(src), np.array(dst)

    def fit(i):
        a, b = src[i][:, [0, 2]], dst[i][:, [0, 2]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((b - cb).T @ (a - ca))
        R2 = U @ np.diag([1, np.linalg.det(U @ Vt)]) @ Vt
        return R2, cb - R2 @ ca

    rng = np.random.default_rng(0)
    best = np.zeros(len(src), bool)
    for _ in range(500):
        R2, t2 = fit(rng.choice(len(src), 2, replace=False))
        inl = np.linalg.norm(src[:, [0, 2]] @ R2.T + t2 - dst[:, [0, 2]], axis=1) < 0.15
        if inl.sum() > best.sum():
            best = inl
    R2, t2 = fit(np.flatnonzero(best))
    res = np.linalg.norm(src[:, [0, 2]] @ R2.T + t2 - dst[:, [0, 2]], axis=1)
    T = np.eye(4)
    T[np.ix_([0, 2], [0, 2])] = R2
    T[[0, 2], 3] = t2
    T[1, 3] = float(np.median(dst[best, 1] - src[best, 1]))
    rms = float(np.sqrt(np.mean(res[best] ** 2)))
    log(f"  boxes: {best.sum()}/{len(src)} detection cameras within 15 cm of the splat's path ({rms * 100:.1f} cm RMS)")
    if best.sum() < 0.6 * len(src):
        log("  boxes left out: the two camera paths disagree too much for the boxes to land on the right things")
        return None
    return {"matrix": T, "yaw_deg": float(np.degrees(np.arctan2(T[0, 2], T[0, 0]))),
            "inliers": int(best.sum()), "pairs": len(src), "rms_m": round(rms, 4)}


def detection_boxes(run: Path, names: list[str], align: dict) -> dict:
    """Object boxes and changes from a detection run, moved into the viewer frame (y up, metres)."""
    T = align["matrix"]

    def box(f: dict, **extra) -> dict:
        x, y, z = f["centre_m"]
        c = T[:3, :3] @ np.array([x, z, -y]) + T[:3, 3]
        return {"centre": np.round(c, 3).tolist(), "size": [f["size_m"][0], f["size_m"][2], f["size_m"][1]],
                "yaw_deg": round(f.get("yaw_deg", 0.0) + align["yaw_deg"], 1), **extra}

    out = {"objects": {}, "changes": []}
    if (run / "objects.json").exists():
        sessions = json.loads((run / "objects.json").read_text())["sessions"]
        for n in names:
            out["objects"][n] = [box(o, label=o["label"]) for o in sessions.get("A" if n in ("room", "A") else n, [])]
    if (run / "changes.json").exists():
        for c in json.loads((run / "changes.json").read_text())["changes"]:
            item = {"id": c["id"], "type": c["type"], "label": c["label"], "sentence": c["sentence"], **box(c)}
            if c["type"] == "moved":
                item["before"], item["after"] = box(c["before"]), box(c["after"])
            out["changes"].append(item)
    return out


def camera_path(imgs: dict[int, dict], M: np.ndarray, times: dict[str, float]) -> list[dict]:
    """Every posed frame as a three.js camera (looks down -z, y up) in the viewer frame, with its video time."""
    s = np.cbrt(np.linalg.det(M[:3, :3]))
    R = M[:3, :3] / s
    flip = np.diag([1.0, -1.0, -1.0])
    out = []
    for im in sorted(imgs.values(), key=lambda v: v["name"]):
        T = cam_to_world(im)
        pos = M[:3, :3] @ T[:3, 3] + M[:3, 3]
        rot = R @ T[:3, :3] @ flip
        out.append({"name": im["name"], "t": times.get(im["name"]), "position": np.round(pos, 4).tolist(),
                    "rotation": np.round(rot, 5).tolist()})
    return out


def build_viewer(out: Path, splats: dict[str, str], frame: dict, cams: dict[str, list], meta: dict, boxes: dict) -> Path:
    scene = {"splats": splats, "matrix": np.round(frame["matrix"], 6).T.reshape(-1).tolist(),  # column-major for three.js
             "cameras": cams, "meta": meta, **boxes}
    (out / "scene.json").write_text(json.dumps(scene))
    html = VIEWER.read_text().replace("/*SCENE*/null", json.dumps(scene))
    (out / "index.html").write_text(html)
    return out / "index.html"


# ---------------------------------------------------------------- pipeline

def build(videos: list[str], out: Path, fps: float = 4.0, max_side: int = 1600, iterations: int = 30000,
          sh_degree: int = 3, camera_height: float = 1.4, detections: Path | None = None, force: bool = False,
          log=print) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    names = ["A", "B", "C", "D"][: len(videos)] if len(videos) > 1 else ["room"]
    frames_json = work / "frames.json"
    if force or not frames_json.exists():
        shutil.rmtree(work / "images", ignore_errors=True)
        log(f"Frames: sharpest per {1 / fps:.2f}s window, longest side {max_side}px")
        frames = {n: extract_frames(v, work / "images" / n, n, fps=fps, max_side=max_side, log=log) for n, v in zip(names, videos)}
        frames_json.write_text(json.dumps(frames, indent=1))
        (work / "dense").exists() and shutil.rmtree(work / "dense")
    frames = json.loads(frames_json.read_text())
    dense = work / "dense"
    if force or not (dense / "sparse" / "0" / "images.bin").exists():
        log("Poses: COLMAP (SIFT, sequential + loop pairs, global mapper)")
        dense = run_colmap(work, [[f["name"] for f in frames[n]] for n in names], log=log)
    imgs = read_images(dense / "sparse" / "0" / "images.bin")
    xyz, rgb, seen = read_points(dense / "sparse" / "0" / "points3D.bin")
    c2w = np.stack([cam_to_world(im) for im in imgs.values()])
    run = Path(detections) if detections else None
    scale = metric_scale(run, names, frames, imgs, xyz, seen, log=log) if run else None
    frame = viewer_frame(c2w, xyz, camera_height, scale)
    log(f"Viewer frame: scale from {frame['scale_from']}, median camera height {frame['median_camera_height_m']} m")
    align = detection_alignment(run, names, frames, imgs, frame["matrix"], log=log) if run else None

    cam0 = next(iter(read_cameras(dense / "sparse" / "0" / "cameras.bin").values()))
    fov_y = float(np.degrees(2 * np.arctan(cam0["height"] / 2 / cam0["params"][1])))
    times = {f["name"]: f["t"] for n in names for f in frames[n]}
    splats, cams = {}, {}
    meta = {"videos": dict(zip(names, videos)), "fov_y_deg": round(fov_y, 2)}
    for n in names:
        ds = work / f"ds_{n}"
        k = session_dataset(dense, n, ds)
        ply = work / f"{n}.ply"
        if force or not ply.exists():
            log(f"Splat {n}: training on {k} posed frames, {iterations} steps (Metal)")
            meta[f"train_{n}"] = train_splat(ds, ply, iterations, log=log)
            (work / f"{n}.train.json").write_text(json.dumps(meta[f"train_{n}"]))
        elif (work / f"{n}.train.json").exists():
            meta[f"train_{n}"] = json.loads((work / f"{n}.train.json").read_text())
        mine = {i: im for i, im in imgs.items() if im["name"].startswith(n + "/")}
        cam_xyz = np.stack([cam_to_world(im)[:3, 3] for im in mine.values()])
        g, stats = tidy(read_splat_ply(ply), cam_xyz, xyz)
        size = write_spz(out / f"{n}.spz", g, sh_degree)
        meta[f"tidy_{n}"] = stats
        log(f"  {n}.spz: {stats['after']:,} splats ({stats['before'] - stats['after']:,} floaters removed), {size / 1e6:.1f} MB")
        splats[n] = f"{n}.spz"
        cams[n] = camera_path(mine, frame["matrix"], times)
    meta["frame"] = {k: v for k, v in frame.items() if k != "matrix"}
    meta["registered"] = {n: len(cams[n]) for n in names}
    meta["extracted"] = {n: len(frames[n]) for n in names}
    boxes = detection_boxes(run, names, align) if align else {}
    if align:
        meta["alignment"] = {k: v for k, v in align.items() if k != "matrix"}
    path = build_viewer(out, splats, frame, cams, meta, boxes)
    log(f"Viewer: {path}  (serve the folder, e.g. python -m http.server -d {out})")
    return path
