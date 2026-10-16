"""Shared 3D scene from two sessions with MapAnything (Apache-2.0 weights).

Two ways to get the cameras (`whatchanged run --poses`):

  mapanything  One joint pass over frames from both videos fixes every camera in a single frame.
  colmap       The cameras come from the same COLMAP model the photoreal splats are trained on
               (room3d), so change boxes land on the right things in the splat viewer.

Either way, depth used for change testing comes from separate A-only and B-only passes, so the
network never sees both versions of a changed spot at once, and everything is levelled so the
floor is z = 0, +z up, metres.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import fit_floor, levelling_transform, transform_poses

MODEL_ID = "facebook/map-anything-apache"


@dataclass
class Session:
    name: str
    images: np.ndarray      # (N, H, W, 3) uint8, model resolution
    depth: np.ndarray       # (N, H, W) float32, metres along camera z
    conf: np.ndarray        # (N, H, W) float32
    mask: np.ndarray        # (N, H, W) bool, valid geometry
    K: np.ndarray           # (N, 3, 3)
    poses: np.ndarray       # (N, 4, 4) cam2world, OpenCV axes, levelled world
    times: np.ndarray       # (N,) seconds in the source video
    paths: list[str]

    def points(self, conf_pct: float = 30.0, stride: int = 2, with_frames: bool = False):
        """World points and colours from all frames, keeping the more confident pixels."""
        pts, cols, fids = [], [], []
        thr = np.percentile(self.conf[self.mask], conf_pct) if self.mask.any() else 0
        for i in range(len(self.depth)):
            d = self.depth[i, ::stride, ::stride]
            m = self.mask[i, ::stride, ::stride] & (self.conf[i, ::stride, ::stride] >= thr) & (d > 0)
            v, u = np.nonzero(m)
            u, v = u * stride, v * stride
            z = self.depth[i, v, u]
            K = self.K[i]
            x = (u - K[0, 2]) / K[0, 0] * z
            y = (v - K[1, 2]) / K[1, 1] * z
            pc = np.stack([x, y, z], 1)
            P = self.poses[i]
            pts.append(pc @ P[:3, :3].T + P[:3, 3])
            cols.append(self.images[i, v, u])
            fids.append(np.full(len(u), i, np.int32))
        if with_frames:
            return np.concatenate(pts), np.concatenate(cols), np.concatenate(fids)
        return np.concatenate(pts), np.concatenate(cols)

    def save(self, path: str | Path) -> None:
        # conf stays float32: MapAnything's confidence starts at 1.0 and a third of pixels sit within
        # 0.001 of it, so float16 rounds the 30th-percentile cut to exactly 1.0 and keeps everything.
        np.savez_compressed(path, images=self.images, depth=self.depth.astype(np.float16),
                            conf=self.conf.astype(np.float32), mask=self.mask, K=self.K,
                            poses=self.poses, times=self.times,
                            paths=np.array([f"frames/{Path(p).name}" for p in self.paths]))

    @classmethod
    def load(cls, name: str, path: str | Path) -> "Session":
        z = np.load(path)
        run_dir = Path(path).resolve().parent.parent          # <run>/cache/A.npz
        paths = [p if Path(p).exists() else str(run_dir / "frames" / Path(p).name) for p in z["paths"]]
        return cls(name, z["images"], z["depth"].astype(np.float32), z["conf"].astype(np.float32),
                   z["mask"], z["K"], z["poses"], z["times"], paths)


def pick_device(requested: str = "auto") -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    # MapAnything runs reliably on Apple CPUs (about 30 s for 16 views on an M4 Pro);
    # MPS is opt-in with --device mps.
    return "cpu"


_MODEL = None


def _model(device: str):
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        from mapanything.models import MapAnything
        _MODEL = MapAnything.from_pretrained(MODEL_ID).to(device).eval()
    return _MODEL


def _infer(paths: list[str], device: str, cameras: list[dict] | None = None) -> dict:
    """MapAnything on these images. With `cameras` (COLMAP pinhole, one per image), "K" is COLMAP's
    at model resolution and the network's own guess is kept as "K_net".

    The cameras are not fed to the network: on the trial it returned its own focal (about 2x too
    short) whether or not it was given one.
    """
    import torch
    from mapanything.utils.image import load_images

    views = load_images(paths)
    with torch.no_grad():
        preds = _model(device).infer(
            views, memory_efficient_inference=True, use_amp=(device == "cuda"), amp_dtype="bf16",
            apply_mask=True, mask_edges=True, apply_confidence_mask=False,
        )
    get = lambda k: np.stack([p[k][0].float().cpu().numpy() for p in preds])  # noqa: E731
    out = {
        "images": (np.clip(get("img_no_norm"), 0, 1) * 255).astype(np.uint8)
        if get("img_no_norm").max() <= 1.01 else get("img_no_norm").astype(np.uint8),
        "depth": get("depth_z")[..., 0],
        "conf": get("conf"),
        "mask": get("mask")[..., 0] > 0.5,
        "K": get("intrinsics"),
        "poses": get("camera_poses"),
    }
    if cameras is not None:
        H, W = out["depth"].shape[1:]
        out["K_net"] = out["K"]
        out["K"] = np.stack([model_K(cam, (W, H)) for cam in cameras])
    return out


def model_K(cam: dict, model_wh: tuple[int, int]) -> np.ndarray:
    """A COLMAP pinhole camera's K on the image MapAnything sees (cover-resize, then centre crop).

    COLMAP puts pixel centres at +0.5, OpenCV (and every projection here) at 0.
    """
    p = cam["params"]
    fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if len(p) == 3 else p[:4]
    W1, H1 = cam["width"], cam["height"]
    tW, tH = model_wh
    s = max(tW / W1, tH / H1) + 1e-8                      # as mapanything.utils.cropping
    w, h = int(np.floor(W1 * s)), int(np.floor(H1 * s))
    sx, sy = w / W1, h / H1
    left, top = (w - tW) // 2, (h - tH) // 2
    return np.array([[fx * sx, 0, cx * sx - left - 0.5], [0, fy * sy, cy * sy - top - 0.5], [0, 0, 1]])


def reconstruct(frames_a: list, frames_b: list, device: str = "auto",
                depth_from: str = "session", log=print) -> tuple[Session, Session, dict]:
    """frames_*: lists of frames.Frame. Returns levelled sessions A, B and a diagnostics dict."""
    device = pick_device(device)
    pa, pb = [f.path for f in frames_a], [f.path for f in frames_b]
    log(f"3D: joint pass over {len(pa) + len(pb)} views on {device}")
    joint = _infer(pa + pb, device)
    na = len(pa)
    split = lambda d, s: {k: (v[:na] if s == "A" else v[na:]) for k, v in d.items()}  # noqa: E731
    out = {"A": split(joint, "A"), "B": split(joint, "B")}
    diag = {"device": device, "views": {"A": len(pa), "B": len(pb)}, "depth_from": depth_from}

    if depth_from == "session":
        for name, paths in (("A", pa), ("B", pb)):
            log(f"3D: {name}-only pass for depth ({len(paths)} views)")
            sess = _infer(paths, device)
            # Bring each frame's depth to the joint pass's scale with a robust per-frame ratio over
            # pixels both passes trust. (Fitting scale to a handful of camera centres is fragile.)
            ratios = []
            for i in range(len(paths)):
                m = out[name]["mask"][i] & sess["mask"][i] & (sess["depth"][i] > 0.05) & (out[name]["depth"][i] > 0.05)
                ratios.append(float(np.median(out[name]["depth"][i][m] / sess["depth"][i][m])) if m.sum() > 500 else np.nan)
            ratios = np.array(ratios)
            fill = np.nanmedian(ratios)
            ratios = np.where(np.isfinite(ratios), ratios, fill)
            spread = float((np.percentile(ratios, 90) - np.percentile(ratios, 10)) / fill)
            scaled = sess["depth"] * ratios[:, None, None]
            leak = np.median(np.abs(scaled - out[name]["depth"])[out[name]["mask"] & sess["mask"]])
            diag[f"session_{name}_depth"] = {"scale_median": round(float(fill), 4), "scale_spread_p10_p90": round(spread, 4),
                                             "joint_vs_session_m_median": round(float(leak), 4)}
            if spread > 0.06:
                log(f"warning: {name}-only depth scale varies {spread:.0%} across frames; geometry may be unreliable")
            out[name]["depth"] = scaled.astype(np.float32)
            out[name]["conf"] = sess["conf"]
            out[name]["mask"] = sess["mask"] & out[name]["mask"]

    times = {"A": np.array([f.t for f in frames_a]), "B": np.array([f.t for f in frames_b])}
    sessions = {k: Session(k, v["images"], v["depth"], v["conf"], v["mask"], v["K"], v["poses"], times[k],
                           pa if k == "A" else pb) for k, v in out.items()}

    level_and_align(sessions["A"], sessions["B"], diag, log)
    return sessions["A"], sessions["B"], diag

def level_and_align(A: Session, B: Session, diag: dict, log=print) -> None:
    """Put the floor at z = 0 (+z up), then align B onto A on unchanged structure. In place."""
    sessions = {"A": A, "B": B}
    # Level the world: floor plane -> z = 0, +z up.
    pts = np.concatenate([sessions[k].points(conf_pct=50, stride=4)[0] for k in "AB"])
    poses = np.concatenate([sessions[k].poses for k in "AB"])
    up_hint = -poses[:, :3, 1].mean(0)          # OpenCV: image-up is camera -Y
    fwd_hint = poses[:, :3, 2].mean(0)
    n, d = fit_floor(pts, up_hint, poses[:, :3, 3])
    R, t = levelling_transform(n, d, fwd_hint)
    for k in "AB":
        sessions[k].poses = transform_poses(1.0, R, t, sessions[k].poses)
    heights = np.concatenate([sessions[k].poses[:, 2, 3] for k in "AB"])
    diag["camera_height_m"] = {"median": round(float(np.median(heights)), 3),
                               "p10": round(float(np.percentile(heights, 10)), 3),
                               "p90": round(float(np.percentile(heights, 90)), 3)}
    if np.percentile(heights, 90) - np.percentile(heights, 10) > 1.0 or not 0.5 < np.median(heights) < 2.2:
        log("warning: camera heights look implausible for a handheld walk; the floor fit may be wrong")

    # Align B onto A on structure that did not change (4-DoF robust ICP), then measure agreement.
    from .align import icp_4dof, nn_stats, subsample
    pa_ = subsample(sessions["A"].points(conf_pct=50, stride=4)[0])
    pb_ = subsample(sessions["B"].points(conf_pct=50, stride=4)[0])
    diag["alignment_before_icp_m"] = nn_stats(pa_, pb_)
    T, icp = icp_4dof(pb_, pa_)
    sessions["B"].poses = T[None] @ sessions["B"].poses
    diag["icp_B_to_A"] = icp
    pb_ = pb_ @ T[:3, :3].T + T[:3, 3]
    diag["alignment_B_to_A_m"] = nn_stats(pa_, pb_)
    log(f"3D: B aligned onto A by {icp['yaw_deg']:+.2f} deg and {np.linalg.norm(icp['shift_m']) * 100:.1f} cm; "
        f"static surfaces agree to {diag['alignment_B_to_A_m']['p50'] * 100:.1f} cm (median), "
        f"cameras at {diag['camera_height_m']['median']:.2f} m")


# ---------------------------------------------------------------- COLMAP poses (shared with the splats)

def colmap_model(videos: list[str], work: Path, log=print) -> tuple[dict, Path]:
    """room3d's sharp frames and ONE COLMAP model for both videos; reused when `work` has them.

    `work` can be a `whatchanged room A B` work folder, so the splats and the detection share poses.
    """
    from . import room3d
    names = ["A", "B"]
    fj, dense = work / "frames.json", work / "dense"
    if fj.exists() and (dense / "sparse" / "0" / "images.bin").exists():
        frames = json.loads(fj.read_text())
        if sorted(frames) != names:
            raise SystemExit(f"{work} holds {len(frames)} video(s), not a before/after pair; build it with "
                             "`whatchanged room BEFORE AFTER`")
        for n, v in zip(names, videos):
            if Path(frames[n][0]["video"]).name != Path(v).name:
                log(f"warning: {work} was built from {frames[n][0]['video']} as {n}, not {v}")
        log(f"Poses: reusing the COLMAP model in {work}")
        return frames, dense
    work.mkdir(parents=True, exist_ok=True)
    log("Frames: sharpest per 0.25 s window, as for the splats")
    frames = {n: room3d.extract_frames(v, work / "images" / n, n, log=log) for n, v in zip(names, videos)}
    fj.write_text(json.dumps(frames, indent=1))
    log("Poses: COLMAP over both videos in one model")
    return frames, room3d.run_colmap(work, [[f["name"] for f in frames[n]] for n in names], log=log)


def pick_posed_keyframes(frames: list[dict], posed: set[str], n: int) -> list[dict]:
    """About n posed frames spread evenly in time: the sharpest in each slot."""
    cand = sorted((f for f in frames if f["name"] in posed), key=lambda f: f["t"])
    if not cand:
        return []
    edges = np.linspace(cand[0]["t"], cand[-1]["t"] + 1e-6, n + 1)
    slots = [[f for f in cand if lo <= f["t"] < hi] for lo, hi in zip(edges[:-1], edges[1:])]
    return [max(s, key=lambda f: f["sharpness"]) for s in slots if s]


# Depth from MapAnything is trusted only where COLMAP points pin it down (see reconstruct_colmap).
ANCHOR_SIGMA_PX = 18.0    # how far, in model pixels, one COLMAP point's correction spreads
ANCHOR_MIN = 2.0          # pixels with less point weight than this (about 2 points right there) are left out


def fit_depth(d: np.ndarray, z: np.ndarray, fallback: float, min_points: int = 200) -> dict:
    """z ≈ a·d + b for one frame (network depth d, COLMAP depth z at the same pixels), robustly.

    Network depth is often stretched or squashed against real depth, not just scaled, so an offset is
    fitted too when the frame has enough points over enough range of depth to pin it down; otherwise
    a plain scale (`fallback` when there are hardly any points). Outside the depth range the points
    cover, the ratio z/d is held at its value at the edge instead of extrapolating the line.
    """
    if len(z) < 30:
        return {"kind": "none", "a": fallback, "b": 0.0, "res": float("nan")}
    fit = {"kind": "scale", "a": float(np.median(z / d)), "b": 0.0}
    if len(z) >= min_points and np.percentile(d, 90) > 1.3 * np.percentile(d, 10):
        keep = np.ones(len(z), bool)
        for _ in range(3):
            fa, fb = np.polyfit(d[keep], z[keep], 1)
            r = z - (fa * d + fb)
            keep = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[keep])) + 1e-9
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        cand = {"kind": "affine", "a": float(fa), "b": float(fb), "lo": float(lo), "hi": float(hi)}
        if fa > 0 and np.all(correct_depth(np.array([lo, hi]), cand) > 0):
            fit = cand
    fit["res"] = float(np.median(np.abs(z - correct_depth(d, fit)) / z))
    return fit


def correct_depth(depth: np.ndarray, fit: dict) -> np.ndarray:
    """Apply a fit_depth result to a depth map (COLMAP units out)."""
    if fit["kind"] != "affine":
        return fit["a"] * depth
    return depth * (fit["a"] + fit["b"] / np.clip(depth, fit["lo"], fit["hi"]))   # = a·d + b inside the range


def local_correction(d: np.ndarray, z: np.ndarray, u: np.ndarray, v: np.ndarray, fit: dict,
                     hw: tuple[int, int], sigma: float = ANCHOR_SIGMA_PX) -> tuple[np.ndarray, np.ndarray]:
    """What the frame-wide fit still gets wrong, spread from the COLMAP points over the image.

    Normalised convolution of each point's log misfit; far from any point the correction fades to 0.
    Also returns the point weight at each pixel (1 = one point right there), i.e. how anchored it is.
    """
    from scipy import ndimage
    S, N = np.zeros(hw), np.zeros(hw)
    np.add.at(S, (v, u), np.log(z / correct_depth(d, fit)))
    np.add.at(N, (v, u), 1.0)
    S, N = (ndimage.gaussian_filter(x, sigma, mode="constant") * (2 * np.pi * sigma ** 2) for x in (S, N))
    return S / (N + 0.5), N


def reconstruct_colmap(videos: list[str], out: Path, n_frames: int = 32, device: str = "auto",
                       work: Path | None = None, log=print) -> tuple[Session, Session, dict, list[dict]]:
    """Sessions whose cameras are COLMAP's and whose depth is MapAnything's, in metres.

    Each frame's depth is fitted to the COLMAP points that frame saw (scale and offset, see fit_depth,
    then a local correction around the points), which takes out the network's frame-to-frame wobble and
    its stretching of depth; depth with no point near it is left out. The one metric scale for the model
    is the median of MapAnything depth over COLMAP depth across all frames (as room3d.metric_scale),
    after correcting the network's depth for its focal-length guess. Then the world is levelled with
    room3d's level_up and find_floor, so it matches the splat viewer's.
    """
    from . import room3d
    from .align import nn_stats, subsample

    device = pick_device(device)
    work = Path(work) if work else out / "colmap"
    frames, dense = colmap_model(videos, work, log)
    sp = dense / "sparse" / "0"
    imgs = room3d.read_images(sp / "images.bin")
    cams = room3d.read_cameras(sp / "cameras.bin")
    xyz, _, seen = room3d.read_points(sp / "points3D.bin")
    obs: dict[int, list[int]] = {}
    for k, ids in enumerate(seen):
        for i in ids:
            obs.setdefault(int(i), []).append(k)
    by_name = {im["name"]: iid for iid, im in imgs.items()}
    diag = {"device": device, "poses_from": "colmap", "views": {}, "depth_from": "session"}

    (out / "frames").mkdir(parents=True, exist_ok=True)
    raw, keyframes = {}, []
    for name in "AB":
        posed = {f["name"] for f in frames[name] if len(obs.get(by_name.get(f["name"], -1), [])) >= 50}
        keys = pick_posed_keyframes(frames[name], posed, n_frames)
        paths = []
        for i, f in enumerate(keys):
            p = out / "frames" / f"{name}_{i:03d}.jpg"
            shutil.copy(dense / "images" / f["name"], p)     # undistorted, so COLMAP's pinhole K holds
            paths.append(str(p))
            keyframes.append({"session": name, "index": i, "path": str(p), "t": f["t"], "sharpness": f["sharpness"]})
        ids = [by_name[f["name"]] for f in keys]
        diag["views"][name] = len(keys)
        log(f"3D: {name}: {len(keys)} keyframes from {len(posed)}/{len(frames[name])} COLMAP-posed frames; "
            f"MapAnything depth on {device}")
        pred = _infer(paths, device, cameras=[cams[imgs[i]["camera_id"]] for i in ids])
        pred["c2w"] = np.stack([room3d.cam_to_world(imgs[i]) for i in ids])
        pred["times"] = np.array([f["t"] for f in keys])
        pred["paths"] = paths
        # Network depth and COLMAP depth at every COLMAP point the frame saw.
        H, W = pred["depth"].shape[1:]
        pred["pairs"], pred["uv"] = [], []
        for j, iid in enumerate(ids):
            w2c = np.linalg.inv(pred["c2w"][j])
            pc = xyz[obs[iid]] @ w2c[:3, :3].T + w2c[:3, 3]
            pc = pc[pc[:, 2] > 1e-6]
            K = pred["K"][j]
            u = np.round(K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2]).astype(int)
            v = np.round(K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]).astype(int)
            ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            d = pred["depth"][j][v[ok], u[ok]]
            good = pred["mask"][j][v[ok], u[ok]] & (d > 0.05)
            pred["pairs"].append((d[good], pc[ok, 2][good]))
            pred["uv"].append((u[ok][good], v[ok][good]))
        raw[name] = pred

    # Metric scale: MapAnything's depth, corrected for its own focal-length guess. It judges distance
    # from how big things look, so a field of view guessed k times too wide gives depth about k times
    # too short (on the trial: 2.3x, which put the phone at 0.72 m instead of the bed and desk's 1.45 m).
    focal = np.concatenate([p["K"][:, 0, 0] / p["K_net"][:, 0, 0] for p in raw.values()])
    ratios = [d * k / z for p in raw.values() for (d, z), k in zip(p["pairs"], p["K"][:, 0, 0] / p["K_net"][:, 0, 0])]
    s = float(np.median(np.concatenate(ratios)))
    s_raw = float(np.median(np.concatenate([d / z for p in raw.values() for d, z in p["pairs"]])))
    diag["metric_scale"] = {"m_per_colmap_unit": round(s, 6), "uncorrected": round(s_raw, 6),
                            "network_focal_short_by": round(float(np.median(focal)), 3)}
    log(f"3D: metric scale {s:.4f} m per COLMAP unit (MapAnything guessed a focal length "
        f"{np.median(focal):.2f}x too short; uncorrected scale {s_raw:.4f})")
    # Shape: each frame's depth fitted to its own COLMAP points (z = a*d + b), then the leftover misfit
    # corrected locally around the points. Depth with no COLMAP point near it is dropped: the network
    # works from its own (wrong) field of view, so a plain white wall can sit 40% too near, and the
    # other video then "sees through" that phantom wall. Unanchored spots become "can't tell" instead.
    for name, p in raw.items():
        k = p["K"][:, 0, 0] / p["K_net"][:, 0, 0]
        fits = [fit_depth(d, z, fallback=ki / s) for (d, z), ki in zip(p["pairs"], k)]
        depth = s * np.stack([correct_depth(dm, f) for dm, f in zip(p["depth"], fits)])
        for j, ((d, z), (u, v), f) in enumerate(zip(p["pairs"], p["uv"], fits)):
            r, w = local_correction(d, z, u, v, f, depth.shape[1:])
            depth[j] *= np.exp(r)
            p["mask"][j] &= w >= ANCHOR_MIN
        p["mask"] = p["mask"] & (depth > 0.05)
        p["depth"] = np.where(p["mask"], depth, 0).astype(np.float32)
        res = np.array([f["res"] for f in fits if f["kind"] != "none"])
        diag[f"session_{name}_depth"] = {
            "fits": {kind: sum(f["kind"] == kind for f in fits) for kind in ("affine", "scale", "none")},
            "anchored_pixels": round(float(p["mask"].mean()), 3),
            "residual_vs_colmap_median": round(float(np.median(res)), 4),
            "residual_vs_colmap_p90": round(float(np.percentile(res, 90)), 4),
            "colmap_points_per_frame": int(np.median([len(z) for _, z in p["pairs"]]))}
        log(f"3D: {name} depth fitted to COLMAP points per frame ({sum(f['kind'] == 'affine' for f in fits)}/{len(fits)} "
            f"with an offset; off by {np.median(res):.1%} median, {np.percentile(res, 90):.1%} p90); "
            f"{p['mask'].mean():.0%} of pixels anchored")

    # Level exactly as the splat viewer does: up from the cameras' x axes, floor = lowest substantial layer.
    all_c2w = np.stack([room3d.cam_to_world(imgs[i]) for i in sorted(imgs, key=lambda i: imgs[i]["name"])])
    up = room3d.level_up(all_c2w)
    n, d = room3d.find_floor(xyz, up, all_c2w[:, :3, 3])
    R, t = levelling_transform(n, d * s, all_c2w[0, :3, 2])
    L = np.eye(4)
    L[:3, :3], L[:3, 3] = s * R, t                       # COLMAP world -> levelled metric world
    sessions = {k: Session(k, p["images"], p["depth"], p["conf"], p["mask"], p["K"],
                           transform_poses(s, R, t, p["c2w"]), p["times"], p["paths"]) for k, p in raw.items()}
    A, B = sessions["A"], sessions["B"]

    heights = np.concatenate([S.poses[:, 2, 3] for S in (A, B)])
    diag["camera_height_m"] = {"median": round(float(np.median(heights)), 3),
                               "p10": round(float(np.percentile(heights, 10)), 3),
                               "p90": round(float(np.percentile(heights, 90)), 3)}
    if np.percentile(heights, 90) - np.percentile(heights, 10) > 1.0 or not 0.5 < np.median(heights) < 2.2:
        log("warning: camera heights look implausible for a handheld walk; the floor fit may be wrong")
    # Both videos sit in one bundle-adjusted model, so B is on A by construction: measure, don't move.
    diag["alignment_B_to_A_m"] = nn_stats(subsample(A.points(conf_pct=50, stride=4)[0]),
                                          subsample(B.points(conf_pct=50, stride=4)[0]))
    diag["colmap"] = {"work": str(work.resolve()), "to_levelled": np.round(L, 8).tolist(),
                      "metric_scale_m_per_unit": round(s, 6),
                      "registered": {k: f"{sum(f['name'] in by_name for f in frames[k])}/{len(frames[k])}" for k in "AB"}}
    log(f"3D: COLMAP poses, {s:.4f} m per COLMAP unit; static surfaces agree to "
        f"{diag['alignment_B_to_A_m']['p50'] * 100:.1f} cm (median), cameras at {diag['camera_height_m']['median']:.2f} m")
    return A, B, diag, keyframes
