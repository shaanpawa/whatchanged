"""Shared 3D scene from two sessions with MapAnything (Apache-2.0 weights).

One joint pass over frames from both videos fixes every camera in a single frame.
Depth used for change testing comes from separate A-only and B-only passes, aligned to
that frame through their camera centres, so the network never sees both versions of a
changed spot at once. Everything is then levelled so the floor is z = 0, +z up, metres.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import fit_floor, levelling_transform, transform_poses, umeyama

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
        np.savez_compressed(path, images=self.images, depth=self.depth.astype(np.float16),
                            conf=self.conf.astype(np.float16), mask=self.mask, K=self.K,
                            poses=self.poses, times=self.times, paths=np.array(self.paths))

    @classmethod
    def load(cls, name: str, path: str | Path) -> "Session":
        z = np.load(path)
        return cls(name, z["images"], z["depth"].astype(np.float32), z["conf"].astype(np.float32),
                   z["mask"], z["K"], z["poses"], z["times"], list(z["paths"]))


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


def _infer(paths: list[str], device: str) -> dict:
    import torch
    from mapanything.utils.image import load_images

    views = load_images(paths)
    with torch.no_grad():
        preds = _model(device).infer(
            views, memory_efficient_inference=True, use_amp=(device == "cuda"), amp_dtype="bf16",
            apply_mask=True, mask_edges=True, apply_confidence_mask=False,
        )
    get = lambda k: np.stack([p[k][0].float().cpu().numpy() for p in preds])  # noqa: E731
    return {
        "images": (np.clip(get("img_no_norm"), 0, 1) * 255).astype(np.uint8)
        if get("img_no_norm").max() <= 1.01 else get("img_no_norm").astype(np.uint8),
        "depth": get("depth_z")[..., 0],
        "conf": get("conf"),
        "mask": get("mask")[..., 0] > 0.5,
        "K": get("intrinsics"),
        "poses": get("camera_poses"),
    }


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
            s, R, t = umeyama(sess["poses"][:, :3, 3], out[name]["poses"][:, :3, 3])
            moved = s * sess["poses"][:, :3, 3] @ R.T + t
            resid = np.linalg.norm(moved - out[name]["poses"][:, :3, 3], axis=1)
            diag[f"session_{name}_to_joint"] = {"scale": round(s, 4), "camera_residual_m_median": round(float(np.median(resid)), 4)}
            # Keep joint poses (shared frame); take this session's depth, rescaled into the joint frame.
            leak = np.median(np.abs(sess["depth"] * s - out[name]["depth"])[out[name]["mask"]])
            diag[f"joint_vs_session_depth_{name}_m_median"] = round(float(leak), 4)
            out[name]["depth"] = sess["depth"] * s
            out[name]["conf"] = sess["conf"]
            out[name]["mask"] = sess["mask"] & out[name]["mask"]

    times = {"A": np.array([f.t for f in frames_a]), "B": np.array([f.t for f in frames_b])}
    sessions = {k: Session(k, v["images"], v["depth"], v["conf"], v["mask"], v["K"], v["poses"], times[k],
                           pa if k == "A" else pb) for k, v in out.items()}

    # Level the world: floor plane → z = 0, +z up.
    pts = np.concatenate([sessions[k].points(conf_pct=50, stride=4)[0] for k in "AB"])
    poses = np.concatenate([sessions[k].poses for k in "AB"])
    up_hint = -poses[:, :3, 1].mean(0)          # OpenCV: image-up is camera -Y
    fwd_hint = poses[:, :3, 2].mean(0)
    n, d = fit_floor(pts, up_hint, poses[:, :3, 3])
    R, t = levelling_transform(n, d, fwd_hint)
    for k in "AB":
        sessions[k].poses = transform_poses(1.0, R, t, sessions[k].poses)
    heights = np.concatenate([sessions[k].poses[:, 2, 3] for k in "AB"])
    diag["camera_height_m_median"] = round(float(np.median(heights)), 3)

    # How well do static surfaces of A and B agree? (Changed objects inflate p90, not p50.)
    from scipy.spatial import cKDTree
    pa_, _ = sessions["A"].points(conf_pct=50, stride=4)
    pb_, _ = sessions["B"].points(conf_pct=50, stride=4)
    rng = np.random.default_rng(0)
    sb = pb_[rng.choice(len(pb_), min(50_000, len(pb_)), replace=False)]
    dist, _ = cKDTree(pa_).query(sb, k=1)
    diag["alignment_B_to_A_m"] = {"p50": round(float(np.percentile(dist, 50)), 4),
                                  "p90": round(float(np.percentile(dist, 90)), 4)}
    log(f"3D: static alignment p50 {diag['alignment_B_to_A_m']['p50'] * 100:.1f} cm, "
        f"cameras at {diag['camera_height_m_median']:.2f} m")
    return sessions["A"], sessions["B"], diag
