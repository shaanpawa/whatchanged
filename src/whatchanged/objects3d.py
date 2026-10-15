"""Labelled oriented 3D boxes for every object, on a laptop CPU.

An open-vocabulary 2D detector (OWLv2 or Grounding DINO, both Apache-2.0) names things in
a few keyframes. Each 2D box is lifted to 3D with the session's own depth, intrinsics and
pose: keep the dominant depth layer inside the box, back-project it, drop support surfaces.
Detections of the same thing from different frames are fused when each one's 3D points
re-project inside the other's 2D boxes; the fused points are carved by those boxes' viewing
frusta, and each object gets a gravity-aligned oriented box (yaw from a min-area footprint, z from
the point heights) in the levelled z-up metric world. Output also in SpatialLM's
`bbox_i=Bbox(class,x,y,z,angle_z,scale_x,scale_y,scale_z)` text format.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import project
from .recon import Session

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
GDINO_MODEL = "IDEA-Research/grounding-dino-tiny"

VOCAB3D = [
    "chair", "stool", "sofa", "armchair", "table", "desk", "bed", "pillow", "lamp", "plant",
    "potted plant", "bin", "basket", "suitcase", "backpack", "bag", "cardboard box", "box",
    "book", "laptop", "monitor", "keyboard", "television", "speaker", "bottle", "mug", "cup",
    "bowl", "vase", "picture frame", "clock", "towel", "jacket", "shoes", "toy", "toy vehicle",
    "ball", "fan", "heater", "bucket", "shelf", "cabinet", "drawer unit", "rug", "placemat",
    "cushion", "umbrella", "kettle", "tray", "jar", "pen", "scissors", "phone", "remote control",
    "refrigerator", "microwave", "oven", "sink", "door", "window",
]
# Labels that are supports or flat things: never strip their own support plane.
FLAT = {"rug", "placemat", "tray"}
FLAT_OR_SUPPORT = {"table", "desk", "shelf", "cabinet", "drawer unit", "bed", "sofa", "rug",
                   "placemat", "tray", "refrigerator", "oven", "sink", "door", "window"}
SYNONYMS = {"potted plant": "plant", "cardboard box": "box", "toy vehicle": "toy", "armchair": "chair",
            "stool": "chair", "basket": "bin"}


@dataclass
class Det2D:
    frame: int
    box: np.ndarray          # x0, y0, x1, y1 in model (depth) pixels
    label: str
    score: float


@dataclass
class Box3D:
    label: str
    score: float             # best 2D score
    centre: np.ndarray       # (3,) metres, levelled world, z up
    size: np.ndarray         # (3,) extent along the box's own x (yaw dir), y, z
    yaw: float               # radians about +z
    frames: list = field(default_factory=list)
    votes: dict = field(default_factory=dict)   # label -> summed score
    points: np.ndarray | None = None            # fused, carved world points

    def corners(self) -> np.ndarray:
        """(8, 3): bottom four then top four, counter-clockwise from above."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        h = self.size / 2
        sx = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
        sy = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
        sz = np.array([-1, -1, -1, -1, 1, 1, 1, 1])
        return (np.stack([sx * h[0], sy * h[1], sz * h[2]], 1) @ R.T) + self.centre

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(self.score, 3),
                "centre_m": [round(float(v), 3) for v in self.centre],
                "size_m": [round(float(v), 3) for v in self.size],
                "yaw_deg": round(float(np.degrees(self.yaw)), 1),
                "frames": [int(f) for f in self.frames]}

    def spatiallm(self, i: int) -> str:
        x, y, z = self.centre
        return (f"bbox_{i}=Bbox({self.label.replace(' ', '_')},{x:.2f},{y:.2f},{z:.2f},"
                f"{self.yaw:.4f},{self.size[0]:.2f},{self.size[1]:.2f},{self.size[2]:.2f})")


# ---------------------------------------------------------------- 2D detection

_DET = {}


def _load(kind: str, device: str):
    key = (kind, device)
    if key not in _DET:
        import torch  # noqa: F401
        if kind == "owlv2":
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            proc = Owlv2Processor.from_pretrained(OWL_MODEL)
            model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL)
        else:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            proc = AutoProcessor.from_pretrained(GDINO_MODEL)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL)
        _DET[key] = (model.to(device).eval(), proc)
    return _DET[key]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    order = np.argsort(-scores)
    keep = []
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while len(order):
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        o = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][o < iou]
    return np.array(keep, int)


def _original_to_model(xyxy: np.ndarray, orig_wh: tuple[int, int], model_wh: tuple[int, int]) -> np.ndarray:
    """Inverse of semantics._model_to_original (MapAnything cover-resize + centre-crop)."""
    (W1, H1), (tW, tH) = orig_wh, model_wh
    s = max(tW / W1, tH / H1)
    left, top = (int(W1 * s) - tW) // 2, (int(H1 * s) - tH) // 2
    out = xyxy * s - np.array([left, top, left, top])
    return np.clip(out, 0, [tW - 1, tH - 1, tW - 1, tH - 1])


def detect(S: Session, frames: list[int], vocab: list[str] = VOCAB3D, kind: str = "owlv2",
           device: str = "cpu", threshold: float | None = None, use_original: bool = True,
           log=print) -> tuple[list[Det2D], dict]:
    """Open-vocabulary 2D boxes on the chosen frames, in model (depth) pixel coordinates."""
    import torch
    model, proc = _load(kind, device)
    thr = threshold if threshold is not None else (0.25 if kind == "owlv2" else 0.30)
    H, W = S.depth.shape[1:]
    dets, times = [], []
    for j in frames:
        img = Image.open(S.paths[j]).convert("RGB") if use_original and Path(S.paths[j]).exists() \
            else Image.fromarray(S.images[j])
        t0 = time.perf_counter()
        with torch.no_grad():
            if kind == "owlv2":
                inp = proc(text=[[f"a photo of a {v}" for v in vocab]], images=img, return_tensors="pt").to(device)
                out = model(**inp)
                side = max(img.size)                         # OWLv2 pads to a square
                res = proc.post_process_grounded_object_detection(out, threshold=thr, target_sizes=[(side, side)])[0]
                names = [vocab[int(k)] for k in res["labels"]]
            else:
                inp = proc(images=img, text=". ".join(vocab) + ".", return_tensors="pt").to(device)
                out = model(**inp)
                res = proc.post_process_grounded_object_detection(
                    out, inp.input_ids, threshold=thr, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
                names = list(res.get("text_labels", res.get("labels")))
        if device == "mps":
            torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        b = res["boxes"].float().cpu().numpy()
        sc = res["scores"].float().cpu().numpy()
        if len(b) == 0:
            continue
        b = _original_to_model(b, img.size, (W, H)) if img.size != (W, H) else b
        names = [SYNONYMS.get(n, n) for n in names]
        # drop whole-image boxes and slivers
        bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        ok = (bw * bh < 0.9 * W * H) & (bw > 6) & (bh > 6) & np.array([n in vocab or n in SYNONYMS.values() for n in names])
        b, sc, names = b[ok], sc[ok], [n for n, k in zip(names, ok) if k]
        # per-label NMS, then a label-agnostic pass for near-identical boxes (synonyms)
        keep = []
        for lab in set(names):
            idx = np.array([i for i, n in enumerate(names) if n == lab])
            keep += list(idx[_nms(b[idx], sc[idx], 0.4)])
        keep = np.array(keep, int)
        keep = keep[_nms(b[keep], sc[keep], 0.85)]
        dets += [Det2D(j, b[i], names[i], float(sc[i])) for i in keep]
    stats = {"detector": kind, "device": device, "frames": len(frames),
             "s_per_frame": round(float(np.median(times[1:] if len(times) > 1 else times)), 3),
             "first_frame_s": round(times[0], 3) if times else None, "dets_2d": len(dets)}
    log(f"boxes: {kind} on {device}: {stats['s_per_frame']} s/frame, {len(dets)} 2D boxes over {len(frames)} frames")
    return dets, stats


# ---------------------------------------------------------------- 2D -> 3D

def support_heights(S: Session, bin_m: float = 0.01, min_share: float = 0.03) -> np.ndarray:
    """Heights of big horizontal surfaces (floor, table tops): sharp peaks in the z histogram."""
    p, _ = S.points(conf_pct=30, stride=4)
    z = p[:, 2]
    z = z[(z > -0.1) & (z < 2.5)]
    hist, edges = np.histogram(z, bins=np.arange(-0.1, 2.5 + bin_m, bin_m))
    from scipy.ndimage import maximum_filter1d
    peaks = (hist == maximum_filter1d(hist, 9)) & (hist > min_share * len(z) / 5)
    # a surface is a sharp peak: much taller than its +-8 cm surroundings
    out = []
    for i in np.nonzero(peaks)[0]:
        ring = np.r_[hist[max(0, i - 8):max(0, i - 4)], hist[i + 5:i + 9]]
        if len(ring) and hist[i] > 3 * np.median(ring):
            out.append((edges[i] + edges[i + 1]) / 2)
    return np.array(out)


def _nested(d: Det2D, others: list[Det2D], inside: float = 0.7, rel_area: float = 0.5) -> list[np.ndarray]:
    """Other boxes in the same frame that sit mostly inside d and are much smaller: things on/in front of it."""
    out = []
    a = np.prod(d.box[2:] - d.box[:2])
    for o in others:
        if o is d or o.frame != d.frame:
            continue
        ao = np.prod(o.box[2:] - o.box[:2])
        lo, hi = np.maximum(d.box[:2], o.box[:2]), np.minimum(d.box[2:], o.box[2:])
        inter = np.prod(np.clip(hi - lo, 0, None))
        if ao < rel_area * a and inter > inside * ao:
            out.append(o.box)
        elif d.label in FLAT_OR_SUPPORT and o.label not in FLAT_OR_SUPPORT and ao < a and inter > 0:
            out.append(o.box)                  # anything standing on a support/flat thing
    return out


def lift(S: Session, d: Det2D, conf_thr: float, shrink: float = 0.12, stride: int = 2,
         bin_log: float = 0.03, exclude: list[np.ndarray] = ()) -> np.ndarray | None:
    """World points of the object inside a 2D box: dominant depth layer (centre-weighted unless flat),
    ignoring pixels of smaller objects detected inside it (a mug on the table is not the table)."""
    H, W = S.depth.shape[1:]
    x0, y0, x1, y1 = d.box
    bw, bh = x1 - x0, y1 - y0
    u0, u1 = int(max(0, x0 + shrink * bw)), int(min(W, x1 - shrink * bw))
    v0, v1 = int(max(0, y0 + shrink * bh)), int(min(H, y1 - shrink * bh))
    if u1 - u0 < 3 or v1 - v0 < 3:
        return None
    vv, uu = np.mgrid[v0:v1:stride, u0:u1:stride]
    vv, uu = vv.ravel(), uu.ravel()
    z = S.depth[d.frame, vv, uu]
    ok = S.mask[d.frame, vv, uu] & (S.conf[d.frame, vv, uu] >= conf_thr) & (z > 0.1)
    for e in exclude:
        ok &= ~((uu >= e[0]) & (uu <= e[2]) & (vv >= e[1]) & (vv <= e[3]))
    vv, uu, z = vv[ok], uu[ok], z[ok]
    if len(z) < 30:
        return None
    # centre weighting: an object is usually in the middle of its box, background at the rim.
    # Flat things and supports (table, rug) are not: weight them uniformly.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w = np.ones(len(z)) if d.label in FLAT_OR_SUPPORT else \
        np.exp(-(((uu - cx) / (0.35 * bw)) ** 2 + ((vv - cy) / (0.35 * bh)) ** 2))
    lz = np.log(z)
    edges = np.arange(lz.min(), lz.max() + bin_log, bin_log)
    if len(edges) < 2:
        edges = np.array([lz.min(), lz.min() + bin_log])
    hist, _ = np.histogram(lz, bins=edges, weights=w)
    occ = hist > 0.02 * hist.sum()
    # connected runs of occupied bins (allow one empty bin inside a run)
    runs, start = [], None
    for i in range(len(occ) + 1):
        on = i < len(occ) and (occ[i] or (i + 1 < len(occ) and occ[i + 1] and start is not None))
        if on and start is None:
            start = i
        elif not on and start is not None:
            runs.append((start, i))
            start = None
    if not runs:
        return None
    a, b = max(runs, key=lambda r: hist[r[0]:r[1]].sum())
    sel = (lz >= edges[a]) & (lz < edges[b] + 1e-9)
    if sel.sum() < 20:
        return None
    uu, vv, z = uu[sel], vv[sel], z[sel]
    K, P = S.K[d.frame], S.poses[d.frame]
    pc = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1)
    return pc @ P[:3, :3].T + P[:3, 3]


def _strip_supports(pts: np.ndarray, label: str, supports: np.ndarray, tol: float = 0.025) -> np.ndarray:
    keep = pts[:, 2] > 0.03 if label not in FLAT_OR_SUPPORT else np.ones(len(pts), bool)
    if label in FLAT and keep.any():           # flat things are thin: drop what stands on them
        keep &= pts[:, 2] < np.percentile(pts[keep, 2], 30) + 0.05
    if label not in FLAT_OR_SUPPORT and len(supports):
        med = np.median(pts[:, 2])
        for h in supports:
            if h < med:
                keep &= ~((pts[:, 2] > h - tol) & (pts[:, 2] < h + tol))
    return pts[keep]


# ---------------------------------------------------------------- fusion and box fitting

def _box_agreement(S: Session, P: np.ndarray, dets: list, margin: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Per point: share of the views (that see it in-image) whose detection box contains it; and that view count."""
    H, W = S.depth.shape[1:]
    yes, seen = np.zeros(len(P)), np.zeros(len(P))
    for d in dets:
        uv, z = project(P, S.poses[d.frame], S.K[d.frame])
        inimg = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        mx, my = margin * (d.box[2] - d.box[0]), margin * (d.box[3] - d.box[1])
        inbox = inimg & (uv[:, 0] >= d.box[0] - mx) & (uv[:, 0] <= d.box[2] + mx) & \
            (uv[:, 1] >= d.box[1] - my) & (uv[:, 1] <= d.box[3] + my)
        seen += inimg
        yes += inbox
    return yes / np.maximum(seen, 1), seen


def _inside_share(pts: np.ndarray, box: "Box3D", pad: float = 0.02) -> float:
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    r = (pts - box.centre) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return float((np.abs(r) <= box.size / 2 + pad).all(1).mean())


def fit_box(pts: np.ndarray, floor_snap: float = 0.08) -> tuple[np.ndarray, np.ndarray, float]:
    """Gravity-aligned oriented box: yaw minimising robust footprint area, z from height percentiles."""
    xy = pts[:, :2]
    mu = np.median(xy, 0)
    best = None
    for deg in range(0, 90, 2):
        t = np.radians(deg)
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])   # world -> box
        r = (xy - mu) @ R.T
        lo, hi = np.percentile(r, 2, 0), np.percentile(r, 98, 0)
        area = np.prod(hi - lo)
        if best is None or area < best[0]:
            best = (area, t, lo, hi, R)
    _, yaw, lo, hi, R = best
    ext = hi - lo
    if ext[1] > ext[0]:                       # make box-x the long side
        yaw += np.pi / 2
        ext = ext[::-1]
        lo, hi = np.array([lo[1], -hi[0]]), np.array([hi[1], -lo[0]])
        R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    mid = mu + ((lo + hi) / 2) @ R            # box -> world (R orthonormal)
    zlo, zhi = np.percentile(pts[:, 2], 1), np.percentile(pts[:, 2], 99)
    if zlo < floor_snap:
        zlo = 0.0
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2   # box is symmetric under 180 degrees
    return np.array([mid[0], mid[1], (zlo + zhi) / 2]), np.array([ext[0], ext[1], zhi - zlo]), float(yaw)


def boxes_for_session(S: Session, n_frames: int = 8, kind: str = "owlv2", device: str = "cpu",
                      vocab: list[str] = VOCAB3D, min_frames: int = 2, solo_score: float = 0.5, log=print) -> tuple[list[Box3D], dict]:
    t0 = time.perf_counter()
    frames = sorted(set(np.linspace(0, len(S.depth) - 1, min(n_frames, len(S.depth))).round().astype(int).tolist()))
    dets, stats = detect(S, frames, vocab, kind, device, log=log)
    conf_thr = float(np.percentile(S.conf[S.mask], 30))
    sup = support_heights(S)
    lifted = []
    for d in dets:
        p = lift(S, d, conf_thr, exclude=_nested(d, dets))
        if p is None:
            continue
        p = _strip_supports(p, d.label, sup)
        if len(p) < 20:
            continue
        ext = np.percentile(p, 98, 0) - np.percentile(p, 2, 0)
        if ext.max() > 4.0:                    # a "box" bigger than a room is background bleed
            continue
        lifted.append((d, p, None))

    # greedy fusion, strongest first. Two detections are the same object when each one's 3D points
    # land inside the other's 2D boxes (re-projection agreement): robust to partial, one-sided views.
    lifted.sort(key=lambda x: -x[0].score)
    rng = np.random.default_rng(0)
    sub = lambda P: P if len(P) <= 400 else P[rng.choice(len(P), 400, replace=False)]  # noqa: E731
    tracks = []
    for d, p, _ in lifted:
        ps = sub(p)
        best, best_o = None, 0.0
        for t in tracks:
            same = d.label in t["votes"]
            a = _box_agreement(S, ps, t["dets"])[0].mean()          # mine inside their boxes
            b = _box_agreement(S, t["sub"], [d])[0].mean()          # theirs inside my box
            if any(m.frame == d.frame for m in t["dets"]) and not same:
                continue                       # two labels in one frame are two things (toy on a mat)
            ok = (a > 0.5 and b > 0.3) if same else (a > 0.7 and b > 0.7)
            if ok and a + b > best_o:
                best, best_o = t, a + b
        if best is None:
            tracks.append({"dets": [d], "pts": [p], "sub": ps, "votes": {d.label: d.score}})
        else:
            best["dets"].append(d)
            best["pts"].append(p)
            best["sub"] = sub(np.concatenate([best["sub"], ps]))
            best["votes"][d.label] = best["votes"].get(d.label, 0) + d.score

    boxes = []
    for t in tracks:
        frames_seen = sorted({m.frame for m in t["dets"]})
        top = max(m.score for m in t["dets"])
        if len(frames_seen) < min_frames and top < solo_score:
            continue
        label = max(t["votes"], key=t["votes"].get)
        # carve: keep points that fall inside the 2D box in most member views that can see them
        # (intersection of the boxes' viewing frusta; trims depth bleed onto coplanar tables)
        P = np.concatenate(t["pts"])
        agree, seen = _box_agreement(S, P, t["dets"], margin=0.05)
        P = P[(agree >= 0.6) | (seen < 2)]
        if len(P) < 20:
            continue
        c, s, yaw = fit_box(P)
        boxes.append(Box3D(label, top, c, s, yaw, frames_seen, dict(t["votes"]), P))

    # a box swallowed by a bigger box of the same label is a duplicate part
    boxes.sort(key=lambda b: -np.prod(b.size))
    kept = []
    for b in boxes:
        dup = False
        for k in kept:
            if k.label == b.label:
                rel = np.abs(b.centre - k.centre) <= (k.size.max() / 2)
                if rel.all():
                    dup = True
                    break
        if not dup:
            kept.append(b)
    kept.sort(key=lambda b: -b.score)
    final = []
    for b in kept:
        part = False
        for k in final:
            if k.label in FLAT_OR_SUPPORT or b.label in FLAT_OR_SUPPORT:
                continue
            if _inside_share(b.points, k) > 0.6:
                part = True
                break
        if not part:
            final.append(b)
    # the same thing under two labels (grey placemat = "rug" and "placemat"): keep the better-voted one
    final.sort(key=lambda b: -sum(b.votes.values()))
    kept = []
    for b in final:
        if not any(_inside_share(b.points, k) > 0.5 and _inside_share(k.points, b) > 0.5 for k in kept):
            kept.append(b)
    stats.update({"lifted_3d": len(lifted), "tracks": len(tracks), "boxes": len(kept),
                  "support_heights_m": [round(float(h), 2) for h in sup],
                  "total_s": round(time.perf_counter() - t0, 2)})
    log(f"boxes: {len(kept)} labelled 3D boxes ({len(lifted)} lifted from {len(dets)} 2D) in {stats['total_s']} s")
    return kept, stats


# ---------------------------------------------------------------- hooks for the rest of the pipeline

def label_for_points(points: np.ndarray, boxes: list[Box3D], min_share: float = 0.3) -> tuple[str | None, float]:
    """Name a change blob (e.g. Blob.voxels) by the 3D box that contains most of it."""
    best, share = None, 0.0
    for b in boxes:
        s = _inside_share(points, b, pad=0.05)
        if s > share:
            best, share = b, s
    return (best.label, share) if best is not None and share >= min_share else (None, share)


def log_rerun(path: str, boxes: list[Box3D]) -> None:
    """Log boxes under `path` (e.g. "world/objects_A") in the current Rerun recording."""
    import rerun as rr
    rr.log(path, rr.Boxes3D(
        centers=[b.centre for b in boxes], half_sizes=[b.size / 2 for b in boxes],
        rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], radians=b.yaw) for b in boxes],
        labels=[b.label for b in boxes]), static=True)


def write(path: str | Path, boxes_by_session: dict[str, list[Box3D]], stats: dict | None = None) -> None:
    """objects.json (robot-friendly) + objects_<S>.txt in SpatialLM's Bbox text format."""
    import json
    path = Path(path)
    path.write_text(json.dumps({"frame": "levelled world: metres, z up, floor z=0; yaw about +z; size = (along yaw, across, height)",
                                "sessions": {k: [b.to_dict() for b in v] for k, v in boxes_by_session.items()},
                                "stats": stats or {}}, indent=1))
    for k, v in boxes_by_session.items():
        (path.parent / f"objects_{k}.txt").write_text("\n".join(b.spatiallm(i) for i, b in enumerate(v)) + "\n")


# ---------------------------------------------------------------- cache, relations, robot formats

def from_dict(d: dict) -> Box3D:
    return Box3D(d["label"], d["score"], np.array(d["centre_m"]), np.array(d["size_m"]),
                 float(np.radians(d["yaw_deg"])), d.get("frames", []), {}, None)


def load(path: str | Path) -> dict[str, list[Box3D]]:
    import json
    d = json.loads(Path(path).read_text())
    return {k: [from_dict(b) for b in v] for k, v in d["sessions"].items()}


def _footprint_contains(b: Box3D, xy: np.ndarray, pad: float = 0.05) -> bool:
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    r = (xy - b.centre[:2]) @ np.array([[c, -s], [s, c]])
    return bool((np.abs(r) <= b.size[:2] / 2 + pad).all())


def _gap(a: Box3D, b: Box3D) -> float:
    """Approximate horizontal gap between two footprints (0 if they overlap)."""
    d = np.linalg.norm(a.centre[:2] - b.centre[:2])
    return max(0.0, d - (a.size[:2].max() + b.size[:2].max()) / 4)


def support_and_near(centre: np.ndarray, bottom: float, boxes: list[Box3D], exclude: Box3D | None = None,
                     near_m: float = 0.6) -> tuple[str, str]:
    """What a thing at `centre` with this `bottom` height rests on, and the nearest other object."""
    on = ""
    for b in sorted(boxes, key=lambda b: -(b.centre[2] + b.size[2] / 2)):
        if b is exclude:
            continue
        top = b.centre[2] + b.size[2] / 2
        if -0.15 <= bottom - top <= 0.08 and _footprint_contains(b, centre[:2]):
            on = b.label
            break
    if not on and bottom < 0.10:
        on = "floor"
    probe = Box3D("", 0, centre, np.array([0.3, 0.3, 0.3]), 0.0)
    cands = [(b, _gap(probe, b)) for b in boxes if b is not exclude and b.label != on and np.linalg.norm(b.centre[:2] - centre[:2]) > 0.2]
    cands = [x for x in cands if x[1] <= near_m]
    near = min(cands, key=lambda x: x[1])[0].label if cands else ""
    return on, near


def scene_graph(boxes: list[Box3D]) -> dict:
    objs, rels = [], []
    for i, b in enumerate(boxes):
        objs.append({"id": f"o{i}", **b.to_dict()})
    for i, b in enumerate(boxes):
        on, near = support_and_near(b.centre, b.centre[2] - b.size[2] / 2, boxes, exclude=b)
        if on and on != "floor":
            j = next(k for k, x in enumerate(boxes) if x.label == on and x is not b)
            rels.append({"subject": f"o{i}", "predicate": "on", "object": f"o{j}"})
        elif on == "floor":
            rels.append({"subject": f"o{i}", "predicate": "on", "object": "floor"})
        if near:
            j = next(k for k, x in enumerate(boxes) if x.label == near and x is not b)
            rels.append({"subject": f"o{i}", "predicate": "near", "object": f"o{j}"})
    return {"frame_id": "map", "convention": "metres, z up, floor z=0, yaw about +z", "objects": objs, "relations": rels}


def marker_array(boxes: list[Box3D], ns: str, colour=(0.62, 0.65, 0.68)) -> dict:
    """visualization_msgs/MarkerArray as rosbridge-style JSON: one CUBE and one label per object."""
    ms = []
    for i, b in enumerate(boxes):
        q = {"x": 0.0, "y": 0.0, "z": float(np.sin(b.yaw / 2)), "w": float(np.cos(b.yaw / 2))}
        pos = {"x": float(b.centre[0]), "y": float(b.centre[1]), "z": float(b.centre[2])}
        ms.append({"header": {"frame_id": "map"}, "ns": ns, "id": 2 * i, "type": 1, "action": 0,
                   "pose": {"position": pos, "orientation": q},
                   "scale": {"x": float(b.size[0]), "y": float(b.size[1]), "z": float(b.size[2])},
                   "color": {"r": colour[0], "g": colour[1], "b": colour[2], "a": 0.35}})
        ms.append({"header": {"frame_id": "map"}, "ns": ns + "_labels", "id": 2 * i + 1, "type": 9, "action": 0,
                   "pose": {"position": {**pos, "z": pos["z"] + float(b.size[2]) / 2 + 0.1}, "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}},
                   "scale": {"x": 0, "y": 0, "z": 0.12}, "color": {"r": 1, "g": 1, "b": 1, "a": 1}, "text": b.label})
    return {"markers": ms}
