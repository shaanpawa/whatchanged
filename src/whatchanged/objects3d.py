"""Labelled oriented 3D boxes for every object, on a laptop CPU.

An open-vocabulary 2D detector (OWLv2 or Grounding DINO, both Apache-2.0) names things in
a few keyframes. Each 2D box is lifted to 3D with the session's own depth, intrinsics and
pose: keep the dominant depth layer inside the box, back-project it, drop support surfaces.
Detections of the same thing from different frames are fused when each one's 3D points
re-project inside the other's 2D boxes; the fused points are carved by those boxes' viewing
frusta, and each object gets a gravity-aligned oriented box (yaw from a min-area footprint, z from
the point heights) in the levelled z-up metric world. Output also in SpatialLM's
`bbox_i=Bbox(class,x,y,z,angle_z,scale_x,scale_y,scale_z)` text format.

With an Anthropic key, naming.py adds the room's own nouns to the detector's queries and has
Claude verify and rename the finished boxes; geometry still makes and places every box.
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
    # provenance of the name (naming.py fills these when Claude checked the box)
    owl_label: str = ""                         # the detector's label, kept when Claude renames
    label_source: str = "owlv2"                 # "owlv2" or "claude"
    verified: bool | None = None                # Claude: True keep, False drop (part, wall, several things)
    colour: str = ""
    part_of: str = ""
    fit: str = ""                               # good, too_big, too_small
    model: str = ""
    crop: str = ""                              # the first view Claude saw, relative to the run folder
    note: str = ""
    merged_from: list = field(default_factory=list)
    # one shared object set for both videos (boxes_shared): who saw it, and which states show it
    id: str = ""
    seen_by: dict = field(default_factory=dict)  # session -> its own keyframe indices that detected the object
    state: str = ""                             # "both", a session name (only there), or "moved"
    change: str = ""                            # the change id it follows (removed / added / moved)
    positions: dict = field(default_factory=dict)   # moved: session -> {"centre_m", "size_m", "yaw_deg"}
    grown: dict = field(default_factory=dict)   # the box was widened to its 2D detections (grow_box)
    dets: list = field(default_factory=list)    # member 2D detections (not saved)

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
        d = {"label": self.label, "score": round(self.score, 3),
             "centre_m": [round(float(v), 3) for v in self.centre],
             "size_m": [round(float(v), 3) for v in self.size],
             "yaw_deg": round(float(np.degrees(self.yaw)), 1),
             "frames": [int(f) for f in self.frames],
             "owl_label": self.owl_label or self.label, "label_source": self.label_source, "verified": self.verified}
        for k in ("colour", "part_of", "fit", "model", "crop", "note", "merged_from"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        for k in ("id", "seen_by", "state", "change", "positions", "grown"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.id:
            d = {"id": self.id, **d}
        return d

    def spatiallm(self, i: int) -> str:
        x, y, z = self.centre
        return (f"bbox_{i}=Bbox({self.label.replace(' ', '_')},{x:.2f},{y:.2f},{z:.2f},"
                f"{self.yaw:.4f},{self.size[0]:.2f},{self.size[1]:.2f},{self.size[2]:.2f})")


# ---------------------------------------------------------------- 2D detection

_DET = {}


def pick_device() -> str:
    """Apple's GPU when there is one (OWLv2: 0.31 s/frame on MPS, 0.69 on the M4 Pro's CPU), else the CPU."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


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


def _nested(d: Det2D, others: list[Det2D], inside: float = 0.7, rel_area: float = 0.5,
            support: set = FLAT_OR_SUPPORT) -> list[np.ndarray]:
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
        elif d.label in support and o.label not in support and ao < a and inter > 0:
            out.append(o.box)                  # anything standing on a support/flat thing
    return out


def lift(S: Session, d: Det2D, conf_thr: float, shrink: float = 0.12, stride: int = 2,
         bin_log: float = 0.03, exclude: list[np.ndarray] = (), support: set = FLAT_OR_SUPPORT) -> np.ndarray | None:
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
    w = np.ones(len(z)) if d.label in support else \
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


def _strip_supports(pts: np.ndarray, label: str, supports: np.ndarray, tol: float = 0.025,
                    support: set = FLAT_OR_SUPPORT, flat: set = FLAT) -> np.ndarray:
    keep = pts[:, 2] > 0.03 if label not in support else np.ones(len(pts), bool)
    if label in flat and keep.any():           # flat things are thin: drop what stands on them
        keep &= pts[:, 2] < np.percentile(pts[keep, 2], 30) + 0.05
    if label not in support and len(supports):
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


def _roles(roles: dict | None) -> tuple[set, set]:
    """Support and flat label sets, with the room vocabulary's extra surfaces and flat things."""
    support = FLAT_OR_SUPPORT | {n for n, k in (roles or {}).items() if k in ("surface", "flat")}
    flat = FLAT | {n for n, k in (roles or {}).items() if k == "flat"}
    return support, flat


def _fuse(S: Session, dets: list[Det2D], conf_thr: float, support: set, flat: set, sup: np.ndarray,
          min_frames: int = 2, solo_score: float = 0.5, kind: str = "owlv2") -> tuple[list[Box3D], dict]:
    """Lift every 2D box to 3D, fuse the ones that are one object, carve, fit, and drop duplicates.

    Each box keeps its member detections in `dets`.
    """
    lifted = []
    for d in dets:
        p = lift(S, d, conf_thr, exclude=_nested(d, dets, support=support), support=support)
        if p is None:
            continue
        p = _strip_supports(p, d.label, sup, support=support, flat=flat)
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
        boxes.append(Box3D(label, top, c, s, yaw, frames_seen, dict(t["votes"]), P, owl_label=label, label_source=kind,
                           dets=list(t["dets"])))

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
            if k.label in support or b.label in support:
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
    return kept, {"lifted_3d": len(lifted), "tracks": len(tracks), "boxes": len(kept)}


def boxes_for_session(S: Session, n_frames: int = 8, kind: str = "owlv2", device: str = "cpu",
                      vocab: list[str] = VOCAB3D, min_frames: int = 2, solo_score: float = 0.5, log=print,
                      roles: dict | None = None) -> tuple[list[Box3D], dict]:
    """Labelled 3D boxes for one session. `roles` (naming.room_vocabulary) marks extra nouns as surfaces or flat things."""
    t0 = time.perf_counter()
    support, flat = _roles(roles)
    frames = sorted(set(np.linspace(0, len(S.depth) - 1, min(n_frames, len(S.depth))).round().astype(int).tolist()))
    dets, stats = detect(S, frames, vocab, kind, device, log=log)
    conf_thr = float(np.percentile(S.conf[S.mask], 30))
    sup = support_heights(S)
    kept, fused = _fuse(S, dets, conf_thr, support, flat, sup, min_frames, solo_score, kind)
    for b in kept:
        b.dets = []
    stats.update({**fused, "support_heights_m": [round(float(h), 2) for h in sup],
                  "total_s": round(time.perf_counter() - t0, 2)})
    log(f"boxes: {len(kept)} labelled 3D boxes ({fused['lifted_3d']} lifted from {len(dets)} 2D) in {stats['total_s']} s")
    return kept, stats



# ---------------------------------------------------------------- one object set for both videos

def combine(sessions: dict[str, Session]) -> tuple[Session, dict[str, tuple[int, int]]]:
    """Sessions that share one world frame (COLMAP poses) as one session, frames in order.
    Returns it and each session's (start, stop) frame range in it."""
    shapes = {S.depth.shape[1:] for S in sessions.values()}
    if len(shapes) != 1:
        raise ValueError(f"sessions have different image sizes {shapes}")
    spans, k = {}, 0
    for n, S in sessions.items():
        spans[n] = (k, k + len(S.depth))
        k += len(S.depth)

    def cat(a):
        return np.concatenate([getattr(S, a) for S in sessions.values()])
    C = Session("+".join(sessions), cat("images"), cat("depth"), cat("conf"), cat("mask"), cat("K"), cat("poses"),
                cat("times"), [p for S in sessions.values() for p in S.paths])
    return C, spans


def room_yaw(S: Session, stride: int = 4) -> float:
    """Direction of the room's walls, modulo 90 degrees (radians, [0, pi/2)): the turn at which the point
    footprint's histograms along both axes are sharpest (walls, wardrobes and bed sides line up)."""
    p, _ = S.points(conf_pct=0, stride=stride)
    p = p[(p[:, 2] > 0.1) & (p[:, 2] < 2.4)]
    if len(p) < 100:
        return 0.0
    if len(p) > 150_000:
        p = p[np.random.default_rng(0).choice(len(p), 150_000, replace=False)]
    xy = p[:, :2] - np.median(p[:, :2], 0)
    best = (-1.0, 0.0)
    for deg in np.arange(0.0, 90.0, 1.0):
        t = np.radians(deg)
        r = xy @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        score = sum(float((np.histogram(r[:, a], bins=np.arange(r[:, a].min(), r[:, a].max() + 0.04, 0.02))[0]
                           .astype(float) ** 2).sum()) for a in (0, 1))
        if score > best[0]:
            best = (score, t)
    return best[1]


def fit_box_at(pts: np.ndarray, yaw: float, floor_snap: float = 0.08) -> tuple[np.ndarray, np.ndarray, float]:
    """fit_box with the yaw given (e.g. snapped to the room's walls): extents from point percentiles."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, s], [-s, c]])                           # world -> box
    r = pts[:, :2] @ R.T
    lo, hi = np.percentile(r, 2, 0), np.percentile(r, 98, 0)
    ext, mid = hi - lo, ((lo + hi) / 2) @ R
    if ext[1] > ext[0]:                                       # box-x is the long side
        yaw, ext = yaw + np.pi / 2, ext[::-1]
    zlo, zhi = np.percentile(pts[:, 2], 1), np.percentile(pts[:, 2], 99)
    if zlo < floor_snap:
        zlo = 0.0
    yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
    return np.array([mid[0], mid[1], (zlo + zhi) / 2]), np.array([ext[0], ext[1], zhi - zlo]), float(yaw)


def snap_yaw(b: Box3D, ryaw: float, tol_deg: float = 12.0, min_side: float = 0.3) -> bool:
    """Turn furniture-sized boxes within tol_deg of the room's walls to line up with them (refit from the points)."""
    if b.points is None or len(b.points) < 20 or b.size[:2].max() < min_side:
        return False
    d = (b.yaw - ryaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if abs(d) > np.radians(tol_deg) or abs(d) < np.radians(0.5):
        return False
    b.centre, b.size, b.yaw = fit_box_at(b.points, b.yaw - d)
    return True


def _rect_area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def coverage(S: Session, b: Box3D, dets: list[Det2D]) -> tuple[float, float, int]:
    """Over the member views: median share of each 2D detection that the box's projection covers (recall),
    median share of the projection inside the detection (precision), and the number of views used."""
    H, W = S.depth.shape[1:]
    corners = b.corners()
    rec, prec = [], []
    for d in dets:
        uv, z = project(corners, S.poses[d.frame], S.K[d.frame])
        if (z <= 0.05).any():
            continue
        r = np.clip([*uv.min(0), *uv.max(0)], 0, [W, H, W, H])
        db = np.clip(d.box, 0, [W, H, W, H])
        inter = _rect_area([max(r[0], db[0]), max(r[1], db[1]), min(r[2], db[2]), min(r[3], db[3])])
        if _rect_area(r) <= 0 or _rect_area(db) <= 0:
            continue
        rec.append(inter / _rect_area(db))
        prec.append(inter / _rect_area(r))
    if not rec:
        return float("nan"), float("nan"), 0
    return float(np.median(rec)), float(np.median(prec)), len(rec)


def _wall_behind(S: Session, d: Det2D, pts: np.ndarray, ring: tuple[int, int] = (4, 14),
                 iters: int = 150) -> tuple[np.ndarray, float] | None:
    """The vertical surface the object stands against in this view (a wall), from the depth in a band around
    its 2D box: (unit normal facing the camera, offset), or None. Only a plane that has the object's
    points in front of it and shows on at least two sides of the box counts."""
    H, W = S.depth.shape[1:]
    x0, y0, x1, y1 = d.box
    a, b = ring
    bands = [(x0 - b, x0 - a, y0, y1), (x1 + a, x1 + b, y0, y1), (x0, x1, y0 - b, y0 - a), (x0, x1, y1 + a, y1 + b)]
    K, P = S.K[d.frame], S.poses[d.frame]
    pts3, side = [], []
    for k, (u0, u1, v0, v1) in enumerate(bands):
        u0, u1, v0, v1 = int(max(0, u0)), int(min(W, u1)), int(max(0, v0)), int(min(H, v1))
        if u1 - u0 < 2 or v1 - v0 < 2:
            continue
        vv, uu = np.mgrid[v0:v1:2, u0:u1:2]
        vv, uu = vv.ravel(), uu.ravel()
        z = S.depth[d.frame, vv, uu]
        ok = S.mask[d.frame, vv, uu] & (z > 0.1)
        if ok.sum() < 8:
            continue
        uu, vv, z = uu[ok], vv[ok], z[ok]
        pc = np.stack([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z], 1) @ P[:3, :3].T + P[:3, 3]
        keep = pc[:, 2] > 0.05                                 # not the floor
        pts3.append(pc[keep])
        side.append(np.full(keep.sum(), k))
    if not pts3:
        return None
    X, side = np.concatenate(pts3), np.concatenate(side)
    if len(X) < 30:
        return None
    rng = np.random.default_rng(d.frame)
    best, best_in = None, None
    for _ in range(iters):
        p = X[rng.choice(len(X), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[2]) > 0.25:                                   # walls are vertical
            continue
        inl = np.abs((X - p[0]) @ n) < 0.025
        if best_in is None or inl.sum() > best_in.sum():
            best, best_in = (n, -float(n @ p[0])), inl
    if best is None or best_in.sum() < max(30, 0.4 * len(X)):
        return None
    if sum(np.sum(best_in & (side == k)) >= 8 for k in range(4)) < 2:
        return None
    n, off = best
    if n @ P[:3, 3] + off < 0:                                 # normal towards the camera
        n, off = -n, -off
    if len(pts) and np.mean(pts @ n + off > -0.03) < 0.95:     # the object must stand in front of it
        return None
    return n, off


def mean_iou(S: Session, corners: np.ndarray, dets: list[Det2D]) -> np.ndarray:
    """For N candidate boxes (corners (N, 8, 3)): the mean over the detections of the IoU between the box's
    projected outline (clipped to the image) and the 2D detection; 0 for a view the box is behind."""
    H, W = S.depth.shape[1:]
    total = np.zeros(len(corners))
    for d in dets:
        P, K = S.poses[d.frame], S.K[d.frame]
        pc = (corners - P[:3, 3]) @ P[:3, :3]
        z = pc[..., 2]
        ok = (z > 0.05).all(1)
        zs = np.where(z > 0.05, z, 1.0)
        u, v = K[0, 0] * pc[..., 0] / zs + K[0, 2], K[1, 1] * pc[..., 1] / zs + K[1, 2]
        r = np.stack([np.clip(u.min(1), 0, W), np.clip(v.min(1), 0, H), np.clip(u.max(1), 0, W), np.clip(v.max(1), 0, H)], 1)
        db = np.clip(d.box, 0, [W, H, W, H])
        iw = np.clip(np.minimum(r[:, 2], db[2]) - np.maximum(r[:, 0], db[0]), 0, None)
        ih = np.clip(np.minimum(r[:, 3], db[3]) - np.maximum(r[:, 1], db[1]), 0, None)
        inter = iw * ih
        union = (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1]) + (db[2] - db[0]) * (db[3] - db[1]) - inter
        total += np.where(ok, inter / np.maximum(union, 1e-9), 0.0)
    return total / max(len(dets), 1)


_SGN = np.array([[sx, sy, sz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)], float)


def _corners_lohi(lo: np.ndarray, hi: np.ndarray, c: np.ndarray, yaw: float) -> np.ndarray:
    """Corners (N, 8, 3) of boxes given by their extents in a frame turned by yaw about c (xy) and world z."""
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    pts = mid[:, None] + _SGN[None] * half[:, None]                    # box frame
    xy = pts[..., :2] @ R2.T + c[:2]
    return np.concatenate([xy, pts[..., 2:]], -1)


def _descend(S: Session, dets: list[Det2D], origin: np.ndarray, yaw: float, faces: np.ndarray,
             end_a: np.ndarray, end_b: np.ndarray, steps: int = 9, sweeps: int = 2) -> tuple[np.ndarray, float, float]:
    """Box faces (lo xyz, hi xyz in the yaw frame about origin) moved one at a time, each within [end_a, end_b],
    to where the mean IoU of the projected box with the detections is highest. Returns faces, IoU before, after."""
    def score(f):
        f = np.atleast_2d(f)
        ok = (f[:, 3:] - f[:, :3] > 0.01).all(1)
        return np.where(ok, mean_iou(S, _corners_lohi(f[:, :3], f[:, 3:], origin, yaw), dets), -1.0)
    iou0 = float(score(faces)[0])
    for _ in range(sweeps):
        for i in range(6):
            cand = np.repeat(faces[None], steps, 0)
            cand[:, i] = np.linspace(end_a[i], end_b[i], steps)
            faces = cand[int(np.argmax(score(cand)))]
    return faces, iou0, float(score(faces)[0])


def shrink_box(S: Session, b: Box3D, why: dict | None = None) -> bool:
    """A box Claude called too big comes in face by face, as far as raises the mean IoU of its projection
    with its 2D detections, but never into the middle 60% of its own points. True if it changed."""
    why = {} if why is None else why
    if not b.dets or b.points is None or len(b.points) < 10:
        return False
    c, size, yaw = b.centre.copy(), b.size.copy(), b.yaw
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    Pb = np.c_[(b.points[:, :2] - c[:2]) @ R2, b.points[:, 2]]
    faces = np.r_[-size[:2] / 2, c[2] - size[2] / 2, size[:2] / 2, c[2] + size[2] / 2]
    core = np.r_[np.percentile(Pb, 20, 0), np.percentile(Pb, 80, 0)]
    inner = np.r_[np.clip(core[:3], faces[:3], faces[3:]), np.clip(core[3:], faces[:3], faces[3:])]
    new, iou0, iou1 = _descend(S, b.dets, np.r_[c[:2], 0.0], yaw, faces, faces, inner)
    why.update(iou=[round(iou0, 2), round(iou1, 2)])
    if iou1 < iou0 + 0.03:
        why["no"] = "no better against the detections"
        return False
    mid, new_size = (new[:3] + new[3:]) / 2, new[3:] - new[:3]
    b.grown = {"mode": "shrunk", "from_centre_m": [round(float(x), 3) for x in c],
               "from_size_m": [round(float(x), 3) for x in size], "iou": [round(iou0, 2), round(iou1, 2)]}
    b.centre, b.size = np.r_[mid[:2] @ R2.T + c[:2], mid[2]], new_size
    return True


def place_by_2d(S: Session, b: Box3D, reach: float = 0.4, floor: float = 0.0) -> tuple[Box3D, float, float] | None:
    """The same box (yaw kept, size scaled 0.8-1.3x, a thin one maybe deeper) moved to where its projections best match its 2D detections
    (mean IoU of the projected box's outline with each detection), searched within `reach` metres. For things
    whose depth points are not on them: a mesh bin's depth is the wall seen through it. Returns the box and its
    mean IoU before and after, or None."""
    H, W = S.depth.shape[1:]
    if not b.dets:
        return None
    R = np.array([[np.cos(b.yaw), -np.sin(b.yaw), 0], [np.sin(b.yaw), np.cos(b.yaw), 0], [0, 0, 1]])
    t = np.arange(-reach, reach + 1e-9, 0.02)
    tz = np.arange(-0.12, 0.121, 0.03)
    sc = np.array([0.8, 0.9, 1.0, 1.15, 1.3])
    asp = np.array([1.0, 1.5, 2.0] if b.size[1] < 0.6 * b.size[0] else [1.0])   # one-sided points: maybe deeper
    dx, dy, dz, k, ka = (a.ravel() for a in np.meshgrid(t, t, tz, sc, asp, indexing="ij"))
    sizes = b.size[None] * k[:, None] * np.stack([np.ones_like(ka), ka, np.ones_like(ka)], 1)
    centres = b.centre[None] + np.stack([dx, dy, dz], 1) @ R.T
    centres[:, 2] = np.maximum(centres[:, 2], floor + sizes[:, 2] / 2)        # not through the floor
    corners = centres[:, None] + (_SGN[None] * sizes[:, None] / 2) @ R.T        # (N, 8, 3)
    iou = mean_iou(S, corners, b.dets)
    i0 = int(np.argmin(np.abs(dx) + np.abs(dy) + np.abs(dz) + np.abs(k - 1) + np.abs(ka - 1)))
    j = int(np.argmax(iou))
    out = Box3D(b.label, b.score, centres[j], sizes[j], b.yaw)
    return out, float(iou[i0]), float(iou[j])


def _dmin(S: Session, f: int, cache: dict) -> np.ndarray:
    """Depth with the nearest surface in a 5 px neighbourhood (unknown = inf): free space is only claimed
    where no nearby pixel shows something at or before that depth."""
    if f not in cache:
        from scipy.ndimage import minimum_filter
        d = np.where(S.mask[f] & (S.depth[f] > 0), S.depth[f], np.inf).astype(np.float32)
        cache[f] = minimum_filter(d, size=5, mode="nearest")
    return cache[f]


def grow_box(S: Session, b: Box3D, free_frames: list[int], supports=(), max_grow: float = 1.0, force: bool = False,
             cache: dict | None = None, vox_budget: int = 100_000, min_recall: float = 0.85, min_precision: float = 0.5,
             allow_place: bool = True, why: dict | None = None) -> bool:
    """Widen a box whose fused points cover too little of its own 2D detections (white radiators, chairs,
    wardrobes: depth is sparse on plain or thin things), keeping its yaw. True if the box changed.

    The most it may take is the visual hull of the member 2D boxes: voxels that project inside the detection
    in at least 75% of the member views that see them, minus space the depth maps show as empty (free_frames:
    views of the videos that saw the object), minus space further behind the visible surface than the object
    is big in 2D, minus anything behind the wall it stands against, and above its floor or support; at most
    max_grow beyond each face, and no side by more than the box's own size along it (15 cm at least). Each
    face then moves out only as far as raises the mean IoU of the projected box with the detections, and the
    result is kept only if that IoU rises by 0.03 or more without losing coverage of the detections. So a
    box grows to its 2D evidence and no further.

    A box that sits mostly OUTSIDE its own detections (precision < min_precision: a mesh bin, whose depth is
    the wall seen through it) is moved instead: place_by_2d (not with allow_place=False: a box merged from
    pieces, each detection showing only part of the thing).
    """
    why = {} if why is None else why
    if not b.dets or b.points is None or len(b.points) < 10:
        why["no"] = "no detections or points"
        return False
    cache = {} if cache is None else cache
    H, W = S.depth.shape[1:]
    rec0, prec0, nv = coverage(S, b, b.dets)
    relocate = allow_place and prec0 < min_precision
    why.update(recall=rec0, precision=prec0, views=nv, mode="place" if relocate else "grow")
    if nv == 0 or (not force and not relocate and rec0 >= min_recall):
        why["no"] = "covers its detections"
        return False
    if relocate:
        pz = float(np.percentile(b.points[:, 2], 2))
        floor = max([float(h) for h in supports if h <= pz + 0.04] + [0.0])
        placed = place_by_2d(S, b, floor=floor)
        if placed is None:
            why["no"] = "no placement"
            return False
        cand, iou0, iou1 = placed
        rec1, prec1, _ = coverage(S, cand, b.dets)
        why.update(iou=[round(iou0, 2), round(iou1, 2)], new_size=[round(float(x), 2) for x in cand.size],
                   recall1=rec1, precision1=prec1)
        if not (iou1 >= iou0 + 0.1 and rec1 >= 0.7 and prec1 >= 0.6):
            why["no"] = "no better placement"
            return False
        b.grown = {"mode": "placed", "from_centre_m": [round(float(x), 3) for x in b.centre],
                   "from_size_m": [round(float(x), 3) for x in b.size], "iou": [round(iou0, 2), round(iou1, 2)],
                   "recall": [round(rec0, 2), round(rec1, 2)], "precision": [round(prec0, 2), round(prec1, 2)], "views": int(nv)}
        b.centre, b.size = cand.centre, cand.size
        return True
    c, size, yaw = b.centre.copy(), b.size.copy(), b.yaw
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])     # box -> world
    P = b.points
    Psub = P if len(P) <= 500 else P[np.random.default_rng(0).choice(len(P), 500, replace=False)]
    g = float(np.clip(1.5 * size.max(), 0.3, max_grow))       # sideways; upwards always up to max_grow
    pz = float(np.percentile(P[:, 2], 2))
    floor = max([float(h) for h in supports if h <= pz + 0.04] + [0.0])
    lo = np.array([-size[0] / 2 - g, -size[1] / 2 - g, floor])
    hi = np.array([size[0] / 2 + g, size[1] / 2 + g, min(2.6, c[2] + size[2] / 2 + max_grow)])
    if hi[2] - lo[2] < 0.05:
        why["no"] = "no room above the support"
        return False
    res = max(0.02, float(np.prod(hi - lo) / vox_budget) ** (1 / 3))
    axes = [np.arange(lo[i] + res / 2, hi[i], res) for i in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    shape = grid[0].shape
    Xb = np.stack([a.ravel() for a in grid], 1)
    Xw = np.c_[Xb[:, :2] @ R2.T + c[:2], Xb[:, 2]]
    n = len(Xw)

    def pix(f):
        uv, z = project(Xw, S.poses[f], S.K[f])
        u, v = uv[:, 0], uv[:, 1]
        inimg = (z > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        ui = np.clip(np.nan_to_num(u, nan=0).astype(int), 0, W - 1)
        vi = np.clip(np.nan_to_num(v, nan=0).astype(int), 0, H - 1)
        return u, v, z, inimg, ui, vi

    by_frame: dict[int, list[Det2D]] = {}
    for d in b.dets:
        by_frame.setdefault(d.frame, []).append(d)
    # How deep behind the visible surface the object may reach: about its size as the 2D boxes show it, in
    # metres at its depth (boxes cut off by the image edge left out of that estimate).
    sizes = []
    for f, ds in by_frame.items():
        _, zp = project(Psub, S.poses[f], S.K[f])
        zc = float(np.median(zp[zp > 0.05])) if (zp > 0.05).any() else 0.0
        for d in ds:
            x0, y0, x1, y1 = d.box
            if zc > 0 and x0 > 2 and y0 > 2 and x1 < W - 3 and y1 < H - 3:
                sizes.append(max((x1 - x0) * zc / S.K[f][0, 0], (y1 - y0) * zc / S.K[f][1, 1]))
    T = float(np.clip(np.median(sizes) if sizes else size.max(), 0.2, 1.2))
    seen, inside = np.zeros(n, np.int32), np.zeros(n, np.int32)
    deep, dvalid = np.zeros(n, np.int32), np.zeros(n, np.int32)
    behind = np.zeros(n, bool)
    walls = 0
    for f, ds in by_frame.items():
        u, v, z, inimg, ui, vi = pix(f)
        inb = np.zeros(n, bool)
        for d in ds:
            x0, y0, x1, y1 = d.box
            mx, my = 0.04 * (x1 - x0) + 2, 0.04 * (y1 - y0) + 2
            inb |= inimg & (u >= x0 - mx) & (u <= x1 + mx) & (v >= y0 - my) & (v <= y1 + my)
        seen += inimg
        inside += inb
        obs = S.depth[f, vi, ui]
        valid = inb & S.mask[f, vi, ui] & (obs > 0)
        dvalid += valid
        deep += valid & (z > obs + T)
        for d in ds:
            wall = _wall_behind(S, d, Psub)
            if wall is not None:
                behind |= Xw @ wall[0] + wall[1] < -0.04
                walls += 1
    free, observed = np.zeros(n, np.int32), np.zeros(n, np.int32)
    for f in free_frames:
        u, v, z, inimg, ui, vi = pix(f)
        if not inimg.any():
            continue
        okd = inimg & S.mask[f, vi, ui]
        observed += okd
        free += okd & (_dmin(S, f, cache)[vi, ui] > z + 0.04 + 0.03 * z)
    need = min(2, len(by_frame))
    hull = (seen >= need) & (inside >= 0.75 * seen) & ~behind
    hull &= ~((free >= 2) & (free >= 0.5 * observed))       # seen through in most views that observe it
    hull &= ~((deep >= 1) & (deep >= 0.5 * np.maximum(dvalid, 1)))
    why.update(thickness=round(T, 2), seen=int((seen >= need).sum()), inbox=int(((seen >= need) & (inside >= 0.75 * seen)).sum()),
               behind=int(behind.sum()), free=int(((free >= 2) & (free >= 0.5 * observed)).sum()),
               deep=int(((deep >= 1) & (deep >= 0.5 * np.maximum(dvalid, 1))).sum()), hull=int(hull.sum()), res=round(res, 3))
    if hull.sum() < 8:
        why["no"] = "empty hull"
        return False
    # the part of the hull that holds the object's own points (thin legs included: the faces are chosen by
    # the 2D detections below, so a stray spur of the hull does not stretch the box)
    from scipy import ndimage
    lab, nlab = ndimage.label(hull.reshape(shape), structure=np.ones((3, 3, 3)))
    Pb = np.c_[(P[:, :2] - c[:2]) @ R2, P[:, 2]]
    idx = np.floor((Pb - lo) / res).astype(int)
    ok = ((idx >= 0) & (idx < np.array(shape))).all(1)
    occ = np.zeros(shape, bool)
    occ[tuple(idx[ok].T)] = True
    occ = ndimage.binary_dilation(occ, iterations=2)
    touched = np.unique(lab[occ & (lab > 0)])
    if not len(touched):
        why["no"] = "hull misses the points"
        return False
    Hb = Xb[np.isin(lab, touched).ravel()]
    if "hull_pts" in why:                      # debugging: the kept hull in world coordinates
        why["hull_pts"] = np.c_[Hb[:, :2] @ R2.T + c[:2], Hb[:, 2]]
    if len(Hb) < 8:
        return False
    # The hull's extent is the most the box may take. Each face then goes as far towards it as the 2D
    # detections want: the mean IoU of the projected box with them, face by face, two sweeps.
    lo_o = np.r_[-size[:2] / 2, c[2] - size[2] / 2]
    hi_o = np.r_[size[:2] / 2, c[2] + size[2] / 2]
    lo_h = np.minimum(np.percentile(Hb, 2, 0) - res / 2, lo_o)
    hi_h = np.maximum(np.percentile(Hb, 98, 0) + res / 2, hi_o)
    lo_h[2] = max(lo_h[2], floor)
    if floor == 0.0 and lo_h[2] < 0.08:
        lo_h[2] = 0.0
    # Each side grows at most by the box's own size along it (15 cm at least), the top by twice its height:
    # a thin slab seen at a slant (an open door) would otherwise match its slanted 2D boxes by turning into a crate.
    side = np.maximum(size[:2], 0.15)
    lo_h[:2] = np.maximum(lo_h[:2], lo_o[:2] - side)
    hi_h = np.minimum(hi_h, hi_o + np.r_[side, max(2 * size[2], 0.5)])
    faces = np.r_[lo_o, hi_o]
    # faces only move out, except the bottom, which may come up to where the points start (a radiator on legs
    # was snapped to the floor)
    inner = faces.copy()
    inner[2] = max(faces[2], min(float(np.percentile(P[:, 2], 2)), faces[2] + 0.15 * size[2]))
    faces, iou0, iou1 = _descend(S, b.dets, np.r_[c[:2], 0.0], yaw, faces, inner, np.r_[lo_h, hi_h])
    why.update(iou=[round(iou0, 2), round(iou1, 2)], hull_size=[round(float(x), 2) for x in hi_h - lo_h])
    lo_n, hi_n = faces[:3], faces[3:]
    new_size = hi_n - lo_n
    mid = (lo_n + hi_n) / 2
    new_c = np.r_[mid[:2] @ R2.T + c[:2], mid[2]]
    new_yaw = yaw
    if new_size[1] > new_size[0]:
        new_size, new_yaw = new_size[[1, 0, 2]], (yaw + np.pi / 2 + np.pi / 2) % np.pi - np.pi / 2
    cand = Box3D(b.label, b.score, new_c, new_size, float(new_yaw))
    rec1, prec1, _ = coverage(S, cand, b.dets)
    why.update(new_size=[round(float(x), 2) for x in new_size], recall1=rec1, precision1=prec1)
    if not (iou1 >= iou0 + 0.03 and rec1 >= rec0 - 0.02):
        why["no"] = "no better against the detections"
        return False
    b.grown = {"mode": "grow", "from_centre_m": [round(float(x), 3) for x in c], "iou": [round(iou0, 2), round(iou1, 2)],
               "from_size_m": [round(float(x), 3) for x in size], "recall": [round(rec0, 2), round(rec1, 2)],
               "precision": [round(prec0, 2), round(prec1, 2)], "views": int(nv), "walls": int(walls)}
    b.centre, b.size, b.yaw = new_c, new_size, float(new_yaw)
    return True


def attach_dets(S: Session, boxes: list[Box3D], dets: list[Det2D], min_agree: float = 0.6) -> int:
    """Detections that joined no box (no usable depth inside them, or a lone view) join the box of the same
    label whose points land inside them, for the visual hull. Skipped when another box of that label would
    fit them too (one 2D box over two radiators). Returns how many joined."""
    used = {id(d) for b in boxes for d in b.dets}
    rng = np.random.default_rng(0)
    subs = [None if b.points is None else b.points if len(b.points) <= 300 else b.points[rng.choice(len(b.points), 300, replace=False)]
            for b in boxes]
    n = 0
    for d in dets:
        if id(d) in used:
            continue
        fits = []
        for b, sub in zip(boxes, subs):
            if sub is None or d.label not in b.votes:
                continue
            agree, seen = _box_agreement(S, sub, [d], margin=0.1)
            vis = seen > 0
            if vis.mean() >= 0.5 and agree[vis].mean() > 0.3:
                fits.append((agree[vis].mean(), b))
        if len(fits) != 1 or fits[0][0] < min_agree:
            continue
        b = fits[0][1]
        if any(m.frame == d.frame for m in b.dets):
            continue
        b.dets.append(d)
        b.frames = sorted(set(b.frames) | {d.frame})
        b.votes[d.label] = b.votes.get(d.label, 0) + d.score
        n += 1
    return n


def free_frames(spans: dict[str, tuple[int, int]], b: Box3D, cap: int = 48, min_share: float = 0.0) -> list[int]:
    """Frames of the videos that saw b (all of them when unknown), at most `cap`, spread evenly. With
    `min_share`, only videos with at least that share of its sightings: a chair seen 6 times after and once
    before is shown from the after video, not from before, where its place may be empty."""
    seen = {n: sum(a <= f < z for f in b.frames) for n, (a, z) in spans.items()}
    total = sum(seen.values())
    own = [n for n, k in seen.items() if k and k >= min_share * total] or list(spans)
    fr = [f for n in own for f in range(*spans[n])]
    if len(fr) > cap:
        fr = [fr[i] for i in np.linspace(0, len(fr) - 1, cap).round().astype(int)]
    return fr


def set_seen_by(b: Box3D, spans: dict[str, tuple[int, int]]) -> None:
    b.seen_by = {n: sorted(int(f - a) for f in b.frames if a <= f < z) for n, (a, z) in spans.items()}
    b.seen_by = {n: v for n, v in b.seen_by.items() if v}


def boxes_shared(sessions: dict[str, Session], n_frames: int = 40, kind: str = "owlv2", device: str = "cpu",
                 vocab: list[str] = VOCAB3D, min_frames: int = 2, solo_score: float = 0.5, log=print,
                 roles: dict | None = None) -> tuple[list[Box3D], dict, dict]:
    """ONE set of labelled 3D boxes for sessions that share a world frame (COLMAP poses): detections from the
    keyframes of every video are fused together, so an object seen poorly by one video still gets the
    other's views. Each box records which video saw it (seen_by). Boxes whose points cover too little of
    their 2D detections grow to them (grow_box); furniture is lined up with the room's walls.

    Returns the boxes, stats, and the context to grow them again later ({"S": combined session, "spans",
    "supports", "room_yaw"})."""
    t0 = time.perf_counter()
    C, spans = combine(sessions)
    support, flat = _roles(roles)
    frames = []
    for a, z in spans.values():
        frames += sorted(set((a + np.linspace(0, z - a - 1, min(n_frames, z - a)).round().astype(int)).tolist()))
    dets, stats = detect(C, frames, vocab, kind, device, log=log)
    sup = support_heights(C)
    # The depth is COLMAP-anchored (unanchored pixels are masked out), so the anchor mask is the filter.
    # MapAnything's confidence sits at its floor (1.0) on white walls and radiators: a percentile cut on it
    # threw away nearly every point of the white things.
    kept, fused = _fuse(C, dets, -np.inf, support, flat, sup, min_frames, solo_score, kind)
    joined = attach_dets(C, kept, dets)
    ry = room_yaw(C)
    snapped = sum(snap_yaw(b, ry) for b in kept)
    cache, grown = {}, 0
    for b in kept:
        grown += grow_box(C, b, free_frames(spans, b), sup, cache=cache)
        set_seen_by(b, spans)
    both = sum(len(b.seen_by) > 1 for b in kept)
    stats.update({**fused, "frames_per_session": {n: min(n_frames, z - a) for n, (a, z) in spans.items()},
                  "joined_2d": joined, "room_yaw_deg": round(float(np.degrees(ry)), 1), "snapped": int(snapped),
                  "grown": int(grown), "seen_by_both": int(both), "support_heights_m": [round(float(h), 2) for h in sup],
                  "total_s": round(time.perf_counter() - t0, 2)})
    log(f"boxes: {len(kept)} labelled 3D boxes for both videos together ({both} seen by both; {fused['lifted_3d']} lifted "
        f"from {len(dets)} 2D, {joined} more 2D boxes joined); {grown} grown to their 2D boxes, {snapped} lined up "
        f"with the walls, in {stats['total_s']} s")
    return kept, stats, {"S": C, "spans": spans, "supports": sup, "room_yaw": ry, "dets": dets, "frames": frames}


def _labels_of(b: Box3D, share: float = 0.2) -> set:
    """The detector labels that make up a box (at least `share` of its summed scores), and its own names."""
    tot = sum(b.votes.values()) or 1.0
    return {k for k, v in b.votes.items() if v >= share * tot} | {b.label, b.owl_label or b.label}


def view_checks(S: Session, b: Box3D, frames: list[int], dets_by_frame: dict[int, list[Det2D]], labels: set,
                min_inside: float = 0.7, min_visible: float = 0.5) -> dict[int, bool]:
    """For each frame (of `frames`) that should show the box (in front of the camera, at least 70% of its outline in
    the image, mostly not hidden behind nearer depth): whether a 2D detection of one of `labels` lands on its
    projection (IoU >= 0.2, or 70% of the smaller of the two inside the other). Frames that cannot show it are left out."""
    from .naming import visible_share, volume_samples
    H, W = S.depth.shape[1:]
    pts, out = volume_samples(b), {}
    for f in frames:
        uv, z = project(b.corners(), S.poses[f], S.K[f])
        if (z <= 0.1).any():
            continue
        lo, hi = uv.min(0), uv.max(0)
        clo, chi = np.clip(lo, 0, [W, H]), np.clip(hi, 0, [W, H])
        if _rect_area([*clo, *chi]) < min_inside * max(_rect_area([*lo, *hi]), 1e-9):
            continue
        vis = visible_share(S, f, b, pts)
        if vis is None or vis < min_visible:
            continue
        r, hit = [*clo, *chi], False
        for d in dets_by_frame.get(f, []):
            if d.label not in labels:
                continue
            db = np.clip(d.box, 0, [W, H, W, H])
            inter = _rect_area([max(r[0], db[0]), max(r[1], db[1]), min(r[2], db[2]), min(r[3], db[3])])
            a, c = _rect_area(r), _rect_area(db)
            if inter / max(a + c - inter, 1e-9) >= 0.2 or inter / max(min(a, c), 1e-9) >= 0.7:
                hit = True
                break
        out[f] = hit
    return out


def drop_twins(S: Session, spans: dict[str, tuple[int, int]], boxes: list[Box3D], dets: list[Det2D],
               min_both: int = 2, twin_hits: float = 0.6, ghost_hits: float = 0.2, log=print) -> int:
    """A kept box whose own 2D detections are not there where it should be seen, while a kept box of the same name
    is, is a second copy of that object (a few close-up frames lifted it to the wrong place): it is dropped.

    For each pair of kept boxes with one name, over the frames of the videos that saw both that should show both
    (view_checks): the twin is detected in at least 60% of them, this box in at most 20%, and there are at least
    two. Two real objects of one name (two pillows, two plants) are detected side by side in those frames, and an
    object that is only in one video (added, removed or moved) shares no video with its namesake, so neither
    is dropped; a video "saw" a box only with at least a quarter of its sightings. Returns how many were dropped."""
    by_frame: dict[int, list[Det2D]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    kept = [b for b in boxes if b.verified is not False]
    checks: dict[tuple[int, int], dict] = {}

    def check(b, sessions, labels):
        key = (id(b), tuple(sorted(sessions)))
        if key not in checks:
            checks[key] = view_checks(S, b, [f for n in sorted(sessions) for f in range(*spans[n])], by_frame, labels)
        return checks[key]

    def videos(b):              # a stray sighting or two does not count: a moved chair's new place, seen once before
        seen = {n: sum(a <= f < z for f in b.frames) for n, (a, z) in spans.items()}
        return {n for n, k in seen.items() if k and k >= 0.25 * sum(seen.values())} or set(b.seen_by)

    dropped = 0
    for b in kept:
        for t in kept:
            if t is b or t.label != b.label or t.verified is False or b.verified is False:
                continue
            common = videos(b) & videos(t)
            if not common:
                continue
            labels = _labels_of(b) | _labels_of(t)
            cb, ct = check(b, common, labels), check(t, common, labels)
            both = [f for f in cb if f in ct]
            if len(both) < min_both:
                continue
            nb, nt = sum(cb[f] for f in both), sum(ct[f] for f in both)
            if nt >= twin_hits * len(both) and nb <= ghost_hits * len(both):
                b.verified = False
                b.note = (f"a second copy of the {t.label} at ({t.centre[0]:.2f}, {t.centre[1]:.2f}): in {len(both)} views "
                          f"that show both, the {t.label} is detected there {nt} times and here {nb}")[:160]
                dropped += 1
                log(f"boxes: dropped a second '{b.label}' at ({b.centre[0]:.2f}, {b.centre[1]:.2f}): its detections are "
                    f"not there in {len(both) - nb} of {len(both)} views that show the one at ({t.centre[0]:.2f}, {t.centre[1]:.2f})")
                break
    return dropped


# ---------------------------------------------------------------- which state shows which object

# What the detector mixes up: one of these found on the spot means the object may well still be there.
KIN = [{"bin", "basket", "bucket"}, {"fan", "heater"}, {"chair", "stool", "armchair"}, {"monitor", "television"},
       {"box", "cardboard box"}, {"cup", "mug"}, {"plant", "potted plant", "vase"}]


def kin(labels: set) -> set:
    """These labels and the ones the detector confuses them with."""
    out = set(labels) | {SYNONYMS.get(x, x) for x in labels}
    for g in KIN:
        if out & g:
            out |= g
    return out


def object_changes(S: Session, spans: dict[str, tuple[int, int]], boxes: list[Box3D], dets: list[Det2D],
                   frames: list[int], names: tuple[str, str] = ("A", "B"), min_views: int = 3, max_hits: float = 0.2,
                   min_hits: int = 2, min_share: float = 0.5, min_move: float = 0.15, log=print) -> list[tuple]:
    """Objects one video detects steadily and the other does not, although it looked straight at where they were.

    The depth test (changes.py) needs depth good to a few per cent, which a very wide lens does not give; the
    detector does not need depth at all. For each kept box and each video: over that video's detector frames
    that should show the box (view_checks: in front of the camera, its outline in the image, not hidden behind
    nearer depth), how often a detection of that kind of thing lands on it. Found in at least `min_hits` and
    `min_share` of one video's frames, and in at most `max_hits` of at least `min_views` of the other's: the
    object is not there in the other video. The other end of a move is often seen too seldom to stand alone (a
    spot only one later frame shows, or a stray detection there in the first video), so a box of the same name
    that the second video detects more often than the first (which finds it in at most twice `max_hits`) is
    taken as that end, for Claude to confirm. Returns (box, "removed" | "added", the other video's
    frames that showed the spot, hits there, its own video's frames that showed it, hits there)."""
    by_frame: dict[int, list[Det2D]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    ran = set(frames)
    kept = [b for b in boxes if b.verified is not False]
    rate = {}
    for b in kept:
        labels = kin(_labels_of(b))
        r = {n: view_checks(S, b, [f for f in range(*spans[n]) if f in ran], by_frame, labels) for n in names}
        rate[id(b)] = {n: (len(r[n]), sum(r[n].values())) for n in names}
    kind = {names[0]: "removed", names[1]: "added"}          # what it is when only this video has the object
    out = []
    for b in kept:
        for mine, other in (names, names[::-1]):
            (vm, hm), (vo, ho) = rate[id(b)][mine], rate[id(b)][other]
            if hm >= min_hits and hm >= min_share * vm and vo >= min_views and ho <= max_hits * vo:
                out.append((b, kind[mine], vo, ho, vm, hm))
    share = lambda x, n: rate[id(x)][n][1] / max(rate[id(x)][n][0], 1)      # noqa: E731
    taken = {id(c[0]) for c in out}
    for c in list(out):                                      # the thinly seen other end of a move
        b, k = c[0], c[1]
        mine = names[0] if k == "removed" else names[1]
        other = names[1] if mine == names[0] else names[0]
        if any(x[1] != k and x[0].label == b.label for x in out):
            continue
        ends = [x for x in kept if id(x) not in taken and x.label == b.label and rate[id(x)][other][1] >= 1
                and share(x, other) > share(x, mine) and share(x, mine) <= 2 * max_hits
                and np.linalg.norm(x.centre[:2] - b.centre[:2]) >= min_move]
        if ends:
            x = max(ends, key=lambda x: (share(x, other) - share(x, mine), -np.linalg.norm(x.centre[:2] - b.centre[:2])))
            (vm, hm), (vo, ho) = rate[id(x)][mine], rate[id(x)][other]
            out.append((x, kind[other], vm, hm, vo, ho))
            taken.add(id(x))
    log(f"Changes (objects): {sum(c[1] == 'removed' for c in out)} objects only before, "
        f"{sum(c[1] == 'added' for c in out)} only after, where the other video looked and found none")
    return out


def _hue_mask(rgb: np.ndarray, hue_deg: float, tol_deg: float = 15.0, min_sat: float = 0.4) -> np.ndarray:
    """Pixels of a strong colour near `hue_deg` (rgb: (..., 3) uint8)."""
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb.reshape(-1, 1, 3)), cv2.COLOR_RGB2HSV).reshape(*rgb.shape[:-1], 3)
    dh = np.abs((hsv[..., 0].astype(float) * 2 - hue_deg + 180) % 360 - 180)
    return (dh <= tol_deg) & (hsv[..., 1] >= min_sat * 255) & (hsv[..., 2] >= 50)


def colour_of(S: Session, b: Box3D, frames: list[int], min_sat: float = 0.4, min_share: float = 0.6) -> float | None:
    """The hue (degrees) of an object with one strong colour, from the middle of its projected box in the frames
    that detected it; None when it has no such colour (most of its strong pixels within 15 degrees of one hue)."""
    import cv2
    H, W = S.depth.shape[1:]
    hues = []
    for f in frames:
        uv, z = project(b.corners(), S.poses[f], S.K[f])
        if (z <= 0.1).any():
            continue
        lo, hi = np.clip(uv.min(0), 0, [W - 1, H - 1]), np.clip(uv.max(0), 0, [W - 1, H - 1])
        c, h = (lo + hi) / 2, (hi - lo) / 4                      # the middle half of the box
        x0, y0, x1, y1 = (int(round(v)) for v in (*(c - h), *(c + h + 1)))
        if x1 <= x0 or y1 <= y0:
            continue
        hsv = cv2.cvtColor(np.ascontiguousarray(S.images[f, y0:y1, x0:x1]), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        strong = (hsv[:, 1] >= min_sat * 255) & (hsv[:, 2] >= 50)
        hues += list(hsv[strong, 0].astype(float) * 2)
    if len(hues) < 20:
        return None
    a = np.radians(hues)
    mean = float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)
    close = np.abs((np.array(hues) - mean + 180) % 360 - 180) <= 15
    return mean if close.mean() >= min_share else None


def find_by_colour(S: Session, spans: dict[str, tuple[int, int]], b: Box3D, dets: list[Det2D], frames: list[int],
                   mine: str, other: str, min_px: int = 30, min_frames: int = 2, spread: float = 0.25,
                   max_hits: float = 0.2) -> tuple | None:
    """Where an object that left its place went, when it has one strong colour (a red speaker): the other video's
    detections of its kind, cut down to the pixels of that colour (so the black box beside it does not pull the
    place off), put in 3D from their depth. Kept when at least `min_frames` agree within `spread` metres and the
    first video, where it looked at that place, saw that colour there in at most `max_hits` of its frames.
    Returns (box at the new place, frames of the first video that showed it, colour hits there, frames that found
    it) or None."""
    own = [spans[mine][0] + f for f in b.seen_by.get(mine, [])]
    hue = colour_of(S, b, own)
    if hue is None:
        return None
    labels = kin(_labels_of(b))
    H, W = S.depth.shape[1:]
    ran = set(frames)
    pts, found = [], []
    for d in dets:
        if d.label not in labels or not (spans[other][0] <= d.frame < spans[other][1]):
            continue
        f = d.frame
        x0, y0 = (int(max(0, np.floor(v))) for v in d.box[:2])
        x1, y1 = int(min(W, np.ceil(d.box[2]))), int(min(H, np.ceil(d.box[3])))
        m = _hue_mask(S.images[f, y0:y1, x0:x1], hue)
        if m.sum() < min_px:
            continue
        v, u = np.nonzero(m)
        v, u = v + y0, u + x0
        ok = S.mask[f, v, u] & (S.depth[f, v, u] > 0)
        if ok.sum() < 5:
            continue
        z = float(np.median(S.depth[f, v[ok], u[ok]]))
        uu, vv = float(np.median(u[ok])), float(np.median(v[ok]))
        K, P = S.K[f], S.poses[f]
        pc = np.array([(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z])
        pts.append(P[:3, :3] @ pc + P[:3, 3])
        found.append(f)
    if len(pts) < min_frames:
        return None
    pts = np.array(pts)
    mid = np.median(pts, 0)
    near = np.linalg.norm(pts - mid, axis=1) <= spread
    if near.sum() < min_frames or np.linalg.norm(mid[:2] - b.centre[:2]) < 0.15:
        return None
    centre = np.median(pts[near], 0)
    nb = Box3D(b.label, b.score, centre, b.size.copy(), b.yaw, [found[i] for i in np.nonzero(near)[0]], dict(b.votes),
               owl_label=b.owl_label, label_source=b.label_source, verified=True, colour=b.colour,
               seen_by={other: sorted(found[i] - spans[other][0] for i in np.nonzero(near)[0])})
    # The first video, where it looked at that place: was that colour there already?
    views = hits = 0
    for f in range(*spans[mine]):
        if f not in ran:
            continue
        uv, zc = project(centre[None], S.poses[f], S.K[f])
        if zc[0] <= 0.2:
            continue
        u, v = (int(round(x)) for x in uv[0])
        if not (0 <= u < W and 0 <= v < H):
            continue
        if S.mask[f, v, u] and 0 < S.depth[f, v, u] < zc[0] - 0.15:       # something nearer hides the place
            continue
        r = max(6, int(np.ceil(np.max(b.size) / 2 * S.K[f][0, 0] / zc[0])))
        sl = (f, slice(max(0, v - r), v + r + 1), slice(max(0, u - r), u + r + 1))
        at = S.mask[sl] & (np.abs(S.depth[sl] - zc[0]) <= max(0.2, 0.1 * zc[0]))   # that colour at that place, not behind it
        views += 1
        hits += int((_hue_mask(S.images[sl], hue) & at).sum() >= min_px // 3)
    if views < min_frames or hits > max_hits * views:
        return None
    return nb, views, hits, int(near.sum())


def colour_ends(S: Session, spans: dict[str, tuple[int, int]], dets: list[Det2D], frames: list[int], found: list[tuple],
                names: tuple[str, str] = ("A", "B"), log=print) -> list[tuple]:
    """For each object-level change with no other end of its name, look for where it went (or came from) by its
    colour (find_by_colour). Returns new candidates in object_changes' form, for Claude to confirm."""
    out = []
    for c in found:
        b, k = c[0], c[1]
        if any(x[1] != k and x[0].label == b.label for x in found):
            continue
        mine, other = (names[0], names[1]) if k == "removed" else (names[1], names[0])
        got = find_by_colour(S, spans, b, dets, frames, mine, other)
        if got is None:
            continue
        nb, views, hits, n = got
        out.append((nb, "added" if k == "removed" else "removed", views, hits, n, n))
        log(f"Changes (objects): the {b.label} that left ({b.centre[0]:.2f}, {b.centre[1]:.2f}) has its colour at "
            f"({nb.centre[0]:.2f}, {nb.centre[1]:.2f}) in {n} frames of video {other}, and video {mine} saw that colour "
            f"there in {hits} of {views}")
    return out


def still_there(S: Session, spans: dict[str, tuple[int, int]], boxes: list[Box3D], dets: list[Det2D], frames: list[int],
                voxels: np.ndarray, min_inside: float = 0.5, min_hits: int = 2, min_share: float = 0.5,
                min_part: float = 0.25) -> Box3D | None:
    """The kept box that holds most of a change blob (at least `min_inside` of its voxels) and that BOTH videos
    still detect steadily where it is (at least `min_hits` and `min_share` of the frames that should show it):
    the object was turned or shifted in place, not removed or added. A chair turned around frees the space its
    back filled and fills space beside it, but it is the same chair in the same spot. Supports (desks, beds,
    tables, shelves) never count, and the blob must be a real part of the object (at least `min_part` of its
    longest side), so a mug gone from a desk stays a removal. None if there is none."""
    by_frame: dict[int, list[Det2D]] = {}
    for d in dets:
        by_frame.setdefault(d.frame, []).append(d)
    ran = set(frames)
    best, best_share = None, min_inside
    for b in boxes:
        if b.verified is False or {b.label, b.owl_label} & FLAT_OR_SUPPORT:
            continue
        if np.max(voxels.max(0) - voxels.min(0)) < min_part * np.max(b.size):
            continue
        share = _inside_share(voxels, b, pad=0.05)
        if share < best_share:
            continue
        labels = kin(_labels_of(b))
        ok = True
        for n, span in spans.items():
            r = view_checks(S, b, [f for f in range(*span) if f in ran], by_frame, labels)
            hits = sum(r.values())
            if hits < min_hits or hits < min_share * len(r):
                ok = False
                break
        if ok:
            best, best_share = b, share
    return best


def change_list(result: dict) -> list[tuple]:
    """The detected changes in semantics.describe's order and numbering:
    (id, type, blob in A or None, blob in B or None, move fit or None)."""
    removed, added, moves = result.get("removed", []), result.get("added", []), result.get("moves", [])
    pr, pa = {m[0] for m in moves}, {m[1] for m in moves}
    out = [("removed", b, None, None) for i, b in enumerate(removed) if i not in pr]
    out += [("added", None, b, None) for j, b in enumerate(added) if j not in pa]
    out += [("moved", removed[i], added[j], fit) for i, j, _, fit in moves]
    return [(f"c{k + 1:02d}", *x) for k, x in enumerate(out)]


def _grid(b: Box3D, n: int = 5, span: float = 0.8) -> np.ndarray:
    t = np.linspace(-span / 2, span / 2, n)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3) * b.size
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    return g @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + b.centre


def change_overlap(b: Box3D, voxels: np.ndarray, other_saw: bool, near: float = 0.08) -> float:
    """How much an object box and a change blob are one thing (0 = not at all): most of the blob inside the
    box and a fair part of the box near the blob, or (when the other video never saw the object) the box
    mostly within the blob. A mug gone from a desk does not take the desk with it."""
    if not len(voxels):
        return 0.0
    from scipy.spatial import cKDTree
    share = _inside_share(voxels, b, pad=0.05)
    d, _ = cKDTree(voxels).query(_grid(b), distance_upper_bound=near)
    cover = float(np.isfinite(d).mean())
    score = share * cover if share >= 0.4 and cover >= 0.2 else 0.0
    if not other_saw and cover >= 0.6:
        score = max(score, cover)
    return score


def _geom(b: Box3D) -> dict:
    return {"centre_m": [round(float(v), 3) for v in b.centre], "size_m": [round(float(v), 3) for v in b.size],
            "yaw_deg": round(float(np.degrees(b.yaw)), 1)}


def _moved(b: Box3D, src, dst, turn: float, forward: bool) -> dict:
    """b's geometry carried along a move from blob src to blob dst (turn about z), or back when not forward."""
    if not forward:
        src, dst, turn = dst, src, -turn
    c, s = np.cos(turn), np.sin(turn)
    xy = dst.centre[:2] + np.array([[c, -s], [s, c]]) @ (b.centre[:2] - src.centre[:2])
    z = b.centre[2] + (dst.bottom - src.bottom)
    yaw = (b.yaw + turn + np.pi / 2) % np.pi - np.pi / 2
    return {"centre_m": [round(float(xy[0]), 3), round(float(xy[1]), 3), round(float(z), 3)],
            "size_m": [round(float(v), 3) for v in b.size], "yaw_deg": round(float(np.degrees(yaw)), 1)}


def assign_states(shared: list[Box3D], result: dict, names: tuple[str, str] = ("A", "B"), log=print) -> list[Box3D]:
    """Which state shows each object. An object that overlaps a detected change follows it: removed ->
    Before only, added -> After only, moved -> both, at its old and new place (the two sightings become one
    object; one seen at only one end is carried along by the change). Of several objects under one change,
    the best-overlapping one follows it, the others only if the other video never saw them. Every other
    object is static and shows in both states, however well each video saw it. Sets state, change and
    positions in place."""
    a_name, b_name = names
    for b in shared:
        b.state, b.change, b.positions = ("both" if b.verified is not False else ""), "", {}
    kept = [b for b in shared if b.verified is not False]
    saw = lambda b, n: not b.seen_by or n in b.seen_by          # noqa: E731
    best: dict[int, tuple] = {}
    changes = change_list(result)
    for cid, kind, rb, ab, fit in changes:
        for i, b in enumerate(kept):
            for end, blob, mine, other in (("before", rb, a_name, b_name), ("after", ab, b_name, a_name)):
                if blob is None or not saw(b, mine):
                    continue
                s = change_overlap(b, blob.voxels, bool(b.seen_by) and other in b.seen_by)
                if s > 0 and s > best.get(i, (0.0,))[0]:
                    best[i] = (s, cid, kind, end, rb, ab, fit)
    # The best-overlapping object at each end of a change follows it. Others under the same blob follow only
    # if the other video never saw them (pillows on a removed bed), not a radiator behind a removed chair.
    top: dict[tuple, float] = {}
    for i, (s, cid, kind, end, *_rest) in best.items():
        top[(cid, end)] = max(top.get((cid, end), 0.0), s)
    for i in list(best):
        s, cid, kind, end, *_rest = best[i]
        other = b_name if end == "before" else a_name
        if s < top[(cid, end)] and len(kept[i].seen_by.get(other, [])) >= 2:
            del best[i]
    groups: dict[str, list] = {}
    for i, (s, cid, kind, end, rb, ab, fit) in best.items():
        b = kept[i]
        b.change = cid
        if kind == "removed":
            b.state = a_name
        elif kind == "added":
            b.state = b_name
        else:
            groups.setdefault(cid, []).append((s, i, end, rb, ab, fit))
    for cid, g in groups.items():
        g.sort(key=lambda x: -x[0])
        _, _, _, rb, ab, fit = g[0]
        if fit and fit.get("in_place"):                      # turned or shifted where it stands: it stays put
            for _, i, *_ in g:
                b = kept[i]
                b.state, b.positions = "moved", {a_name: _geom(b), b_name: _geom(b)}
            continue
        turn = np.radians(fit["turn_deg"]) if fit and fit.get("turn_determined") and fit.get("turn_deg") is not None else 0.0
        first = {end: i for _, i, end, *_ in reversed(g)}       # the best-overlapping object at each end
        if "before" in first and "after" in first:             # seen at both ends: one object, two places
            base, other = kept[first["before"]], kept[first["after"]]
            base.state, base.positions = "moved", {a_name: _geom(base), b_name: _geom(other)}
            other.state = "merged"
            base.merged_from = sorted(set(base.merged_from) | {other.id or other.label})
            done = {first["before"], first["after"]}
        else:
            done = set()
        for _, i, end, *_ in g:
            if i in done:
                continue
            b = kept[i]
            b.state = "moved"
            here, there = (a_name, b_name) if end == "before" else (b_name, a_name)
            b.positions = {here: _geom(b), there: _moved(b, rb, ab, turn, forward=end == "before")}
    n = {k: sum(b.state == k for b in kept) for k in ("both", a_name, b_name, "moved", "merged")}
    if changes:
        log(f"boxes: {n['both']} objects in both states, {n[a_name]} only in {a_name}, {n[b_name]} only in {b_name}, "
            f"{n['moved']} moved ({len(changes)} changes)")
    return shared


def link_changes(shared: list[Box3D], found: list[dict]) -> None:
    """Point each object's change at the described change (same id and type; else the nearest of its type)."""
    by_id = {c["id"]: c for c in found}
    for b in shared:
        if not b.change:
            continue
        kind = {"moved": "moved", "merged": "moved"}.get(b.state, "removed" if b.state == "A" else "added")
        c = by_id.get(b.change)
        if c is None or c["type"] != kind:
            same = [c for c in found if c["type"] == kind]
            c = min(same, key=lambda c: np.linalg.norm(np.subtract(c["centre_m"], b.centre)[:2])) if same else None
        b.change = c["id"] if c else ""


def for_state(shared: list[Box3D], name: str) -> list[Box3D]:
    """The objects state `name` shows, each where it is in that state."""
    from dataclasses import replace
    out = []
    for b in shared:
        st = b.state or "both"
        if st in ("both", name):
            out.append(b)
        elif st == "moved" and name in b.positions:
            p = b.positions[name]
            out.append(replace(b, centre=np.array(p["centre_m"], float), size=np.array(p["size_m"], float),
                               yaw=float(np.radians(p["yaw_deg"]))))
    return out


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
    """objects.json (robot-friendly) + objects_<S>.txt in SpatialLM's Bbox text format.

    Boxes Claude dropped stay in objects.json with "verified": false (viewers skip them); the text files leave them out.
    """
    import json
    path = Path(path)
    path.write_text(json.dumps({"frame": "levelled world: metres, z up, floor z=0; yaw about +z; size = (along yaw, across, height)",
                                "sessions": {k: [b.to_dict() for b in v] for k, v in boxes_by_session.items()},
                                "stats": stats or {}}, indent=1))
    for k, v in boxes_by_session.items():
        kept = [b for b in v if b.verified is not False]
        (path.parent / f"objects_{k}.txt").write_text("\n".join(b.spatiallm(i) for i, b in enumerate(kept)) + "\n")


# ---------------------------------------------------------------- cache, relations, robot formats

def from_dict(d: dict) -> Box3D:
    return Box3D(d["label"], d["score"], np.array(d["centre_m"]), np.array(d["size_m"]),
                 float(np.radians(d["yaw_deg"])), d.get("frames", []), {}, None,
                 owl_label=d.get("owl_label", d["label"]), label_source=d.get("label_source", "owlv2"),
                 verified=d.get("verified"), colour=d.get("colour", ""), part_of=d.get("part_of", ""),
                 fit=d.get("fit", ""), model=d.get("model", ""), crop=d.get("crop", ""), note=d.get("note", ""),
                 merged_from=d.get("merged_from", []), id=d.get("id", ""), seen_by=d.get("seen_by", {}),
                 state=d.get("state", ""), change=d.get("change", ""), positions=d.get("positions", {}),
                 grown=d.get("grown", {}))


def write_shared(path: str | Path, shared: list[Box3D], stats: dict | None = None, names: tuple = ("A", "B")) -> None:
    """objects.json for one shared object set: "shared" holds every object once (who saw it, which state
    shows it, the change it follows); "sessions" (derived, for older readers) lists what each state shows,
    where it is then. Objects Claude dropped stay in "shared" with "verified": false."""
    import json
    path = Path(path)
    kept = [b for b in shared if b.verified is not False]
    path.write_text(json.dumps({
        "frame": "levelled world: metres, z up, floor z=0; yaw about +z; size = (along yaw, across, height)",
        "states": "state: both = in both videos; A or B = only there (follows a removed / added change); moved = "
                  "positions per state; merged = folded into the moved object it was the other end of. In shared, "
                  "frames count A's keyframes then B's; seen_by and the sessions lists use each video's own",
        "shared": [b.to_dict() for b in shared],
        "sessions": {n: [{**b.to_dict(), "frames": b.seen_by.get(n, [])} if b.seen_by else b.to_dict()
                         for b in for_state(kept, n)] for n in names},
        "stats": stats or {}}, indent=1))
    for n in names:
        (path.parent / f"objects_{n}.txt").write_text("\n".join(b.spatiallm(i) for i, b in enumerate(for_state(kept, n))) + "\n")


def load_shared(path: str | Path) -> list[Box3D] | None:
    """The shared object set of a run (every object, dropped ones too), or None for a per-session objects.json."""
    import json
    d = json.loads(Path(path).read_text())
    return [from_dict(b) for b in d["shared"]] if "shared" in d else None


def load(path: str | Path, rejected: bool = False) -> dict[str, list[Box3D]]:
    """Boxes per session; the ones Claude dropped (verified false) only with rejected=True."""
    import json
    d = json.loads(Path(path).read_text())
    return {k: [from_dict(b) for b in v if rejected or b.get("verified") is not False] for k, v in d["sessions"].items()}


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
    ids = [b.id or f"o{i}" for i, b in enumerate(boxes)]      # the shared object id when there is one
    for i, b in enumerate(boxes):
        objs.append({**b.to_dict(), "id": ids[i]})
    for i, b in enumerate(boxes):
        on, near = support_and_near(b.centre, b.centre[2] - b.size[2] / 2, boxes, exclude=b)
        if on and on != "floor":
            j = next(k for k, x in enumerate(boxes) if x.label == on and x is not b)
            rels.append({"subject": ids[i], "predicate": "on", "object": ids[j]})
        elif on == "floor":
            rels.append({"subject": ids[i], "predicate": "on", "object": "floor"})
        if near:
            j = next(k for k, x in enumerate(boxes) if x.label == near and x is not b)
            rels.append({"subject": ids[i], "predicate": "near", "object": ids[j]})
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
