"""Claude as the judge of the object boxes, never the box maker.

Geometry makes, places and sizes every box (objects3d.py). Claude Opus 5.5 only looks at pictures
and answers in a fixed JSON schema, in two places:
  1. room vocabulary: one call on a contact sheet of a dozen sharp keyframes from both videos lists
     the things in this room. They join OWLv2's text queries, so a radiator is no longer forced
     into "heater" and a window blind into "refrigerator".
  2. verify and name: one call per fused 3D box, with up to three crops from exact-pose views that
     see the whole box unoccluded and its wireframe drawn in: keep or drop, a plain name, part-of,
     colour and fit.
Overlapping boxes that get the same name, or are parts of the same thing, are then merged by
geometry (union of their points, refit); a part whose whole has no box of its own stands in for
it. Answers are cached on disk, keyed by the images and the prompt, so a rerun costs nothing. Without ANTHROPIC_API_KEY none of this runs and the boxes keep
the detector's labels.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .geometry import project
from .objects3d import Box3D, fit_box
from .recon import Session
from .semantics import Cache, _model_to_original

MODEL = os.environ.get("WHATCHANGED_CLAUDE_MODEL", "claude-opus-5-5")
PRICE_PER_MTOK = {"claude-opus-5-5": (4.0, 20.0)}   # USD per million input, output tokens
VERSION = "n1"
KINDS = ("surface", "flat", "object")
FITS = ("good", "too_big", "too_small")
NOT_OBJECTS = {"wall", "walls", "floor", "ceiling", "room", "corner", "light", "shadow", "reflection", "person", "hand"}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
WIRE = (255, 0, 255)

VOCAB_PROMPT = """This contact sheet shows frames from two phone videos of the same room.
List the distinct physical things in this room that an object detector should look for: furniture, fixtures (such as a radiator, door or window blind) and smaller movable objects.
Use everyday names of 1-3 lowercase words, singular, one name per kind of thing, for example "radiator", "wardrobe", "tv bench", "office chair", "bedside table", "backpack". Do not list walls, floor, ceiling, light or shadows. At most 40, most prominent first.
For each, "kind" is "surface" if other things usually stand on it (bed, desk, table, shelf, tv bench, bedside table, chest of drawers), "flat" if it lies flat on the floor (rug, mat), else "object"."""

VOCAB_SCHEMA = {
    "type": "object",
    "properties": {"objects": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "kind": {"type": "string", "enum": list(KINDS)}},
        "required": ["name", "kind"], "additionalProperties": False}}},
    "required": ["objects"], "additionalProperties": False,
}

BOX_PROMPT = """These {n} pictures show one 3D box (magenta wireframe) from a scan of a room, seen from different camera positions.
Geometry fitted the box around detector hits labelled "{owl}"; that label is often wrong. The box is about {a} x {w} m across and {h} m tall, and its bottom is {z} m above the floor.{room}

Say what the box holds:
- keep: true if the box mostly holds one physical object. false if it is mostly wall, floor, window or empty space, holds several separate objects, or covers only part of a bigger object (for example one side of a bed, or a cupboard door). Things that rest on furniture, like a pillow on a bed or a monitor on a desk, are objects, not parts.
- name: the object's everyday name in 1-3 lowercase words, for example "radiator", "wardrobe", "tv bench", "window blind", "office chair", "wooden chair", "bedside table".{prefer} If keep is false, name what the box mostly shows.
- part_of: if the box covers only part of a bigger object, that object's name (for example "bed" for a bed's base), else "".
- colour: the object's main colour in one word, or "".
- fit: "good" if the wireframe hugs the object, "too_big" if it also takes in a lot of other things, "too_small" if the object extends well beyond it.
- reason: a few words on what you see."""

BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "keep": {"type": "boolean"},
        "name": {"type": "string"},
        "part_of": {"type": "string"},
        "colour": {"type": "string"},
        "fit": {"type": "string", "enum": list(FITS)},
        "reason": {"type": "string"},
    },
    "required": ["keep", "name", "part_of", "colour", "fit", "reason"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- calls and their cost

class Meter:
    """Claude calls and tokens (thread-safe); also counted into a parent, the run's total."""

    def __init__(self, parent: Meter | None = None):
        self.parent, self.lock = parent, threading.Lock()
        self.n = {"calls": 0, "cached": 0, "refused": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0}

    def add(self, **counts) -> None:
        with self.lock:
            for k, v in counts.items():
                self.n[k] += int(v or 0)
        if self.parent:
            self.parent.add(**counts)

    def usage(self, u) -> None:
        tokens_in = sum(getattr(u, k, 0) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
        self.add(calls=1, input_tokens=tokens_in, output_tokens=getattr(u, "output_tokens", 0))

    def summary(self) -> dict:
        p_in, p_out = PRICE_PER_MTOK.get(MODEL, (4.0, 20.0))
        cost = (self.n["input_tokens"] * p_in + self.n["output_tokens"] * p_out) / 1e6
        return {"model": MODEL, **self.n, "cost_usd_est": round(cost, 3)}


USAGE = Meter()   # every Claude call this run: box naming and change labels


def _api_errors() -> tuple:
    """What a failed call raises: the SDK's errors, and a bad JSON answer."""
    try:
        import anthropic
        return anthropic.APIError, ValueError
    except ImportError:
        return (ValueError,)


def available() -> bool:
    """Claude naming runs only with a key in the environment (or .env) and the SDK installed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _jpeg(im: Image.Image) -> dict:
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buf.getvalue()).decode()}}


def ask(client, images: list[Image.Image], text: str, schema: dict, effort: str, meter: Meter) -> dict | None:
    """One structured-output call, as in semantics._claude. None on a refusal or a cut-off answer."""
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": [_jpeg(im) for im in images] + [{"type": "text", "text": text}]}],
    )
    meter.usage(resp.usage)
    if resp.stop_reason in ("refusal", "max_tokens"):
        meter.add(refused=1)
        return None
    out = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(out) if out else None


# ---------------------------------------------------------------- pictures

def _frame(S: Session, j: int) -> tuple[Image.Image, object]:
    """Frame j at full resolution when its file exists (else the model-size image), and a map from model pixels into it."""
    H, W = S.depth.shape[1:]
    p = S.paths[j]
    if p and Path(p).exists():
        img = Image.open(p).convert("RGB")
        return img, lambda uv: _model_to_original(uv, img.size, (W, H))
    return Image.fromarray(S.images[j]), lambda uv: uv


def _sharpness(rgb: np.ndarray) -> float:
    g = rgb.astype(np.float32).mean(2)
    return float((4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]).var())


def contact_sheet(sessions: list[Session], n: int = 12, cols: int = 4, tile: tuple[int, int] = (640, 360)) -> Image.Image:
    """The sharpest frame from each of n / len(sessions) stretches of every video, tiled."""
    picks = []
    for S in sessions:
        for part in np.array_split(np.arange(len(S.images)), min(len(S.images), max(1, n // len(sessions)))):
            if len(part):
                picks.append((S, int(part[np.argmax([_sharpness(S.images[j]) for j in part])])))
    sheet = Image.new("RGB", (cols * tile[0], -(-len(picks) // cols) * tile[1]), (20, 20, 20))
    for k, (S, j) in enumerate(picks):
        im = _frame(S, j)[0]
        im.thumbnail(tile)
        sheet.paste(im, ((k % cols) * tile[0] + (tile[0] - im.width) // 2, (k // cols) * tile[1] + (tile[1] - im.height) // 2))
    return sheet


def _rot(yaw: float) -> np.ndarray:
    """Box frame -> world (as in Box3D.corners)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def volume_samples(b: Box3D, n: int = 5, span: float = 0.8) -> np.ndarray:
    """n^3 points on a grid through the box; span 1.0 reaches its faces."""
    t = np.linspace(-span / 2, span / 2, n)
    g = np.stack(np.meshgrid(t, t, t, indexing="ij"), -1).reshape(-1, 3) * b.size
    return g @ _rot(b.yaw).T + b.centre


def visible_share(S: Session, j: int, b: Box3D, pts: np.ndarray) -> float | None:
    """Share of these in-box points whose camera ray reaches the box before the depth map says
    something nearer is in the way. None when too few of them land on valid depth."""
    H, W = S.depth.shape[1:]
    uv, z = project(pts, S.poses[j], S.K[j])
    ok = (z > 0.1) & np.isfinite(uv).all(1)
    uv = np.where(ok[:, None], uv, -1)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, p = u[ok], v[ok], z[ok], pts[ok]
    obs = S.depth[j, v, u]
    good = S.mask[j, v, u] & (obs > 0)
    if good.sum() < 10:
        return None
    # depth at which each ray enters the box: slab test in the box frame, t = 1 at the sample
    R, C = _rot(b.yaw), S.poses[j][:3, 3]
    o, d = (C - b.centre) @ R, (p - C) @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-b.size / 2 - o) / d, (b.size / 2 - o) / d
    z_in = np.clip(np.nanmax(np.minimum(t1, t2), 1), 0, 1) * z
    clear = obs > z_in - np.maximum(0.05, 0.04 * z_in)
    return float(clear[good].mean())


def pick_views(S: Session, b: Box3D, k: int = 3, min_sep_deg: float = 20.0,
               frames: list[int] | None = None) -> list[tuple[int, np.ndarray]]:
    """Up to k frames (of `frames`, default all) that see the whole box, mostly unoccluded: largest first,
    ideally >= 20 deg apart. Returns (frame, the box's corners projected into it in model pixels)."""
    H, W = S.depth.shape[1:]
    corners, pts = b.corners(), volume_samples(b)
    cands = []
    for j in (range(len(S.depth)) if frames is None else frames):
        uv, z = project(corners, S.poses[j], S.K[j])
        if (z <= 0.1).any():
            continue
        lo, hi = uv.min(0), uv.max(0)
        clo, chi = np.clip(lo, 0, [W, H]), np.clip(hi, 0, [W, H])
        area = float(np.prod(chi - clo))
        inside = area / max(float(np.prod(hi - lo)), 1e-9)
        if inside < 0.6:
            continue
        vis = visible_share(S, j, b, pts)
        if vis is None or vis < 0.5:
            continue
        cands.append((area * vis * (1.0 if inside > 0.999 else 0.5), j, uv))
    cands.sort(key=lambda c: -c[0])

    def ray(j):
        r = b.centre - S.poses[j][:3, 3]
        return r / np.linalg.norm(r)

    picked = []
    for _, j, uv in cands:                     # strongest first, each from a new direction
        if len(picked) < k and all(np.degrees(np.arccos(np.clip(ray(j) @ ray(i), -1, 1))) >= min_sep_deg for i, _ in picked):
            picked.append((j, uv))
    for _, j, uv in cands:                     # too few directions: the next best views
        if len(picked) < k and all(j != i for i, _ in picked):
            picked.append((j, uv))
    return picked


def view_crop(S: Session, j: int, uv: np.ndarray, margin: float = 0.35, long_side: int = 768, min_px: int = 160) -> Image.Image:
    """The box in frame j with ~35% margin, long side <= 768 px, its 3D wireframe drawn in."""
    img, to_img = _frame(S, j)
    p = to_img(uv)
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    w, h = max(x1 - x0, min_px) * (0.5 + margin), max(y1 - y0, min_px) * (0.5 + margin)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(max(0, cx - w)), int(max(0, cy - h)), int(min(img.width, cx + w)), int(min(img.height, cy + h)))
    crop = img.crop(box)
    s = min(long_side / max(crop.size), 2.0)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
    q = (p - np.array(box[:2])) * s
    draw = ImageDraw.Draw(crop)
    for a, c in EDGES:
        draw.line([tuple(q[a]), tuple(q[c])], fill=WIRE, width=3)
    return crop


def _save(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=88)
    return path


# ---------------------------------------------------------------- answers

def clean(name, words: int = 3) -> str:
    """Lowercase plain words; "" when longer than `words` words."""
    s = " ".join(re.sub(r"[^a-z0-9 \-]+", " ", str(name).lower()).split())
    return s if len(s.split()) <= words else ""


def parse_vocabulary(ans: dict | None, limit: int = 40) -> list[dict]:
    """Claude's room list -> up to 40 unique {"name", "kind"}, without walls, floors and long phrases."""
    out, seen = [], set()
    for o in (ans or {}).get("objects", []):
        o = o if isinstance(o, dict) else {"name": o}
        name = clean(o.get("name", ""))
        if not name or len(name) > 30 or name in NOT_OBJECTS or name in seen:
            continue
        seen.add(name)
        out.append({"name": name, "kind": o.get("kind") if o.get("kind") in KINDS else "object"})
    return out[:limit]


def _m(v: float) -> str:
    return f"{v:.2f}" if v < 0.3 else f"{v:.1f}"


def box_prompt(b: Box3D, n: int, vocab: list[str]) -> str:
    return BOX_PROMPT.format(
        n=n, owl=b.owl_label or b.label, a=_m(b.size[0]), w=_m(b.size[1]), h=_m(b.size[2]),
        z=_m(max(0.0, b.centre[2] - b.size[2] / 2)),
        room=f"\nThings seen elsewhere in this room: {', '.join(vocab)}." if vocab else "",
        prefer=" Prefer a name from the room list when one fits." if vocab else "")


def apply(b: Box3D, ans: dict) -> None:
    """Claude's verdict onto the box; the detector's label stays in owl_label."""
    b.owl_label = b.owl_label or b.label
    b.label = clean(ans.get("name", ""), words=4) or b.label
    b.verified = bool(ans.get("keep"))
    b.part_of = clean(ans.get("part_of", ""), words=4)
    b.colour = clean(ans.get("colour", ""), words=2)
    b.fit = ans.get("fit") if ans.get("fit") in FITS else ""
    b.note = str(ans.get("reason", ""))[:160]
    b.label_source, b.model = "claude", MODEL


# ---------------------------------------------------------------- merging

def _in_footprint(b: Box3D, xy: np.ndarray, pad: float = 0.0) -> np.ndarray:
    c, s = np.cos(b.yaw), np.sin(b.yaw)
    r = (xy - b.centre[:2]) @ np.array([[c, -s], [s, c]])
    return (np.abs(r) <= b.size[:2] / 2 + pad).all(1)


def footprint_overlap(a: Box3D, b: Box3D, pad: float = 0.0, n: int = 80) -> tuple[float, float]:
    """(IoU, intersection over the smaller footprint), on a grid; `pad` grows both footprints."""
    xy = np.concatenate([a.corners()[:4, :2], b.corners()[:4, :2]])
    lo, hi = xy.min(0) - pad, xy.max(0) + pad
    g = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)), -1).reshape(-1, 2)
    ia, ib = _in_footprint(a, g, pad), _in_footprint(b, g, pad)
    inter = (ia & ib).sum()
    return float(inter / max((ia | ib).sum(), 1)), float(inter / max(min(ia.sum(), ib.sum()), 1))


def _touch(a: Box3D, b: Box3D, iou: float, near_m: float, slack: float = 0.1) -> bool:
    """Heights overlap, and the footprints overlap: IoU, or a share of the smaller one with 10 cm slack
    (a bed's side panel against the bed), or centres closer than near_m."""
    za, zb = a.centre[2] + np.array([-0.5, 0.5]) * a.size[2], b.centre[2] + np.array([-0.5, 0.5]) * b.size[2]
    if min(za[1], zb[1]) + slack / 2 <= max(za[0], zb[0]):
        return False
    if np.linalg.norm(a.centre[:2] - b.centre[:2]) < near_m:
        return True
    return footprint_overlap(a, b)[0] > iou or footprint_overlap(a, b, pad=slack / 2)[1] > iou


def _is(part: str, whole: str) -> bool:
    """A part's part_of names this whole: the same words, or its head noun ("chair" for "wooden chair")."""
    return bool(part) and (part == whole or (" " not in part and whole.split()[-1] == part))


def _part(b: Box3D) -> str:
    """What a dropped box is a part of: its part_of, or, for a box dropped only for holding too little of a
    real thing ("covers only the middle of a longer radiator"), that thing."""
    if b.part_of or b.verified is not False:
        return b.part_of
    return b.label if b.fit == "too_small" and b.label not in NOT_OBJECTS and "empty" not in b.label else ""


def _same(a: Box3D, b: Box3D) -> bool:
    """Claude says these are one thing: both kept under one name, a part and its whole, or two parts of one whole."""
    if a.label_source != "claude" or b.label_source != "claude":
        return False
    if a.verified and b.verified:
        return a.label == b.label
    if a.verified or b.verified:
        whole, part = (a, b) if a.verified else (b, a)
        return _is(_part(part), whole.label)
    return bool(_part(a)) and _part(a) == _part(b)


def merge(boxes: list[Box3D], iou: float = 0.2, near_m: float = 0.4, room_yaw: float | None = None) -> tuple[list[Box3D], int]:
    """Overlapping boxes that are one thing (same name, or parts of it) become one box: the union of
    their points, refit, leaving out boxes Claude called too big when others fit better.
    Parts whose whole has no box of its own nearby stand in for it, named after it and
    marked too small, so an object seen only in part is not lost; parts of a whole that is boxed
    right there stay dropped. With room_yaw, a merged piece of furniture lines up with the walls again.
    Returns the new list and how many boxes were folded into others."""
    root = list(range(len(boxes)))

    def find(i):
        while root[i] != i:
            root[i] = root[root[i]]
            i = root[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _same(boxes[i], boxes[j]) and _touch(boxes[i], boxes[j], iou, near_m):
                root[find(j)] = find(i)
    groups = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    wholes = [b for b in boxes if b.label_source == "claude" and b.verified]
    out, folded = [], 0
    for g in sorted(groups.values(), key=min):
        members = [boxes[i] for i in g]
        kept = [m for m in members if m.verified]
        part = _part(members[0]) if not kept and members[0].label_source == "claude" else ""
        # a part of a whole that has its own box right there stays dropped; a room can hold three radiators,
        # so a radiator box elsewhere does not make this one a mere part
        if not kept and (not part or any(_is(part, w.label) and footprint_overlap(w, m, pad=0.1)[1] > 0
                                         for w in wholes for m in members)):
            out += members                     # dropped, or a part of something boxed right there
            continue
        if len(g) == 1 and kept:
            out.append(members[0])
            continue
        fitting = [m for m in members if m.fit != "too_big"] or members
        rep = max([m for m in kept if m.fit != "too_big"] or kept or members, key=lambda m: float(np.prod(m.size)))
        if len(g) == 1:                        # a lone part stands in for its whole
            pts, (c, s, yaw) = rep.points, (rep.centre, rep.size, rep.yaw)
        else:
            # a box grown to its 2D detections (objects3d.grow_box) brings its whole volume, not just its points
            pts = np.concatenate([m.points if m.points is not None and len(m.points) and not m.grown
                                  else volume_samples(m, 9, 1.0) for m in fitting])
            c, s, yaw = fit_box(pts)
            if room_yaw is not None:
                from .objects3d import fit_box_at
                d = (yaw - room_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
                if abs(d) <= np.radians(12) and s[:2].max() >= 0.3:
                    c, s, yaw = fit_box_at(pts, yaw - d)
        votes = {}
        for m in members:
            for k, v in m.votes.items():
                votes[k] = votes.get(k, 0) + v
        seen_by = {}
        for m in members:
            for k, v in m.seen_by.items():
                seen_by[k] = sorted(set(seen_by.get(k, [])) | set(v))
        out.append(Box3D(rep.label if kept else part, max(m.score for m in members), c, s, yaw,
                         sorted({f for m in members for f in m.frames}), votes, pts,
                         owl_label=rep.owl_label, label_source="claude", verified=True, colour=rep.colour,
                         fit=rep.fit if kept else "too_small", model=rep.model, crop=rep.crop,
                         note=rep.note if kept else f"only part of the {part} was boxed: {rep.note}"[:160],
                         merged_from=[m.owl_label or m.label for m in members] if len(g) > 1 else [],
                         seen_by=seen_by, grown=rep.grown if len(g) == 1 else {},
                         dets=[d for m in members for d in m.dets]))
        folded += len(g) - 1
    return out, folded


# ---------------------------------------------------------------- the pass

GONE_PROMPT = """The first {n} picture(s) come from video {own} of a room; the magenta wireframe marks {what}.
The other {m} picture(s) come from video {other} of the same room, filmed at another time; the same wireframe marks the same place in the room.
Answer in JSON:
  "object": true if the wireframe in the first picture(s) surrounds one real, movable object, false if it is a part of furniture, a wall, a reflection, a shadow or several things;
  "there": is that same physical object at the marked place in the later picture(s)? "present" if it is still there (even partly hidden or from another side), "absent" if the place is empty or holds something else, "unsure" if the pictures cannot tell (blocked, blurred, cut off);
  "name": a short common name for the object (1-3 words, lowercase)."""
GONE_SCHEMA = {
    "type": "object",
    "properties": {"object": {"type": "boolean"}, "there": {"type": "string", "enum": ["present", "absent", "unsure"]},
                   "name": {"type": "string"}},
    "required": ["object", "there", "name"],
    "additionalProperties": False,
}


def confirm_changes(out_dir: str | Path, S: Session, spans: dict[str, tuple[int, int]], found: list[tuple],
                    log=print, client=None) -> list[tuple] | None:
    """Claude looks at each object-level change (objects3d.object_changes): the object in its own video next
    to the other video's view of the same place, and keeps the change only if the object is real and absent
    there. Unnamed boxes take Claude's name. Answers are cached, so a replay without a key (CI) gets the same
    verdicts. None when some change has no answer and Claude is not available (keep every change then)."""
    if not found:
        return None
    namer = Namer(out_dir, log=log, client=client)
    jobs = []
    for c in found:
        b, kind = c[0], c[1]
        own, other = ("A", "B") if kind == "removed" else ("B", "A")
        mine = pick_views(S, b, k=2, frames=list(range(*spans[own])))
        theirs = pick_views(S, b, k=2, frames=list(range(*spans[other])))
        if not mine or not theirs:
            continue
        ims = [view_crop(S, j, uv) for j, uv in mine + theirs]
        what = f"a {b.label} about {_m(b.size[0])} x {_m(b.size[1])} x {_m(b.size[2])}"
        text = GONE_PROMPT.format(n=len(mine), own=own, what=what, m=len(theirs), other=other)
        jobs.append((c, ims, text, namer.cache.key(*ims, extra=f"{VERSION}|gone|{text}")))
    answers, todo = {}, []
    for job in jobs:
        if job[3] in namer.cache.data:
            answers[id(job[0])] = namer.cache.data[job[3]]
            namer.meter.add(cached=1)
        else:
            todo.append(job)
    if todo and not available():
        return None
    if todo:
        pool = ThreadPoolExecutor(namer.workers)
        futures = {pool.submit(ask, namer._client(), ims, text, GONE_SCHEMA, "medium", namer.meter): (c, key)
                   for c, ims, text, key in todo}
        try:
            for f in as_completed(futures):
                c, key = futures[f]
                try:
                    ans = f.result()
                except _api_errors() as e:
                    namer._failed(f"check of the '{c[0].label}' change", e)
                    if namer.error:
                        return None
                    continue
                if ans:
                    namer.cache.data[key] = ans
                    namer.cache.save()
                    answers[id(c)] = ans
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    kept = []
    for c, ims, _, _ in jobs:
        ans = answers.get(id(c))
        if not ans or not ans.get("object") or ans.get("there") != "absent":
            continue
        b = c[0]
        if b.label_source != "claude" and clean(ans.get("name", "")):
            b.owl_label, b.label = b.owl_label or b.label, clean(ans["name"])
        kept.append(c)
    log(f"Changes (objects): {MODEL} confirmed {len(kept)} of {len(found)} ({len(found) - len(jobs)} had no clear view)")
    return kept


class Namer:
    """Room vocabulary and box verdicts for one run folder, with a disk cache and a cost meter."""

    def __init__(self, out_dir: str | Path, log=print, client=None, workers: int = 8):
        self.out, self.log, self.client, self.workers = Path(out_dir), log, client, workers
        self.cache = Cache(self.out / "naming_cache.json")
        self.meter = Meter(parent=USAGE)
        self.error, self.vocab, self.answered = "", [], 0
        self.counts = {"checked": 0, "no_view": 0, "kept": 0, "dropped": 0, "merged": 0}

    def _client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic(max_retries=4)
        return self.client

    def _failed(self, what: str, e: Exception, fatal=(401, 403, 404)) -> None:
        code = getattr(e, "status_code", None)
        why = type(e).__name__ + (f" {code}" if code else "")
        if code in fatal:                      # bad key, no access, unknown model: every call would fail alike
            self.error = why
            self.log(f"boxes: Claude unavailable ({why}); keeping OWLv2 labels")
        else:
            self.meter.add(failed=1)
            self.log(f"warning: Claude {what} failed ({why}); carrying on without it")

    def vocabulary(self, sessions: list[Session]) -> list[dict]:
        """Up to 40 {"name", "kind"} for the things in this room, from one call on a contact sheet ([] if none)."""
        sheet = contact_sheet(sessions)
        _save(sheet, self.out / "object_crops" / "contact_sheet.jpg")
        key = self.cache.key(sheet, extra=f"{VERSION}|vocab|{VOCAB_PROMPT}")
        ans = self.cache.data.get(key)
        if ans is not None:
            self.meter.add(cached=1)
        else:
            try:
                ans = ask(self._client(), [sheet], VOCAB_PROMPT, VOCAB_SCHEMA, "low", self.meter)
            except _api_errors() as e:
                self._failed("room vocabulary", e, fatal=(400, 401, 403, 404))   # the first call: a 400 here is systematic
                return []
            if ans is None:
                self.log("boxes: Claude gave no room vocabulary; OWLv2 keeps its own list")
                return []
            self.cache.data[key] = ans
            self.cache.save()
        self.vocab = parse_vocabulary(ans)
        self.log(f"boxes: room vocabulary from {MODEL}: {', '.join(o['name'] for o in self.vocab)}")
        return self.vocab

    def check(self, S: Session, key: str, boxes: list[Box3D], views=None, tag: str = "") -> int:
        """Claude verifies and names these boxes (one call each, in parallel, cached by the pictures and the
        prompt); `views(box)` lists the frames a box may be shown from (default: all). Boxes without an
        answer keep their label. Returns how many got an answer."""
        vocab = [o["name"] for o in self.vocab]
        jobs = []
        for i, b in enumerate(boxes):
            b.owl_label = b.owl_label or b.label
            chosen = pick_views(S, b, frames=views(b) if views else None)
            if not chosen:
                self.counts["no_view"] += 1
                continue
            ims = [view_crop(S, j, uv) for j, uv in chosen]
            b.crop = str(_save(ims[0], self.out / "object_crops" / f"{key}{tag}_{i:02d}.jpg").relative_to(self.out))
            text = box_prompt(b, len(ims), vocab)
            jobs.append((b, ims, text, self.cache.key(*ims, extra=f"{VERSION}|box|{text}")))
        self.counts["checked"] += len(jobs)
        todo, got = [], 0
        for job in jobs:
            if job[3] in self.cache.data:
                apply(job[0], self.cache.data[job[3]])
                self.answered += 1
                got += 1
                self.meter.add(cached=1)
            else:
                todo.append(job)
        if todo and not self.error:
            client = self._client()
            pool = ThreadPoolExecutor(self.workers)
            futures = {pool.submit(ask, client, ims, text, BOX_SCHEMA, "medium", self.meter): (b, key)
                       for b, ims, text, key in todo}
            try:
                for f in as_completed(futures):
                    b, key = futures[f]
                    try:
                        ans = f.result()
                    except _api_errors() as e:
                        self._failed(f"on the '{b.owl_label}' box", e)
                        if self.error:
                            break
                        continue
                    if ans:
                        self.cache.data[key] = ans
                        self.cache.save()
                        apply(b, ans)
                        self.answered += 1
                        got += 1
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
        return got

    def name(self, sessions: dict[str, Session], boxes: dict[str, list[Box3D]], views: dict | None = None,
             room_yaw: float | None = None) -> dict[str, list[Box3D]]:
        """Claude verifies and names every box (one call each, in parallel, cached); overlapping boxes with the
        same name or part-of are then merged. Boxes without an answer keep the detector's label.
        `views` maps a session to a function giving the frames each box may be shown from."""
        self.counts["checked"] = 0
        for s, bs in boxes.items():
            self.check(sessions[s], s, bs, (views or {}).get(s))
        out = {}
        for s, bs in boxes.items():
            out[s], folded = merge(bs, room_yaw=room_yaw)
            self.counts["merged"] += folded
        self._tally(out)
        m = self.meter.n
        flat = [b for v in out.values() for b in v]
        self.log(f"boxes: {MODEL} checked {self.counts['checked']} boxes: {self.counts['kept']} kept, {self.counts['dropped']} dropped, "
                 f"{self.counts['merged']} merged into others; {m['calls']} calls ({m['cached']} cached), "
                 f"about ${self.meter.summary()['cost_usd_est']:.2f}")
        renamed = sorted({f"{b.owl_label} -> {b.label}" for b in flat if b.verified and b.owl_label != b.label})
        if renamed:
            self.log("boxes: renamed " + ", ".join(renamed))
        return out

    def _tally(self, out: dict[str, list[Box3D]]) -> None:
        flat = [b for v in out.values() for b in v]
        self.counts["kept"] = sum(b.verified is True for b in flat)
        self.counts["dropped"] = sum(b.verified is False for b in flat)

    def stats(self) -> dict:
        """Provenance for objects.json and diag."""
        return {"label_source": "claude" if self.answered else "owlv2", "prompt_version": VERSION,
                "vocabulary": self.vocab, **self.counts, **self.meter.summary(),
                **({"error": self.error} if self.error else {})}
