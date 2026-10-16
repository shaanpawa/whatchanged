"""Name each change and write its sentence.

Geometry decides what changed, where, and by how much. A vision model only names the
object and the fixed thing it sits on. Sentences are filled from measured facts, so the
words, the 3D boxes and the numbers cannot disagree. When Claude has already named the 3D
object box at a change's spot (naming.py), the labeller is told that name, so the change and
the box say "radiator" alike.

Labellers, in order of preference:
  claude  - Claude (Opus 5.5) via the Anthropic SDK, when ANTHROPIC_API_KEY is set
  gemini  - Gemini via the google-genai SDK, when GEMINI_API_KEY is set
  siglip  - SigLIP 2 zero-shot over an indoor vocabulary, on CPU, no key needed
  none    - "object"
Every answer is cached on disk, keyed by image hashes and the prompt version.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .changes import Blob
from .geometry import project
from .recon import Session

PROMPT_VERSION = "v1"
GEMINI_MODEL = os.environ.get("WHATCHANGED_GEMINI_MODEL", "gemini-3.8-flash")
CLAUDE_MODEL = os.environ.get("WHATCHANGED_CLAUDE_MODEL", "claude-opus-5-5")
SIGLIP_MODEL = "google/siglip2-base-patch16-224"

VOCAB = [
    "chair", "office chair", "stool", "sofa", "armchair", "table", "desk", "coffee table", "bed", "pillow",
    "blanket", "duvet", "lamp", "floor lamp", "desk lamp", "plant", "potted plant", "bin", "laundry basket",
    "suitcase", "backpack", "bag", "cardboard box", "box", "storage box", "plastic tub", "shoebox", "shoes",
    "book", "stack of books", "laptop", "monitor", "keyboard", "television", "speaker", "guitar", "bottle",
    "mug", "cup", "vase", "picture frame", "mirror", "clock", "towel", "jacket", "clothes", "hat", "toy",
    "ball", "fan", "heater", "vacuum cleaner", "mop", "bucket", "shelf", "cabinet", "drawer unit", "rug",
    "cushion", "bicycle", "umbrella", "printer", "router", "kettle", "tray", "basket", "trolley", "tote",
]
SUPPORTS = ["floor", "desk", "table", "bed", "shelf", "chair", "sofa", "cabinet", "windowsill", "counter"]


# ---------------------------------------------------------------- views and crops

def _model_to_original(uv: np.ndarray, orig_wh: tuple[int, int], model_wh: tuple[int, int]) -> np.ndarray:
    """Undo MapAnything's cover-resize + centre-crop."""
    (W1, H1), (tW, tH) = orig_wh, model_wh
    s = max(tW / W1, tH / H1)
    w, h = int(W1 * s), int(H1 * s)
    left, top = (w - tW) // 2, (h - tH) // 2
    return (uv + np.array([left, top])) / s


def best_view(points: np.ndarray, S: Session, require_visible: bool = True) -> tuple[int, np.ndarray, int]:
    """Frame of S that sees most of these points; returns (index, bbox in model pixels, count)."""
    H, W = S.depth.shape[1:]
    best = (-1, None, 0)
    for j in range(len(S.depth)):
        uv, z = project(points, S.poses[j], S.K[j])
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (z > 0.2) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if require_visible and ok.any():
            d = S.depth[j, v[ok], u[ok]]
            vis = np.abs(d - z[ok]) < np.maximum(0.05, 0.05 * z[ok])
            idx = np.nonzero(ok)[0][vis]
        else:
            idx = np.nonzero(ok)[0]
        # Prefer views where the object is fully inside the frame and not tiny.
        n = len(idx) * (0.5 if len(idx) < 0.6 * len(points) else 1.0)
        if n > best[2] and len(idx) >= 5:
            box = np.array([uv[idx, 0].min(), uv[idx, 1].min(), uv[idx, 0].max(), uv[idx, 1].max()])
            best = (j, box, int(n))
    return best


def crop(S: Session, j: int, box_model: np.ndarray, margin: float = 0.35, min_px: int = 160) -> Image.Image:
    img = Image.open(S.paths[j]).convert("RGB")
    H, W = S.depth.shape[1:]
    corners = _model_to_original(box_model.reshape(2, 2), img.size, (W, H)).reshape(-1)
    x0, y0, x1, y1 = corners
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0, min_px) * (1 + margin) / 2
    box = (int(max(0, cx - half)), int(max(0, cy - half)), int(min(img.width, cx + half)), int(min(img.height, cy + half)))
    return img.crop(box)


def _save(img: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)
    return str(path)


# ---------------------------------------------------------------- labellers

class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def key(self, *imgs: Image.Image, extra: str = "") -> str:
        h = hashlib.sha1((PROMPT_VERSION + extra).encode())
        for im in imgs:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            h.update(buf.getvalue())
        return h.hexdigest()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1))


GEMINI_PROMPT = """You are labelling one change found by a 3D change detector in a room.
The detector already decided the change type ({kind}) and its position from geometry. Do not second-guess
the geometry. Look at the images and answer only about the object.
Image 1: BEFORE view of the spot. Image 2: AFTER view of the spot.{extra}
Return JSON with:
  "label": the object's name in 1-3 lowercase words, specific but common (e.g. "desk chair", "suitcase", "floor lamp");
  "colour": its main colour in one word, or "" if unclear;
  "on_before": the large fixed thing it rests on or stands next to in the BEFORE image (e.g. "floor", "desk", "bed", "shelf"), or "" if not visible or if it is absent before;
  "on_after": the same for the AFTER image, or "" if it is absent after;
  "same_object": for a moved object, true if images 1 and 3 show the same physical object, else false; null otherwise;
  "visible": true if the object is clearly visible in at least one image."""


def _gemini(images: list[Image.Image], kind: str, extra: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    parts = []
    for im in images:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=90)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))
    parts.append(GEMINI_PROMPT.format(kind=kind, extra=extra))
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=parts,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0),
    )
    return json.loads(resp.text)


LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "colour": {"type": "string"},
        "on_before": {"type": "string"},
        "on_after": {"type": "string"},
        "same_object": {"type": ["boolean", "null"]},
        "visible": {"type": "boolean"},
    },
    "required": ["label", "colour", "on_before", "on_after", "same_object", "visible"],
    "additionalProperties": False,
}


def _claude(images: list[Image.Image], kind: str, extra: str) -> dict | None:
    """Claude names the object from before/after crops. Returns None on a refusal."""
    import base64

    import anthropic

    content = []
    for im in images:
        im = im.convert("RGB")
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.standard_b64encode(buf.getvalue()).decode()}})
    content.append({"type": "text", "text": GEMINI_PROMPT.format(kind=kind, extra=extra)})
    from .naming import USAGE
    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": LABEL_SCHEMA}},
        messages=[{"role": "user", "content": content}],
    )
    USAGE.usage(resp.usage)
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    return json.loads(text) if text else None


_SIGLIP = None


def _siglip(images: list[Image.Image], kind: str) -> dict:
    global _SIGLIP
    import torch
    from transformers import AutoModel, AutoProcessor

    if _SIGLIP is None:
        _SIGLIP = (AutoModel.from_pretrained(SIGLIP_MODEL).eval(), AutoProcessor.from_pretrained(SIGLIP_MODEL))
    model, proc = _SIGLIP
    obj_img = images[0] if kind in ("removed", "moved") else images[1]
    texts = [f"a photo of a {w}." for w in VOCAB]
    with torch.no_grad():
        inp = proc(text=texts, images=[obj_img], padding="max_length", return_tensors="pt")
        logits = model(**inp).logits_per_image[0]
    probs = torch.sigmoid(logits).numpy()
    top = int(np.argmax(probs))
    return {"label": VOCAB[top], "colour": "", "on_before": "", "on_after": "", "same_object": None,
            "visible": True, "score": float(probs[top])}


def pick_labeller(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic  # noqa: F401
            return "claude"
        except ImportError:
            pass
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import google.genai  # noqa: F401
            return "gemini"
        except ImportError:
            pass
    try:
        import transformers  # noqa: F401
        return "siglip"
    except ImportError:
        return "none"


def label(images: list[Image.Image], kind: str, labeller: str, cache: Cache, extra: str = "") -> dict:
    # Keyed on the images and the question only, so a cached Gemini answer replays without a key.
    key = cache.key(*images, extra=f"{kind}|{extra}")
    if key in cache.data:
        return cache.data[key]
    if labeller == "claude":
        try:
            ans = _claude(images, kind, extra)
        except Exception as e:  # noqa: BLE001  (a failed call, e.g. overloaded, must not lose the whole run)
            print(f"warning: Claude change label failed ({type(e).__name__}); using SigLIP for this one", flush=True)
            return {**_siglip(images, kind), "labeller": "siglip"}   # not cached, so a rerun asks Claude again
        if ans is None:                      # declined: fall back to the offline labeller
            labeller = "siglip"
            ans = _siglip(images, kind)
    elif labeller == "gemini":
        ans = _gemini(images, kind, extra)
    elif labeller == "siglip":
        ans = _siglip(images, kind)
    else:
        ans = {"label": "object", "colour": "", "on_before": "", "on_after": "", "same_object": None, "visible": True}
    ans["labeller"] = labeller
    cache.data[key] = ans
    cache.save()
    return ans


def _pick(ans: dict, box_name: str | None, labeller: str) -> str:
    """Gemini's name wins; otherwise a 3D box's name (seen from several views) beats one SigLIP crop."""
    if ans.get("labeller", labeller) in ("claude", "gemini") and ans.get("label"):   # not a SigLIP stand-in
        return ans["label"]
    return box_name or ans.get("label", "object")


# ---------------------------------------------------------------- sentences

def _support(bottom: float, named: str) -> str:
    if named and named not in ("unknown",):
        return f"the {named}" if named != "floor" else "the floor"
    if bottom < 0.12:
        return "the floor"
    return f"a surface about {bottom:.1f} m up"


def sentence(c: dict) -> str:
    name = (c["colour"] + " " if c.get("colour") else "") + c["label"]
    size = c["size_m"]
    dims = f"about {round(size[0] * 100, -1):.0f} × {round(size[1] * 100, -1):.0f} cm, {round(size[2] * 100, -1):.0f} cm tall"
    near_a = f" next to the {c['near_after']}" if c.get("near_after") else ""
    near_b = f" next to the {c['near_before']}" if c.get("near_before") else ""
    if c["type"] == "added":
        return f"A {name} ({dims}) appeared on {_support(c['bottom_m'], c.get('on_after', ''))}{near_a}."
    if c["type"] == "removed":
        return (f"The {name} on {_support(c['bottom_m'], c.get('on_before', ''))}{near_b} is gone; "
                f"video B saw that spot empty from {c['evidence']['free_views']} viewpoints.")
    move = f"moved about {c['move']['distance_m']:.1f} m"
    a, b = c.get("on_before", ""), c.get("on_after", "")
    if a and b and a != b:
        move += f" from the {a} to the {b}"
    elif c.get("near_before") and c.get("near_after") and c["near_before"] != c["near_after"]:
        move += f" from beside the {c['near_before']} to beside the {c['near_after']}"
    elif c["move"].get("height_change_m", 0) > 0.25:
        move += f" up onto {_support(c['bottom_m'], b)}"
    if c["move"].get("turn_deg") is not None and abs(c["move"]["turn_deg"]) >= 20:
        move += f" and turned about {abs(round(c['move']['turn_deg'], -1)):.0f}°"
    return f"The {name} {move}."


def check_sentence(c: dict) -> list[str]:
    """Every number in a sentence must come from the change's own facts."""
    import re
    problems = []
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", c["sentence"])]
    allowed = {round(v * 100, -1) for v in c["size_m"]} | {c["evidence"].get("free_views", -1)}
    if "move" in c:
        allowed |= {round(c["move"]["distance_m"], 1)}
        if c["move"].get("turn_deg") is not None:
            allowed |= {abs(round(c["move"]["turn_deg"], -1))}
    allowed |= {round(c["bottom_m"], 1)}
    for n in nums:
        if not any(abs(n - a) < 1e-6 for a in allowed):
            problems.append(f"{c['id']}: number {n} in sentence is not in its facts")
    return problems


# ---------------------------------------------------------------- assemble

def _boxes_context(pts: np.ndarray, bottom: float, boxes) -> tuple[str | None, str, str, str]:
    """(name from the object box containing these points, what it rests on, what it is next to,
    that name again if Claude checked it, else "")."""
    if not boxes:
        return None, "", "", ""
    from .objects3d import label_for_points, support_and_near
    name, _ = label_for_points(pts, boxes)
    own = None
    if name:
        from .objects3d import _inside_share
        own = max(boxes, key=lambda b: _inside_share(pts, b, pad=0.05))
    on, near = support_and_near(pts.mean(0), bottom, boxes, exclude=own)
    checked = own.label if own is not None and own.label_source == "claude" else ""
    return name, on, near, checked


def _hint(checked: str, labeller: str) -> str:
    """Tell a vision labeller what Claude called the object box here, so the change and the box share a name."""
    if not checked or labeller not in ("claude", "gemini"):
        return ""
    return f' A 3D object box at this spot was named "{checked}" from several other views; use that name if it is the same object.'


def describe(result: dict, A: Session, B: Session, out_dir: Path, labeller: str = "auto", log=print,
             boxes: dict | None = None) -> list[dict]:
    labeller = pick_labeller(labeller)
    boxes = boxes or {}
    log(f"Labels: using {labeller}")
    cache = Cache(out_dir / "label_cache.json")
    crops = out_dir / "crops"
    changes = []
    removed, added = result["removed"], result["added"]
    paired_r = {m[0] for m in result["moves"]}
    paired_a = {m[1] for m in result["moves"]}

    def facts(b: Blob) -> dict:
        return {"centre_m": [round(float(x), 3) for x in b.centre], "size_m": [round(float(x), 3) for x in b.size],
                "yaw_deg": round(float(np.degrees(b.yaw)), 1), "bottom_m": round(b.bottom, 3), "top_m": round(b.top, 3)}

    def pair_views(b: Blob, own: Session, other: Session) -> tuple[Image.Image, Image.Image, int, int]:
        j, box, _ = best_view(b.voxels, own)
        k, box2, _ = best_view(b.voxels, other, require_visible=False)
        im1 = crop(own, j, box) if j >= 0 else Image.new("RGB", (64, 64), "gray")
        im2 = crop(other, k, box2) if k >= 0 else Image.new("RGB", (64, 64), "gray")
        return im1, im2, j, k

    n = 0
    for i, b in enumerate(removed):
        if i in paired_r:
            continue
        n += 1
        before, after, j, k = pair_views(b, A, B)
        bname, on, near, checked = _boxes_context(b.voxels, b.bottom, boxes.get("A"))
        ans = label([before, after], "removed", labeller, cache, extra=_hint(checked, labeller))
        c = {"id": f"c{n:02d}", "type": "removed", **facts(b), "label": _pick(ans, bname, labeller), "colour": ans.get("colour", ""),
             "on_before": on or ans.get("on_before", ""), "on_after": "", "near_before": near, "near_after": "",
             "evidence": {"free_views": b.views, "free_share": round(b.score, 2), "noise_margin": round(b.margin, 1)},
             "frames": {"A": A.paths[j] if j >= 0 else None, "B": B.paths[k] if k >= 0 else None},
             "times": {"A": float(A.times[j]) if j >= 0 else None, "B": float(B.times[k]) if k >= 0 else None},
             "crops": {"before": _save(before, crops / f"c{n:02d}_before.jpg"), "after": _save(after, crops / f"c{n:02d}_after.jpg")}}
        changes.append(c)
    for i, b in enumerate(added):
        if i in paired_a:
            continue
        n += 1
        after, before, k, j = pair_views(b, B, A)
        bname, on, near, checked = _boxes_context(b.voxels, b.bottom, boxes.get("B"))
        ans = label([before, after], "added", labeller, cache, extra=_hint(checked, labeller))
        c = {"id": f"c{n:02d}", "type": "added", **facts(b), "label": _pick(ans, bname, labeller), "colour": ans.get("colour", ""),
             "on_before": "", "on_after": on or ans.get("on_after", ""), "near_before": "", "near_after": near,
             "evidence": {"free_views": b.views, "free_share": round(b.score, 2), "noise_margin": round(b.margin, 1)},
             "frames": {"A": A.paths[j] if j >= 0 else None, "B": B.paths[k] if k >= 0 else None},
             "times": {"A": float(A.times[j]) if j >= 0 else None, "B": float(B.times[k]) if k >= 0 else None},
             "crops": {"before": _save(before, crops / f"c{n:02d}_before.jpg"), "after": _save(after, crops / f"c{n:02d}_after.jpg")}}
        changes.append(c)
    for i, j_, cost, fit in result["moves"]:
        r, a = removed[i], added[j_]
        n += 1
        before, _, j, _ = pair_views(r, A, B)
        after, _, k, _ = pair_views(a, B, A)
        nb, onb, nearb, cb = _boxes_context(r.voxels, r.bottom, boxes.get("A"))
        na, ona, neara, ca = _boxes_context(a.voxels, a.bottom, boxes.get("B"))
        ans = label([before, after], "moved", labeller, cache,
                    extra=" Image 1 shows the object at its old place, image 2 at its new place." + _hint(cb or ca, labeller))
        c = {"id": f"c{n:02d}", "type": "moved", **facts(a), "label": _pick(ans, nb or na, labeller), "colour": ans.get("colour", ""),
             "on_before": onb or ans.get("on_before", ""), "on_after": ona or ans.get("on_after", ""),
             "near_before": nearb, "near_after": neara,
             "before": facts(r), "after": facts(a),
             "move": {"distance_m": round(float(np.linalg.norm(a.centre[:2] - r.centre[:2])), 2),
                      "height_change_m": round(float(a.bottom - r.bottom), 2),
                      "turn_deg": round(fit["turn_deg"]) if fit["turn_determined"] else None,
                      "shape_fit": fit["fitness"], "shape_rmse_m": fit["rmse_m"], "match_cost": round(cost, 2)},
             "evidence": {"free_views": min(r.views, a.views), "free_share": round(min(r.score, a.score), 2),
                          "noise_margin": round(min(r.margin, a.margin), 1)},
             "frames": {"A": A.paths[j] if j >= 0 else None, "B": B.paths[k] if k >= 0 else None},
             "times": {"A": float(A.times[j]) if j >= 0 else None, "B": float(B.times[k]) if k >= 0 else None},
             "crops": {"before": _save(before, crops / f"c{n:02d}_before.jpg"), "after": _save(after, crops / f"c{n:02d}_after.jpg")}}
        changes.append(c)
    for c in changes:
        c["sentence"] = sentence(c)
        problems = check_sentence(c)
        if problems:
            raise AssertionError("; ".join(problems))
    return changes
