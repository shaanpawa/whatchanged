"""A results video for any before/after pair, made by the code (no hand editing).

  whatchanged video ROOM_OUT --detections RUN --out results.mp4

Four parts, each saved on its own and then joined with short crossfades (1920x1080, 30 fps, H.264, no audio):

  1. Title card: "What changed in the room", one summary line and the change count. Claude Opus 5.5 (effort
     low) may write the summary line from the run's measured facts; every number it writes must appear in
     those facts, or the line falls back to a template. Nothing else in the video comes from a model.
  2. The real phone footage, Before and After side by side, with the change boxes drawn in 3D on every frame
     (SpatialLM style): red removed, green added, blue moved with an arrow; solid where the object is in that
     video, dashed where it is in the other one. Every frame's camera comes from the shared COLMAP model:
     its posed keyframes, plus extra frames registered into it with `colmap image_registrator`. Between two
     poses only a short, calm gap is interpolated; where the pose is unknown the boxes fade out, so no box is
     ever drawn from a guessed camera. Frames are undistorted with the model's own lens, so a box edge is a
     straight line exactly as in the splats. Only stretches where the changed spot is in view are shown.
  3. The 3D walkthrough (`index.html`, the same tour as ?tour=changes) rendered in headless Chrome on a virtual
     clock: every frame is stepped by exactly 1/30 s, so the timing does not depend on the machine.
  4. End card: every change's sentence and how the result was made.

With no changes the title says "No changes found" and parts 2 and 3 show the detected objects instead.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
BG = (11, 11, 12)
TEXT, DIM, SOFT = (236, 236, 236), (154, 154, 159), (208, 208, 212)
ACCENT = (242, 193, 78)
CHANGE_RGB = {"removed": (255, 90, 90), "added": (76, 217, 123), "moved": (90, 169, 255)}
# The viewer's object palette, in the same order, so a class has the same colour in the video and the viewer.
PALETTE = ["#f5b841", "#a78bfa", "#5eead4", "#f472b6", "#bef264", "#cbd5e1", "#e879f9", "#fde047", "#7dd3fc", "#fdba74",
           "#c4b5fd", "#d6c7a1", "#fda4af", "#67e8f9", "#c8a27a", "#fb923c"]
LABEL = {"A": "Before", "B": "After", "room": "Room"}
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
CLAUDE_MODEL = os.environ.get("WHATCHANGED_CLAUDE_MODEL", "claude-opus-5-5")
SUMMARY_VERSION = "s1"
# What morning.sh counts as a heavy job; COLMAP registration waits politely (nice) if one is running.
HEAVY = r"(^|/)colmap |bin/whatchanged run|bin/whatchanged room|-m whatchanged|msplat|mapanything|dense_depth"


def _rgb(hex_: str) -> tuple[int, int, int]:
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5))


# ---------------------------------------------------------------- text and drawing helpers

_FONTS = {
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0), ("/System/Library/Fonts/Helvetica.ttc", 0),
                ("DejaVuSans.ttf", 0), ("Arial.ttf", 0)],
    "medium": [("/System/Library/Fonts/HelveticaNeue.ttc", 10), ("/System/Library/Fonts/Helvetica.ttc", 0),
               ("DejaVuSans.ttf", 0), ("Arial.ttf", 0)],
    "bold": [("/System/Library/Fonts/HelveticaNeue.ttc", 1), ("/System/Library/Fonts/Helvetica.ttc", 1),
             ("DejaVuSans-Bold.ttf", 0), ("Arial Bold.ttf", 0)],
}


@functools.lru_cache(maxsize=None)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for path, index in _FONTS[weight]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: float, max_lines: int = 99) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + "…", font=f) > width:
            lines[-1] = lines[-1].rsplit(" ", 1)[0] if " " in lines[-1] else lines[-1][:-1]
        lines[-1] += "…"
    return lines


def pill(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, f, fill, fg=(17, 17, 17), pad=(12, 5), r=8,
         anchor: str = "ls") -> tuple[float, float, float, float]:
    """A rounded tag; xy is its left-baseline (anchor 'ls') or bottom-centre (anchor 'mb') corner. Returns its box."""
    tw = draw.textlength(text, font=f)
    asc, desc = f.getmetrics()
    w, h = tw + 2 * pad[0], asc + desc + 2 * pad[1]
    if anchor == "mb":
        x0, y0 = xy[0] - w / 2, xy[1] - h
    else:
        x0, y0 = xy[0], xy[1] - asc - pad[1]
    draw.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=r, fill=fill)
    draw.text((x0 + pad[0], y0 + pad[1]), text, font=f, fill=fg)
    return (x0, y0, x0 + w, y0 + h)


def text_arrows(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, f, fill) -> float:
    """Draw text (top-left at xy) with every "→" drawn as a vector arrow, since most fonts here lack the glyph.
    Returns the width drawn."""
    x, y = xy
    asc, _ = f.getmetrics()
    size = f.size
    for i, part in enumerate(text.split("→")):
        if i:
            w = 0.9 * size
            ym = y + asc * 0.62
            lw = max(2, round(size / 13))
            draw.line((x + 0.12 * size, ym, x + w - 0.12 * size, ym), fill=fill, width=lw)
            draw.line((x + w - 0.36 * size, ym - 0.24 * size, x + w - 0.1 * size, ym, x + w - 0.36 * size, ym + 0.24 * size),
                      fill=fill, width=lw, joint="curve")
            x += w
        draw.text((x, y), part, font=f, fill=fill)
        x += draw.textlength(part, font=f)
    return x - xy[0]


def text_arrows_width(draw: ImageDraw.ImageDraw, text: str, f) -> float:
    parts = text.split("→")
    return sum(draw.textlength(t, font=f) for t in parts) + 0.9 * f.size * (len(parts) - 1)


class FrameWriter:
    """RGB frames into an H.264 mp4 through an ffmpeg pipe."""

    def __init__(self, path: Path, size=(W, H), fps: int = FPS, crf: int = 18):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.size, self.n = path, size, 0
        cmd = ["nice", "-n", "5", "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{size[0]}x{size[1]}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame) -> None:
        a = np.asarray(frame.convert("RGB") if isinstance(frame, Image.Image) else frame, dtype=np.uint8)
        assert a.shape[:2] == (self.size[1], self.size[0]), a.shape
        self.proc.stdin.write(np.ascontiguousarray(a).tobytes())
        self.n += 1

    def close(self) -> float:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")
        return self.n / FPS

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.close()
        else:
            self.proc.kill()


def still_clip(img: Image.Image, seconds: float, path: Path) -> float:
    with FrameWriter(path) as w:
        for _ in range(round(seconds * FPS)):
            w.write(img)
    return round(seconds * FPS) / FPS


# ---------------------------------------------------------------- the scene (viewer frame: metres, y up, floor y=0)

def load_scene(room_out: Path) -> dict:
    p = room_out / "scene.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: run `whatchanged room A B --out {room_out} --detections RUN` first")
    return json.loads(p.read_text())


def box_corners(b: dict) -> np.ndarray:
    """The 8 corners of a viewer box, exactly as room.html builds it (three.js rotation.y = yaw), index bits = (x, y, z)."""
    sx, sy, sz = (max(v, 0.02) / 2 for v in b["size"])
    th = math.radians(b.get("yaw_deg", 0.0))
    Ry = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    local = np.array([[sx if i & 1 else -sx, sy if i & 2 else -sy, sz if i & 4 else -sz] for i in range(8)])
    return local @ Ry.T + np.asarray(b["centre"], float)


def keep_object(o: dict) -> bool:
    return o.get("verified") is not False and o.get("keep") is not False


def object_name(o: dict) -> str:
    return o.get("name") or o.get("label") or "object"


def class_colours(scene: dict) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for n in scene.get("splats", {}):
        for o in (scene.get("objects") or {}).get(n, []):
            if keep_object(o):
                out.setdefault(object_name(o), _rgb(PALETTE[len(out) % len(PALETTE)]))
    return out


def change_boxes(c: dict, after: bool) -> list[tuple[dict, bool]]:
    """(box, solid) pairs for change c in one video: solid where the object is in that video, ghost where it is in the other."""
    if c["type"] == "moved":
        now, then = (c["after"], c["before"]) if after else (c["before"], c["after"])
        return [(now, True), (then, False)]
    return [(c, (c["type"] == "removed") != after)]


# ---------------------------------------------------------------- facts, numbers and the summary line

def change_facts(scene: dict, run: Path | None) -> dict:
    """What the summary may use: the run's measured facts (or, without a run, the scene's boxes), numbers pre-rounded."""
    names = list(scene.get("splats", {}))
    facts: dict = {"videos": {LABEL.get(n, n).lower(): Path(v).name for n, v in (scene.get("meta", {}).get("videos") or {}).items()}}
    run_changes = None
    if run and (run / "changes.json").exists():
        run_changes = {c["id"]: c for c in json.loads((run / "changes.json").read_text()).get("changes", [])}
    items = []
    for c in scene.get("changes") or []:
        f: dict = {"id": c["id"], "type": c["type"], "object": c["label"], "sentence": c.get("sentence", "")}
        r = (run_changes or {}).get(c["id"])
        if r:
            if r.get("colour"):
                f["colour"] = r["colour"]
            f["size_cm"] = [int(round(v * 100, -1)) for v in r["size_m"]]
            for k in ("on_before", "on_after", "near_before", "near_after"):
                if r.get(k):
                    f[k] = r[k]
            if r.get("move"):
                f["moved_m"] = round(r["move"]["distance_m"], 1)
                if r["move"].get("turn_deg") is not None and abs(r["move"]["turn_deg"]) >= 20:
                    f["turned_deg"] = int(abs(round(r["move"]["turn_deg"], -1)))
        elif c["type"] == "moved":
            f["moved_m"] = round(float(np.linalg.norm(np.subtract(c["after"]["centre"], c["before"]["centre"])[[0, 2]])), 1)
        items.append(f)
    facts["change_count"] = len(items)
    facts["by_type"] = {t: sum(f["type"] == t for f in items) for t in ("removed", "added", "moved") if any(f["type"] == t for f in items)}
    facts["changes"] = items
    objs = scene.get("objects") or {}
    if objs:
        facts["objects_found"] = {LABEL.get(n, n).lower(): sum(keep_object(o) for o in objs.get(n, [])) for n in names}
    return facts


_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
              "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
              "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
              "single": 1, "dozen": 12, "dozens": 12, "half": 0.5, "quarter": 0.25, "twice": 2, "thrice": 3}


def allowed_numbers(facts) -> set[float]:
    """Every number that appears anywhere in the facts, including inside their sentences and names."""
    out: set[float] = set()

    def walk(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            out.add(float(x))
        elif isinstance(x, str):
            out.update(float(m) for m in re.findall(r"\d+(?:\.\d+)?", x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(facts)
    return out


def check_numbers(text: str, facts) -> list[str]:
    """Like semantics.check_sentence: every number in the text (digits or a number word) must come from the facts."""
    allowed = allowed_numbers(facts)
    fact_words = set(re.findall(r"[a-z]+", json.dumps(facts).lower()))   # "double bed", "seen from two sides"
    problems = []
    for m in re.findall(r"\d+(?:[.,]\d+)?", text):
        v = float(m.replace(",", "."))
        if not any(abs(v - a) < 1e-6 for a in allowed):
            problems.append(f"number {m} is not in the facts")
    for w in re.findall(r"[a-z]+", text.lower()):
        v = _NUM_WORDS.get(w)
        if v is None or w in fact_words:
            continue
        if w not in ("one", "single") and not any(abs(v - a) < 1e-6 for a in allowed):
            problems.append(f"number word '{w}' is not in the facts")
        if w in ("one", "single") and facts.get("change_count") != 1 and 1.0 not in allowed:
            problems.append(f"number word '{w}' is not in the facts")
    return problems


def _move_text(d: float) -> str:
    return f"{d:.1f} m" if d >= 1 else f"{int(round(d * 100))} cm"


def template_summary(facts: dict) -> str:
    items = facts.get("changes", [])
    if not items:
        of = facts.get("objects_found")
        if of and len(of) == 2:
            (a, na), (b, nb) = of.items()
            return f"Nothing was moved, added or removed: {na} objects found {a}, {nb} {b}."
        return "Nothing was moved, added or removed between the two videos."

    def part(f):
        if f["type"] == "moved":
            return f"{f['object']} moved {_move_text(f['moved_m'])}" if "moved_m" in f else f"{f['object']} moved"
        return f"{f['object']} {f['type']}"
    n = len(items)
    return f"{n} change{'s' if n > 1 else ''}: " + ", ".join(part(f) for f in items) + "."


SUMMARY_PROMPT = """You write the one-line summary on the title card of a results video. A 3D change-detection system compared
two phone videos of the same room (before and after) and measured these facts:

{facts}

Write ONE plain sentence, at most 22 words, telling a viewer what changed in the room.
Rules:
- Use only these facts. Name objects exactly as the facts name them.
- Every number you write must appear in the facts exactly as written there: no unit conversions, no rounding, no sums.
  Prefer words over numbers when unsure, but never write a number word (two, three...) whose value is not in the facts.
- No speculation about why, no praise, no mention of AI, models or the system itself. No quotes, no emoji.
- If there are no changes, say that nothing was moved, added or removed."""

SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"],
                  "additionalProperties": False}


def claude_summary(facts: dict) -> tuple[str | None, str]:
    """(summary or None, why) from Claude Opus 5.5 at effort low, using the same call pattern as semantics.py."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=CLAUDE_MODEL, max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SUMMARY_SCHEMA}},
        messages=[{"role": "user", "content": SUMMARY_PROMPT.format(facts=json.dumps(facts, indent=1))}],
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None, f"stop_reason {resp.stop_reason}"
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None, "no text"
    s = " ".join(str(json.loads(text).get("summary", "")).split())
    u = resp.usage
    return s, f"{getattr(u, 'input_tokens', 0)} in / {getattr(u, 'output_tokens', 0)} out tokens"


def summary_line(facts: dict, cache_path: Path, use_claude: bool = True, log=print) -> dict:
    """{'text', 'source'}: Claude's line if it passes the number check, else the template. Cached on disk."""
    key = hashlib.sha1(f"{SUMMARY_VERSION}|{CLAUDE_MODEL}|{json.dumps(facts, sort_keys=True)}".encode()).hexdigest()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if use_claude and key in cache:
        return cache[key]
    fallback = {"text": template_summary(facts), "source": "template"}
    if not use_claude:
        return fallback
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log("  summary: no ANTHROPIC_API_KEY, using the template")
        return fallback
    try:
        text, why = claude_summary(facts)
    except Exception as e:  # noqa: BLE001  (a failed call must not lose the video)
        log(f"  summary: Claude call failed ({type(e).__name__}); using the template")
        return fallback
    if not text:
        log(f"  summary: Claude gave no line ({why}); using the template")
        return fallback
    problems = check_numbers(text, facts) + (["longer than 200 characters"] if len(text) > 200 else [])
    if problems:
        log(f"  summary: Claude's line failed the check ({'; '.join(problems)}); using the template")
        return {**fallback, "rejected": text, "problems": problems}
    out = {"text": text, "source": "claude", "model": CLAUDE_MODEL, "usage": why}
    cache[key] = out
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1))
    log(f"  summary: Claude Opus 5.5 ({why}), numbers checked")
    return out


def provenance(scene: dict, run: Path | None) -> dict:
    """What the end card may claim, read from the outputs rather than assumed."""
    al = scene.get("meta", {}).get("alignment", {})
    named_by_claude = False
    if run and (run / "objects.json").exists():
        o = json.loads((run / "objects.json").read_text())
        named_by_claude = o.get("stats", {}).get("naming", {}).get("label_source") == "claude" or any(
            x.get("label_source") == "claude" for xs in o.get("sessions", {}).values() for x in xs)
    if run and (run / "changes.json").exists() and not named_by_claude:
        cs = json.loads((run / "changes.json").read_text()).get("changes", [])
        named_by_claude = any(c.get("labeller") == "claude" for c in cs)
    return {"shared_model": bool(al.get("shared_model")), "names_by_claude": named_by_claude, "has_run": bool(run)}


# ---------------------------------------------------------------- cards
# The cards match the viewer's glass: a dark room lit by soft fields in the colours of the changes found, seen
# through one frosted panel with a bright rim along its top edge. Geist, the viewer's face, when it is installed.

_CARD_FONTS = {
    "regular": [(str(Path.home() / "Library/Fonts/Geist-Regular.ttf"), 0), ("/Library/Fonts/Geist-Regular.ttf", 0)],
    "medium": [(str(Path.home() / "Library/Fonts/Geist-Regular.ttf"), 0), ("/Library/Fonts/Geist-Regular.ttf", 0)],
    "bold": [(str(Path.home() / "Library/Fonts/Geist-Bold.ttf"), 0), ("/Library/Fonts/Geist-Bold.ttf", 0)],
}
AMBER = (255, 200, 87)
INK, INK2 = (246, 247, 249), (206, 209, 215)


@functools.lru_cache(maxsize=None)
def card_font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    for path, index in _CARD_FONTS[weight]:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return font(weight, size)


def _round_mask(size: tuple[int, int], box, r: float, width: int = 0) -> Image.Image:
    """An anti-aliased rounded rectangle (filled, or an outline `width` px wide) as an L mask."""
    k = 3
    m = Image.new("L", (size[0] * k, size[1] * k), 0)
    x0, y0, x1, y1 = (v * k for v in box)
    ImageDraw.Draw(m).rounded_rectangle((x0, y0, x1, y1), radius=r * k, fill=None if width else 255,
                                        outline=255 if width else None, width=width * k)
    return m.resize(size, Image.LANCZOS)


def _vgrad(h: int, stops: list[tuple[float, float]]) -> np.ndarray:
    """A column of h values interpolated from (position 0..1, value) stops."""
    return np.interp(np.linspace(0, 1, h), [p for p, _ in stops], [v for _, v in stops])


def card_backdrop(colours: list[tuple[int, int, int]]) -> Image.Image:
    """Near-black with a few large, very soft fields of light behind where the glass will be."""
    img = Image.new("RGB", (W, H), (10, 11, 13))
    light = Image.new("RGB", (W, H), (0, 0, 0))
    d = ImageDraw.Draw(light)
    spots = [(0.26, 0.26), (0.78, 0.76), (0.6, 0.2), (0.2, 0.8)]
    for (u, v), c in zip(spots, colours or [(91, 100, 114), AMBER]):
        cx, cy, rr = u * W, v * H, 0.22 * W
        d.ellipse((cx - rr, cy - 0.7 * rr, cx + rr, cy + 0.7 * rr), fill=tuple(int(0.4 * x) for x in c))
    light = light.filter(ImageFilter.GaussianBlur(190))
    return Image.fromarray(np.clip(np.asarray(img, np.int16) + np.asarray(light, np.int16), 0, 255).astype(np.uint8))


def glass(img: Image.Image, box, r: float = 48, tint: float = 0.42, blur: float = 48, shadow: bool = True) -> None:
    """Frost the part of img under box in place: blurred and saturated, darkened by `tint`, a faint sheen from the
    top, a bright rim at the top edge fading to a soft one at the bottom, and a soft shadow underneath."""
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    w, h = x1 - x0, y1 - y0
    if shadow:
        sh = _round_mask(img.size, (x0 + 10, y0 + 28, x1 - 10, y1 + 28), r).filter(ImageFilter.GaussianBlur(38))
        img.paste((0, 0, 0), mask=sh.point(lambda v: int(v * 0.6)))
    pad = int(blur * 2)
    big = img.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad)).filter(ImageFilter.GaussianBlur(blur))
    region = ImageEnhance.Color(big.crop((pad, pad, pad + w, pad + h))).enhance(1.7)
    a = np.asarray(region, np.float32)
    a = a * (1 - tint) + np.array([20, 22, 27], np.float32) * tint
    sheen = _vgrad(h, [(0, 0.11), (0.42, 0.025), (1, 0.045)])[:, None, None]
    a = a * (1 - sheen) + 255 * sheen
    glow = np.exp(-np.arange(h) / 26.0)[:, None, None] * 0.10                   # light caught just inside the top edge
    a = a * (1 - glow) + 255 * glow
    region = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    img.paste(region, (x0, y0), _round_mask((w, h), (0, 0, w - 1, h - 1), r))
    rim = np.asarray(_round_mask((w, h), (0, 0, w - 1, h - 1), r, width=2), np.float32) / 255
    rim *= _vgrad(h, [(0, 0.62), (0.08, 0.3), (0.5, 0.1), (0.92, 0.12), (1, 0.22)])[:, None]
    img.paste((255, 255, 255), (x0, y0), Image.fromarray((rim * 255).astype(np.uint8)))


def chip(img: Image.Image, xy: tuple[float, float], text: str, f, dot=None, h: int = 64, pad: int = 26) -> float:
    """A small glass capsule at top-left xy with an optional coloured light before the text. Returns its right edge."""
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=f)
    dw = 0 if dot is None else 30
    x0, y0 = xy
    x1 = x0 + 2 * pad + dw + tw
    glass(img, (x0, y0, x1, y0 + h), r=h / 2, tint=0.3, blur=20, shadow=False)
    d = ImageDraw.Draw(img)
    if dot is not None:
        cy, cx, rr = y0 + h / 2, x0 + pad + 8, 8
        halo = Image.new("L", img.size, 0)
        ImageDraw.Draw(halo).ellipse((cx - 2 * rr, cy - 2 * rr, cx + 2 * rr, cy + 2 * rr), fill=80)
        img.paste(dot, mask=halo.filter(ImageFilter.GaussianBlur(7)))
        d = ImageDraw.Draw(img)
        d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=dot)
    d.text((x0 + pad + dw, y0 + h / 2), text, font=f, fill=INK, anchor="lm")
    return x1


def title_card(facts: dict, summary: dict) -> Image.Image:
    n = facts.get("change_count", 0)
    by_type = facts.get("by_type", {})
    img = card_backdrop([CHANGE_RGB[t] for t, k in by_type.items() if k] if n else [])
    x0, y0, x1, y1 = 130, 150, W - 130, 900
    glass(img, (x0, y0, x1, y1), r=60)
    d = ImageDraw.Draw(img)
    x = x0 + 90
    d.text((x, y0 + 92), "Two phone videos of one room, before and after", font=card_font("regular", 34), fill=INK2)
    d.text((x, y0 + 146), "What changed in the room" if n else "No changes found", font=card_font("bold", 112), fill=INK)
    y = y0 + 318
    for line in wrap(d, summary["text"], card_font("regular", 46), x1 - x0 - 180, max_lines=3):
        d.text((x, y), line, font=card_font("regular", 46), fill=INK2)
        y += 62
    y = max(y + 54, y0 + 560)
    big = card_font("bold", 96)
    d.text((x, y + 52), str(n), font=big, fill=AMBER, anchor="ls")
    cx = x + d.textlength(str(n), font=big) + 20
    d.text((cx, y + 52), "change" + ("" if n == 1 else "s"), font=card_font("regular", 44), fill=INK, anchor="ls")
    cx += d.textlength("changes", font=card_font("regular", 44)) + 48
    for t, k in by_type.items():
        cx = chip(img, (cx, y - 14), f"{k} {t}", card_font("regular", 34), dot=CHANGE_RGB[t]) + 16
    d = ImageDraw.Draw(img)
    vids = facts.get("videos", {})
    foot = "    ".join(f"{k.capitalize()}: {v}" for k, v in vids.items())
    d.text((x0 + 12, H - 110), foot, font=card_font("regular", 26), fill=INK2)
    src = ("Summary line written by Claude Opus 5.5 from the measured facts; its numbers are checked"
           if summary.get("source") == "claude" else "Summary line filled in from the measured facts")
    d.text((x1 - 12, H - 110), src, font=card_font("regular", 26), fill=INK2, anchor="ra")
    return img


def end_card(scene: dict, facts: dict, prov: dict) -> Image.Image:
    changes = scene.get("changes") or []
    img = card_backdrop(list(dict.fromkeys(CHANGE_RGB.get(c["type"], TEXT) for c in changes)))
    d = ImageDraw.Draw(img)
    x0, x1 = 130, W - 130
    x = x0 + 80
    size = 42 if len(changes) <= 3 else 36 if len(changes) <= 5 else 31 if len(changes) <= 7 else 26
    f, fb = card_font("regular", size), card_font("bold", size - 8)
    gap = 18 if len(changes) <= 5 else 12
    rows, h = [], 0                     # each change's tag width, wrapped sentence and row height
    for c in changes:
        tw = d.textlength(f"{c['id']}  {c['type']}", font=fb) + 2 * 16
        lines = wrap(d, c.get("sentence") or f"{c['label']} {c['type']}", f, x1 - x - tw - 120,
                     max_lines=2 if len(changes) <= 4 else 1)
        rh = max(1, len(lines)) * (size + 12) + (30 if len(changes) <= 4 else 20)
        if h + rh > 510:             # keeps the panel clear of the footer
            break
        rows.append((c, tw, lines, rh))
        h += rh + gap
    more = len(changes) - len(rows)
    h += 50 if more else 0
    if changes:                         # the panel fits its rows, centred above the footer
        ph = max(210 + h + 50, 470)
        y0 = max(50, 90 + (760 - ph) // 2)
        y1 = y0 + ph
    else:
        y0, y1 = 90, 850
    glass(img, (x0, y0, x1, y1), r=60)
    d = ImageDraw.Draw(img)
    d.text((x, y0 + 70), "What changed" if changes else "No changes found", font=card_font("bold", 84), fill=INK)
    y = y0 + 210
    for c, tw, lines, rh in rows:
        col = CHANGE_RGB.get(c["type"], TEXT)
        glass(img, (x - 24, y, x1 - 56, y + rh), r=22, tint=0.25, blur=16, shadow=False)
        d = ImageDraw.Draw(img)
        ty = y + (rh - size - 6) / 2 if len(lines) == 1 else y + 16
        d.rounded_rectangle((x, ty, x + tw, ty + size + 6), radius=(size + 6) / 2, fill=col)
        d.text((x + tw / 2, ty + (size + 6) / 2 + 2), f"{c['id']}  {c['type']}", font=fb, fill=(17, 18, 20), anchor="mm")
        for i, line in enumerate(lines):
            d.text((x + tw + 28, y + (15 if len(changes) <= 4 else 10) + i * (size + 12)), line, font=f, fill=INK)
        y += rh + gap
    if more:
        d.text((x, y + 4), f"and {more} more, listed in the report", font=card_font("regular", 30), fill=INK2)
    if not changes:
        of = facts.get("objects_found")
        d.text((x, y), "Nothing was moved, added or removed between the two videos.", font=card_font("regular", 42), fill=INK2)
        y += 100
        if of:
            cx = x
            for k, v in of.items():
                d.text((cx, y + 84), str(v), font=card_font("bold", 96), fill=AMBER, anchor="ls")
                cx += d.textlength(str(v), font=card_font("bold", 96)) + 18
                d.text((cx, y + 84), f"objects found {k}", font=card_font("regular", 36), fill=INK, anchor="ls")
                cx += d.textlength(f"objects found {k}", font=card_font("regular", 36)) + 90
            y += 150
        seen: list[str] = []
        names = list(scene.get("splats", {}))
        for o in sorted((scene.get("objects") or {}).get(names[-1] if names else "", []), key=lambda o: -float(np.prod(o["size"]))):
            if keep_object(o) and object_name(o) not in seen:
                seen.append(object_name(o))
        if seen:
            for line in wrap(d, "In the 3D model: " + ", ".join(seen) + ".", card_font("regular", 32), x1 - x - 80, max_lines=4):
                d.text((x, y), line, font=card_font("regular", 32), fill=INK2)
                y += 46
    how = ["Both videos → one shared 3D model" if prov.get("shared_model") else "Both videos → one 3D frame",
           "detection on measured geometry"]
    if prov.get("names_by_claude"):
        how.append("names checked by Claude Opus 5.5")
    line, size = "   ·   ".join(how), 34
    while text_arrows_width(d, line, card_font("regular", size)) > W - 2 * x0 and size > 20:
        size -= 1
    text_arrows(d, (x0 + 12, 912), line, card_font("regular", size), INK)
    d.text((x0 + 12, 976), "Made from the phone videos by the whatchanged video command; the boxes on the footage use the same 3D model.",
           font=card_font("regular", 24), fill=INK2)
    return img



# ---------------------------------------------------------------- per-frame camera poses

def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def slerp_R(Ra: np.ndarray, Rb: np.ndarray, a: float) -> np.ndarray:
    """Rotation a of the way from Ra to Rb (constant angular speed)."""
    from scipy.spatial.transform import Rotation, Slerp
    return Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([Ra, Rb])))([a]).as_matrix()[0]


@dataclass
class PoseTrack:
    """Known cameras of one video in the viewer frame: x_cam = R @ X + t (OpenCV axes, metres), sorted by video time."""
    t: np.ndarray
    R: np.ndarray
    C: np.ndarray
    source: list[str] = field(default_factory=list)   # "key" (COLMAP keyframe) or "reg" (registered for the video)

    @staticmethod
    def build(samples: list[tuple[float, np.ndarray, np.ndarray, str]]) -> "PoseTrack":
        samples = sorted(samples, key=lambda s: s[0])
        dedup = []
        for s in samples:
            if dedup and abs(s[0] - dedup[-1][0]) < 1e-4:
                if s[3] == "key":
                    dedup[-1] = s
                continue
            dedup.append(s)
        t = np.array([s[0] for s in dedup], float)
        R = np.stack([s[1] for s in dedup]) if dedup else np.zeros((0, 3, 3))
        C = np.stack([s[2] for s in dedup]) if dedup else np.zeros((0, 3))
        return PoseTrack(t, R, C, [s[3] for s in dedup])

    def drop_jumps(self, max_speed: float = 1.5, max_turn: float = 150.0) -> int:
        """Drop registered poses that jump away from BOTH neighbours (a PnP failure); keyframes are trusted."""
        bad = []
        for i in range(len(self.t)):
            if self.source[i] != "reg":
                continue
            jumps = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(self.t):
                    dt = max(abs(self.t[j] - self.t[i]), 1e-3)
                    jumps.append(np.linalg.norm(self.C[j] - self.C[i]) / dt > max_speed
                                 or _rot_angle_deg(self.R[i], self.R[j]) / dt > max_turn)
            if jumps and all(jumps):
                bad.append(i)
        keep = np.setdiff1d(np.arange(len(self.t)), bad)
        self.t, self.R, self.C = self.t[keep], self.R[keep], self.C[keep]
        self.source = [self.source[i] for i in keep]
        return len(bad)

    def at(self, times, max_gap: float = 0.4, max_turn_deg: float = 10.0, max_move_m: float = 0.3,
           snap: float = 0.5 / FPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(R, t, ok) at each time. A pose is interpolated only inside a short, calm gap between two known poses
        (slerp + lerp of the centre); anything else is unknown (ok False) and must not be drawn."""
        times = np.atleast_1d(np.asarray(times, float))
        Rs, ts, ok = np.tile(np.eye(3), (len(times), 1, 1)), np.zeros((len(times), 3)), np.zeros(len(times), bool)
        n = len(self.t)
        for k, tau in enumerate(times):
            if n == 0:
                continue
            j = int(np.argmin(np.abs(self.t - tau)))
            if abs(self.t[j] - tau) <= snap:
                R, C = self.R[j], self.C[j]
            else:
                i = int(np.searchsorted(self.t, tau))
                if i == 0 or i >= n:
                    continue
                a, b = i - 1, i
                gap = self.t[b] - self.t[a]
                if gap > max_gap or _rot_angle_deg(self.R[a], self.R[b]) > max_turn_deg \
                        or np.linalg.norm(self.C[b] - self.C[a]) > max_move_m:
                    continue
                w = (tau - self.t[a]) / gap
                R = slerp_R(self.R[a], self.R[b], w)
                C = (1 - w) * self.C[a] + w * self.C[b]
            Rs[k], ts[k], ok[k] = R, -R @ C, True
        return Rs, ts, ok


def fade_weights(ok: np.ndarray, fps: float = FPS, fade_s: float = 0.25, min_run_s: float = 0.3) -> np.ndarray:
    """Box opacity per frame: 0 where the pose is unknown, ramping up over fade_s away from it.
    Known stretches shorter than min_run_s are dropped too, so a box never flashes on for a frame or two."""
    ok = np.asarray(ok, bool).copy()
    n = len(ok)
    i = 0
    while i < n:
        if ok[i]:
            j = i
            while j < n and ok[j]:
                j += 1
            if (j - i) / fps < min_run_s and not (i == 0 and j == n):
                ok[i:j] = False
            i = j
        else:
            i += 1
    if ok.all():
        return np.ones(n)
    bad = np.flatnonzero(~ok)
    dist = np.array([np.min(np.abs(bad - k)) for k in range(n)], float)
    return np.clip(dist / max(1.0, fade_s * fps), 0, 1)


# ---------------------------------------------------------------- the COLMAP model behind the room

@dataclass
class ColmapModel:
    work: Path                     # has frames.json, images/, dense/ (and colmap.db + sparse/ for registration)
    dense: Path                    # undistorted model (PINHOLE) that the splats were trained on
    sparse: Path | None            # the distorted model the database belongs to
    db: Path | None
    frames: dict                   # frames.json: state -> [{name, video, t}]
    M: np.ndarray                  # COLMAP world -> viewer frame (similarity)
    images: dict = field(default_factory=dict)
    cams_dense: dict = field(default_factory=dict)
    cams_sparse: dict = field(default_factory=dict)


def find_colmap(room_out: Path, run: Path | None, scene: dict, log=print) -> ColmapModel | None:
    from .room3d import read_cameras, read_images
    cands = []
    if run and (run / "cache" / "diag.json").exists():
        c = json.loads((run / "cache" / "diag.json").read_text()).get("colmap")
        if c and c.get("work"):
            cands.append(Path(c["work"]))
    w = room_out / "work"
    if (w / "dense").is_symlink():
        cands.append((w / "dense").resolve().parent)
    if w.exists():
        cands.append(w)
    M = np.array(scene["matrix"], float).reshape(4, 4).T
    for work in cands:
        dense = work / "dense"
        if not (dense / "sparse" / "0" / "images.bin").exists():
            continue
        fj = work / "frames.json" if (work / "frames.json").exists() else room_out / "work" / "frames.json"
        if not fj.exists():
            continue
        imgs = read_images(dense / "sparse" / "0" / "images.bin")
        # Same model as the viewer? The scene's first camera must sit where this model puts it.
        cam0 = next((c for cs in scene.get("cameras", {}).values() for c in cs), None)
        by = {im["name"]: im for im in imgs.values()}
        if cam0 is None or cam0["name"] not in by:
            continue
        from .room3d import cam_to_world
        pos = M[:3, :3] @ cam_to_world(by[cam0["name"]])[:3, 3] + M[:3, 3]
        if np.linalg.norm(pos - np.array(cam0["position"])) > 0.01:
            log(f"  footage: {work} is not the model behind this room ({np.linalg.norm(pos - np.array(cam0['position'])):.2f} m off)")
            continue
        sparse = None
        if (work / "sparse").exists():
            best = 0
            for m in sorted(p for p in (work / "sparse").iterdir() if (p / "images.bin").exists()):
                k = len(read_images(m / "images.bin"))
                if k > best:
                    sparse, best = m, k
        db = work / "colmap.db" if (work / "colmap.db").exists() else None
        cm = ColmapModel(work=work, dense=dense, sparse=sparse, db=db, frames=json.loads(fj.read_text()), M=M, images=imgs,
                         cams_dense=read_cameras(dense / "sparse" / "0" / "cameras.bin"),
                         cams_sparse=read_cameras(sparse / "cameras.bin") if sparse else {})
        return cm
    return None


def viewer_pose(im: dict, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A COLMAP image's camera in the viewer frame: (R, C) with x_cam = R @ (X - C), metres, OpenCV axes."""
    from .room3d import cam_to_world
    s = float(np.cbrt(np.linalg.det(M[:3, :3])))
    Rm = M[:3, :3] / s
    T = cam_to_world(im)
    R = T[:3, :3].T @ Rm.T
    C = M[:3, :3] @ T[:3, 3] + M[:3, 3]
    return R, C


def _K(params, model: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV K (pixel centres at integers) and distortion from a COLMAP camera (pixel centres at +0.5)."""
    p = list(params)
    if model == 0:
        fx = fy = p[0]; cx, cy = p[1:3]; D = [0, 0, 0, 0]
    elif model == 1:
        fx, fy, cx, cy = p[:4]; D = [0, 0, 0, 0]
    elif model == 2:
        fx = fy = p[0]; cx, cy = p[1:3]; D = [p[3], 0, 0, 0]
    elif model == 3:
        fx = fy = p[0]; cx, cy = p[1:3]; D = [p[3], p[4], 0, 0]
    else:
        fx, fy, cx, cy = p[:4]; D = p[4:8]
    return np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]], float), np.array(D, float)


@dataclass
class Lens:
    """One video's lens: video frame (resized to the COLMAP image size) -> undistorted tile, and the tile's K."""
    src_wh: tuple[int, int]
    dense_wh: tuple[int, int]
    tile_wh: tuple[int, int]
    K_tile: np.ndarray
    map1: np.ndarray
    map2: np.ndarray

    def undistort(self, frame_src: np.ndarray) -> np.ndarray:
        und = cv2.remap(frame_src, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return cv2.resize(und, self.tile_wh, interpolation=cv2.INTER_AREA)


def make_lens(cm: ColmapModel, state: str, tile_w: int) -> Lens:
    cid = next(im["camera_id"] for im in cm.images.values() if im["name"].startswith(state + "/"))
    cd = cm.cams_dense[cid]
    Kd, _ = _K(cd["params"], cd["model"])
    dw, dh = int(cd["width"]), int(cd["height"])
    if cm.cams_sparse.get(cid):
        cs = cm.cams_sparse[cid]
        Ks, D = _K(cs["params"], cs["model"])
        sw, sh = int(cs["width"]), int(cs["height"])
    else:   # no distorted model: the frames are taken as already pinhole at the dense size
        Ks, D, sw, sh = Kd, np.zeros(4), dw, dh
    map1, map2 = cv2.initUndistortRectifyMap(Ks, D, None, Kd, (dw, dh), cv2.CV_32FC1)
    th = int(round(tile_w * dh / dw))
    k = tile_w / dw
    K_tile = np.array([[Kd[0, 0] * k, 0, (Kd[0, 2] + 0.5) * k - 0.5], [0, Kd[1, 1] * k, (Kd[1, 2] + 0.5) * k - 0.5], [0, 0, 1]])
    return Lens((sw, sh), (dw, dh), (tile_w, th), K_tile, map1, map2)


def lens_check(cm: ColmapModel, lens: Lens, state: str) -> float | None:
    """Mean absolute difference (0-255) between our undistortion of a keyframe and COLMAP's own undistorted image."""
    for f in cm.frames.get(state, [])[len(cm.frames.get(state, [])) // 2:]:
        a, b = cm.work / "images" / f["name"], cm.dense / "images" / f["name"]
        if a.exists() and b.exists():
            src = cv2.imread(str(a))
            if src.shape[1::-1] != lens.src_wh:
                src = cv2.resize(src, lens.src_wh, interpolation=cv2.INTER_AREA)
            ours = cv2.remap(src, lens.map1, lens.map2, cv2.INTER_LINEAR)
            ref = cv2.imread(str(b))
            if ref.shape != ours.shape:
                return None
            m = 8
            return float(np.mean(np.abs(ours[m:-m, m:-m].astype(int) - ref[m:-m, m:-m].astype(int))))
    return None


# ---------------------------------------------------------------- the videos

def resolve_video(path_str: str, room_out: Path) -> Path | None:
    p = Path(path_str)
    tries = [p] if p.is_absolute() else [Path.cwd() / p, Path(__file__).resolve().parents[2] / p, *[a / p for a in room_out.resolve().parents]]
    return next((q for q in tries if q.exists()), None)


class VideoReader:
    """Frames at given times, on the same clock as frames.json (OpenCV's position after each decoded frame)."""

    def __init__(self, path: Path, size: tuple[int, int], offset: float = 0.0):
        self.path, self.size, self.offset = path, size, offset
        cap = cv2.VideoCapture(str(path))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / self.fps
        cap.release()

    def frames(self, times):
        """Yields (decoded time, BGR frame resized to self.size) for each wanted time (ascending)."""
        times = [t + self.offset for t in times]
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (times[0] - 0.5) * 1000))
        prev = cur = None
        for tau in times:
            while cur is None or cur[0] < tau - 0.5 / self.fps:
                ok, img = cap.read()
                if not ok:
                    break
                prev, cur = cur, (cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, img)
            pick = cur
            if prev is not None and cur is not None and abs(prev[0] - tau) < abs(cur[0] - tau):
                pick = prev
            if pick is None:
                yield None, None
                continue
            img = pick[1]
            if img.shape[1::-1] != self.size:
                img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
            yield pick[0] - self.offset, img
        cap.release()


def measure_sync(reader: VideoReader, keyframes: list[tuple[float, Path]], n: int = 5) -> tuple[float, float]:
    """Seconds to add to frames.json times to land on the decoder's frame (0 when they agree), and the median match error.

    Compares a few COLMAP keyframe images with the decoded frames around their time."""
    if not keyframes:
        return 0.0, float("nan")
    pick = [keyframes[int(i)] for i in np.linspace(0, len(keyframes) - 1, min(n, len(keyframes)))]
    offs, errs = [], []
    small = (192, 108)
    for t, img_path in pick:
        ref = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            continue
        ref = cv2.resize(ref, small, interpolation=cv2.INTER_AREA).astype(np.float32)
        cand = [t + k / reader.fps for k in range(-3, 4)]
        best = (1e9, 0.0)
        for (tt, img), want in zip(reader.frames(cand), cand):
            if img is None:
                continue
            g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), small, interpolation=cv2.INTER_AREA).astype(np.float32)
            e = float(np.mean(np.abs(g - ref)))
            if e < best[0]:
                best = (e, want - t)
        offs.append(best[1])
        errs.append(best[0])
    if not offs:
        return 0.0, float("nan")
    off = float(np.median(offs))
    return (0.0 if abs(off) < 0.5 / reader.fps else off), float(np.median(errs))


# ---------------------------------------------------------------- registering extra frames into the model

def _heavy_running() -> list[str]:
    try:
        out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    except OSError:
        return []
    me = {os.getpid(), os.getppid()}
    rows = []
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        exe = Path(cmd.split(" ", 1)[0]).name if cmd else ""
        if exe in ("zsh", "bash", "sh", "grep", "pgrep", "ps", "tail", "time", "nice") or "whatchanged video" in cmd:
            continue                     # a shell or grep that merely mentions the pattern is not a heavy job
        if pid.isdigit() and int(pid) not in me and re.search(HEAVY, cmd):
            rows.append(cmd[:120])
    return rows


def _colmap(*args) -> None:
    cmd = ["nice", "-n", "10", "colmap", *map(str, args)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        raise RuntimeError(f"colmap {args[0]} failed:\n{tail}")


def register_frames(cm: ColmapModel, wants: dict[str, list[float]], readers: dict[str, VideoReader], reg_dir: Path,
                    log=print) -> dict[str, list[tuple[float, dict]]]:
    """Register video frames at the wanted times into a COPY of the shared COLMAP model (image_registrator, intrinsics
    fixed). Returns state -> [(time, COLMAP image)] for those that registered. Cached in reg_dir."""
    from .room3d import read_images
    if not (cm.db and cm.sparse and shutil.which("colmap")):
        log("  footage: no COLMAP database or binary, so only the model's own keyframes give poses")
        return {}
    plan = {n: [round(t, 3) for t in ts] for n, ts in wants.items() if ts}
    stamp = {"model": str(cm.sparse), "db_mtime": cm.db.stat().st_mtime, "times": plan}
    done = reg_dir / "reg.json"
    out_model = reg_dir / "sparse"
    if not (done.exists() and json.loads(done.read_text()) == stamp and (out_model / "images.bin").exists()):
        busy = _heavy_running()
        if busy:
            log(f"  footage: another heavy job is running ({busy[0]}...); COLMAP runs at low priority")
        shutil.rmtree(reg_dir, ignore_errors=True)
        (reg_dir / "images").mkdir(parents=True)
        t0 = time.time()
        names: dict[str, list[tuple[float, str]]] = {}
        for n, ts in plan.items():
            (reg_dir / "images" / f"v{n}").mkdir()
            names[n] = []
            for (tt, img), want in zip(readers[n].frames(ts), ts):
                if img is None:
                    continue
                name = f"v{n}/{n}_{int(round(want * 1000)):07d}.jpg"
                cv2.imwrite(str(reg_dir / "images" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                names[n].append((want, name))
        db = reg_dir / "colmap.db"
        shutil.copy(cm.db, db)
        sp_imgs = read_images(cm.sparse / "images.bin")
        pairs = set()
        for n, items in names.items():
            if not items:
                continue
            (reg_dir / f"list_{n}.txt").write_text("".join(nm + "\n" for _, nm in items))
            cid = next(im["camera_id"] for im in sp_imgs.values() if im["name"].startswith(n + "/"))
            _colmap("feature_extractor", "--database_path", db, "--image_path", reg_dir / "images",
                    "--image_list_path", reg_dir / f"list_{n}.txt", "--ImageReader.existing_camera_id", cid,
                    "--FeatureExtraction.use_gpu", 0, "--SiftExtraction.max_num_features", 8192)
            posed = {im["name"] for im in sp_imgs.values()}
            keys = sorted((f["t"], f["name"]) for f in cm.frames.get(n, []) if f["name"] in posed)
            kt = np.array([k[0] for k in keys])
            for i, (t, nm) in enumerate(items):
                for j in np.argsort(np.abs(kt - t))[:8]:
                    pairs.add((nm, keys[j][1]))
                for _, nm2 in items[i + 1:i + 3]:
                    pairs.add((nm, nm2))
        (reg_dir / "pairs.txt").write_text("".join(f"{a} {b}\n" for a, b in sorted(pairs)))
        t1 = time.time()
        _colmap("matches_importer", "--database_path", db, "--match_list_path", reg_dir / "pairs.txt",
                "--match_type", "pairs", "--FeatureMatching.use_gpu", 0)
        t2 = time.time()
        out_model.mkdir()
        _colmap("image_registrator", "--database_path", db, "--input_path", cm.sparse, "--output_path", out_model,
                "--Mapper.ba_refine_focal_length", 0, "--Mapper.ba_refine_principal_point", 0, "--Mapper.ba_refine_extra_params", 0)
        db.unlink(missing_ok=True)   # hundreds of MB, and only needed for this step
        done.write_text(json.dumps(stamp))
        log(f"  footage: registered frames in {time.time() - t0:.0f}s (features {t1 - t0:.0f}s, matching {t2 - t1:.0f}s, "
            f"{len(pairs)} pairs)")
    new = read_images(out_model / "images.bin")
    old = {im["name"]: im for im in cm.images.values()}
    # image_registrator must not move the model; if a bundle adjustment did, map the new poses back onto the old ones.
    from .room3d import cam_to_world
    common = [(cam_to_world(old[im["name"]])[:3, 3], cam_to_world(im)[:3, 3]) for im in new.values() if im["name"] in old]
    shift = max((np.linalg.norm(a - b) for a, b in common), default=0.0)
    fix = None
    if shift > 1e-3 and len(common) >= 3:
        A, B = np.array([b for _, b in common]), np.array([a for a, _ in common])
        ca, cb = A.mean(0), B.mean(0)
        U, S, Vt = np.linalg.svd((B - cb).T @ (A - ca))
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        Rf = U @ D @ Vt
        s = float(np.trace(np.diag(S) @ D) / np.sum((A - ca) ** 2))
        fix = (s, Rf, cb - s * Rf @ ca)
        log(f"  footage: the registration moved the model by up to {shift:.4f} units; mapped back")
    out: dict[str, list[tuple[float, dict]]] = {}
    for im in new.values():
        m = re.match(r"v(\w+)/\1_(\d{7})\.jpg$", im["name"])
        if not m:
            continue
        if fix:
            s, Rf, tf = fix
            T = cam_to_world(im)
            Rcw, C = Rf @ T[:3, :3], s * Rf @ T[:3, 3] + tf
            Rwc = Rcw.T
            from scipy.spatial.transform import Rotation
            x, y, z, w = Rotation.from_matrix(Rwc).as_quat()
            im = {**im, "q": (w, x, y, z), "t": tuple(-Rwc @ C)}
        out.setdefault(m.group(1), []).append((int(m.group(2)) / 1000.0, im))
    return out


# ---------------------------------------------------------------- drawing boxes on the footage

def clip_segment(a: np.ndarray, b: np.ndarray, near: float = 0.05) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip a camera-space segment to z >= near (in front of the lens)."""
    za, zb = a[2], b[2]
    if za < near and zb < near:
        return None
    if za < near:
        a = a + (b - a) * (near - za) / (zb - za)
    elif zb < near:
        b = b + (a - b) * (near - zb) / (za - zb)
    return a, b


def _proj(K: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]])


def _line(img, p, q, col, width, dashed=False):
    big = 1 << 14
    ok, p1, q1 = cv2.clipLine((-big, -big, 2 * big, 2 * big), tuple(int(round(v)) for v in p), tuple(int(round(v)) for v in q))
    if not ok:
        return
    ok, p1, q1 = cv2.clipLine((-4, -4, img.shape[1] + 8, img.shape[0] + 8), p1, q1)
    if not ok:
        return
    if not dashed:
        cv2.line(img, p1, q1, col, width, cv2.LINE_AA)
        return
    p1, q1 = np.array(p1, float), np.array(q1, float)
    L = float(np.linalg.norm(q1 - p1))
    on, off = 14.0, 9.0
    s = 0.0
    while s < L:
        e = min(L, s + on)
        a, b = p1 + (q1 - p1) * s / max(L, 1e-6), p1 + (q1 - p1) * e / max(L, 1e-6)
        cv2.line(img, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), col, width, cv2.LINE_AA)
        s = e + off


def draw_box(img, corners, R, t, K, col, width, dashed=False, face=0.0):
    P = corners @ R.T + t
    if face > 0 and (P[:, 2] > 0.1).all():
        uv = np.array([_proj(K, p) for p in P])
        layer = img.copy()
        hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(layer, hull, col, cv2.LINE_AA)
        cv2.addWeighted(layer, face, img, 1 - face, 0, dst=img)
    for i, j in EDGES:
        seg = clip_segment(P[i], P[j])
        if seg:
            _line(img, _proj(K, seg[0]), _proj(K, seg[1]), col, width, dashed)


def draw_arrow(img, a, b, R, t, K, col, width=3):
    pa, pb = R @ a + t, R @ b + t
    seg = clip_segment(pa, pb, 0.1)
    if not seg:
        return
    p, q = _proj(K, seg[0]), _proj(K, seg[1])
    L = float(np.linalg.norm(q - p))
    if L < 4:
        return
    big = 1 << 13
    ok, p1, q1 = cv2.clipLine((-big, -big, 2 * big, 2 * big), tuple(int(round(v)) for v in p), tuple(int(round(v)) for v in q))
    if ok:
        cv2.arrowedLine(img, p1, q1, col, width, cv2.LINE_AA, tipLength=min(0.3, 28 / max(L, 1)))


def _box_top(b: dict) -> np.ndarray:
    c = np.asarray(b["centre"], float).copy()
    c[1] += b["size"][1] / 2
    return c


def render_tile(frame_tile: np.ndarray, R, t, alpha: float, K, scene: dict, state: str, after: bool,
                focus: dict | None, show_objects: bool, colours: dict, object_alpha: float = 0.5,
                max_object_tags: int = 7) -> np.ndarray:
    """One undistorted frame (BGR) with the change boxes (and faint object boxes) drawn from camera (R, t)."""
    if alpha <= 0:
        return frame_tile
    ov = frame_tile.copy()
    th, tw = ov.shape[:2]
    tags = []   # (priority, anchor uv, distance, text, bg colour RGB)
    changes = scene.get("changes") or []
    solid_at = []
    for c in changes:
        for b, solid in change_boxes(c, after):
            if solid:
                solid_at.append(np.asarray(b["centre"], float))
    if show_objects:
        layer = ov.copy()
        for o in (scene.get("objects") or {}).get(state, []):
            if not keep_object(o) or any(np.linalg.norm(np.asarray(o["centre"]) - p) < 0.25 for p in solid_at):
                continue
            rgb = colours.get(object_name(o), (200, 200, 200))
            draw_box(layer, box_corners(o), R, t, K, rgb[::-1], 2, face=0.0)
            top = R @ _box_top(o) + t
            if 0.3 < top[2] < 4.0:
                tags.append((0, _proj(K, top), float(np.linalg.norm(top)), object_name(o), rgb))
        cv2.addWeighted(layer, object_alpha, ov, 1 - object_alpha, 0, dst=ov)
    for c in changes:
        rgb = CHANGE_RGB.get(c["type"], (255, 255, 255))
        hot = focus is None or c["id"] == focus["id"]
        width = 4 if hot else 2
        for b, solid in change_boxes(c, after):
            draw_box(ov, box_corners(b), R, t, K, rgb[::-1], width if solid else max(2, width - 1),
                     dashed=not solid, face=(0.20 if hot else 0.08) if solid else 0.04)
        if c["type"] == "moved":
            draw_arrow(ov, np.asarray(c["before"]["centre"], float), np.asarray(c["after"]["centre"], float), R, t, K, rgb[::-1], 4 if hot else 2)
        now = change_boxes(c, after)[0][0]
        top = R @ _box_top(now) + t
        if top[2] > 0.2:
            tags.append((2 if hot else 1, _proj(K, top), float(np.linalg.norm(top)), f"{c['id']} {c['label']} · {c['type']}", rgb))
    if tags:
        pil = Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        placed = []
        n_obj = 0
        for pri, uv, dist, text, rgb in sorted(tags, key=lambda x: (-x[0], x[2])):
            lo, hi, base = {2: (17, 30, 25), 1: (14, 22, 18), 0: (12, 17, 15)}[pri]
            size = int(np.clip(base * (1.8 / max(dist, 0.3)) ** 0.35, lo, hi))
            f = font("bold" if pri else "medium", size)
            tw_ = d.textlength(text, font=f) + (16 if pri else 10)
            asc, desc = f.getmetrics()
            hh = asc + desc + (8 if pri else 4)
            if not (-tw * 0.1 < uv[0] < tw * 1.1 and -th * 0.1 < uv[1] < th * 1.2):
                continue
            x = float(np.clip(uv[0], tw_ / 2 + 2, tw - tw_ / 2 - 2))
            y = float(np.clip(uv[1] - 4, hh + 2, th - 2))
            r = [x - tw_ / 2, y - hh, x + tw_ / 2, y]
            clash = lambda r: any(r[0] < q[2] and r[2] > q[0] and r[1] < q[3] and r[3] > q[1] for q in placed)  # noqa: E731
            if pri:
                for _ in range(5):
                    if not clash(r):
                        break
                    r = [r[0], r[1] - hh - 2, r[2], r[3] - hh - 2]
            elif clash(r) or n_obj >= max_object_tags:
                continue
            if r[1] < 0:
                continue
            n_obj += pri == 0
            placed.append(r)
            a = 255 if pri == 2 else 225 if pri == 1 else 170
            d.rounded_rectangle(r, radius=5, fill=(*rgb, a))
            d.text(((r[0] + r[2]) / 2, (r[1] + r[3]) / 2), text, font=f, fill=(17, 17, 17, 255 if pri else 220), anchor="mm")
        ov = cv2.cvtColor(np.asarray(Image.alpha_composite(pil, layer).convert("RGB")), cv2.COLOR_RGB2BGR)
    if alpha >= 1:
        return ov
    return cv2.addWeighted(ov, alpha, frame_tile, 1 - alpha, 0)


def view_score(boxes: list[dict], R, t, K, wh) -> float:
    """How well a camera shows these boxes: whole, central, a fair size, 0.7-3.5 m away (no occlusion test)."""
    w, h = wh
    total = 0.0
    for k, b in enumerate(boxes):
        P = box_corners(b) @ R.T + t
        if (P[:, 2] < 0.3).any():
            continue
        uv = np.array([_proj(K, p) for p in P])
        inside = float(np.mean((uv[:, 0] > 0.03 * w) & (uv[:, 0] < 0.97 * w) & (uv[:, 1] > 0.03 * h) & (uv[:, 1] < 0.97 * h)))
        c = _proj(K, R @ np.asarray(b["centre"], float) + t)
        cent = 1 - min(1.0, math.hypot(c[0] / w - 0.5, c[1] / h - 0.5) / 0.7)
        lo, hi = np.clip(uv.min(0), 0, [w, h]), np.clip(uv.max(0), 0, [w, h])
        area = float(np.prod(hi - lo)) / (w * h)
        size = min(1.0, area / 0.05) * (1.0 if area < 0.55 else 0.5)
        d = float(np.linalg.norm(R @ np.asarray(b["centre"], float) + t))
        dist = 1.0 if 0.7 <= d <= 3.5 else 0.4
        total += (1.0 if k == 0 else 0.3) * inside ** 2 * (0.35 + 0.65 * cent) * (0.3 + 0.7 * size) * dist
    return total


def best_window(track: PoseTrack, boxes: list[dict], K, wh, duration: float, video_len: float) -> tuple[float, float]:
    """(start, mean score) of the stretch of `duration` s that shows these boxes best, judged on the known cameras."""
    if len(track.t) == 0:
        return 0.0, 0.0
    sc = np.array([view_score(boxes, track.R[i], -track.R[i] @ track.C[i], K, wh) for i in range(len(track.t))])
    lo, hi = max(0.0, track.t[0]), min(video_len or track.t[-1], track.t[-1]) - duration
    if hi <= lo:
        return max(0.0, lo), float(sc.mean())
    best = (lo, -1.0)
    grid = np.arange(0, duration, 0.1)
    for s in np.arange(lo, hi + 1e-9, 0.1):
        tt = s + grid
        j = np.clip(np.searchsorted(track.t, tt), 1, len(track.t) - 1)
        j = np.where(np.abs(track.t[j - 1] - tt) < np.abs(track.t[j] - tt), j - 1, j)
        near = np.abs(track.t[j] - tt) < 0.6
        v = float(np.mean(np.where(near, sc[j], 0.0)))
        if v > best[1] + 1e-9:
            best = (float(s), v)
    return best


# ---------------------------------------------------------------- part 2: the real footage

def render_footage(room_out: Path, run: Path | None, scene: dict, path: Path, parts: Path, register: bool = True,
                   reg_fps: float = 10.0, show_objects: bool = True, log=print) -> dict | None:
    names = list(scene.get("splats", {}))
    if names != ["A", "B"]:
        log("  footage: skipped (needs a before/after pair)")
        return None
    cm = find_colmap(room_out, run, scene, log)
    if cm is None:
        log("  footage: skipped (no COLMAP model behind this room: pass --detections RUN, or build the room with it)")
        return None
    readers, lenses, tracks, keyframes, stats = {}, {}, {}, {}, {"states": {}}
    tile_w, margin = 912, 36
    for n in names:
        vpath = next((resolve_video(f["video"], room_out) for f in cm.frames.get(n, []) if f.get("video")), None)
        if vpath is None:
            log(f"  footage: skipped (the {LABEL[n]} video is not on this machine)")
            return None
        lenses[n] = make_lens(cm, n, tile_w)
        readers[n] = VideoReader(vpath, lenses[n].src_wh)
        tof = {f["name"]: f["t"] for f in cm.frames.get(n, [])}
        mine = sorted(((tof[im["name"]], im) for im in cm.images.values() if im["name"] in tof), key=lambda x: x[0])
        keyframes[n] = [(t, cm.work / "images" / im["name"]) for t, im in mine]
        tracks[n] = PoseTrack.build([(t, *viewer_pose(im, cm.M), "key") for t, im in mine])
        err = lens_check(cm, lenses[n], n)
        off, merr = measure_sync(readers[n], keyframes[n])
        readers[n].offset = off
        stats["states"][n] = {"video": vpath.name, "keyframes": len(mine), "lens_check_mad": None if err is None else round(err, 2),
                              "sync_offset_s": round(off, 4), "sync_match_mad": round(merr, 2)}
        log(f"  footage {LABEL[n]}: {vpath.name}, {len(mine)} posed keyframes, undistortion vs COLMAP's {err if err is None else round(err, 1)}/255, "
            f"clock offset {off:+.3f} s (match {merr:.1f}/255)")
    changes = scene.get("changes") or []
    # Which stretch of each video to show.
    segs = []
    if changes:
        dur = float(np.clip(24.0 / len(changes), 4.5, 8.0))
        for c in changes:
            seg = {"change": c, "dur": dur}
            for n in names:
                after = n == "B"
                main = change_boxes(c, after)[0][0]
                others = [b for b, _ in change_boxes(c, after)[1:]]
                s, v = best_window(tracks[n], [main, *others], lenses[n].K_tile, lenses[n].tile_wh, dur, readers[n].duration)
                seg[n] = (s, v)
            segs.append(seg)
    else:
        dur = 8.0
        seg = {"change": None, "dur": dur}
        for n in names:
            objs = [o for o in (scene.get("objects") or {}).get(n, []) if keep_object(o)]
            objs.sort(key=lambda o: -float(np.prod(o["size"])))
            best = (0.0, -1.0)
            for anchor in objs[:6]:   # the window that shows a big object and as much else as possible
                s, v = best_window(tracks[n], [anchor, *objs[:12]], lenses[n].K_tile, lenses[n].tile_wh, dur, readers[n].duration)
                if v > best[1]:
                    best = (s, v)
            seg[n] = best
        segs.append(seg)
    # Register more frames inside those stretches, so every output frame has a camera close by.
    if register:
        wants = {n: [] for n in names}
        for seg in segs:
            for n in names:
                s = seg[n][0]
                for t in np.arange(s - 0.3, s + seg["dur"] + 0.3, 1.0 / reg_fps):
                    if t >= 0 and (len(tracks[n].t) == 0 or np.min(np.abs(tracks[n].t - t)) > 0.5 / reg_fps):
                        wants[n].append(float(t))
        try:
            reg = register_frames(cm, {n: sorted(set(round(t, 3) for t in ts)) for n, ts in wants.items()}, readers, parts / "reg", log)
        except RuntimeError as e:
            log(f"  footage: registration failed ({str(e).splitlines()[0]}); keyframes only")
            reg = {}
        for n in names:
            got = reg.get(n, [])
            tracks[n] = PoseTrack.build([(t, R, C, "key") for t, R, C in zip(tracks[n].t, tracks[n].R, tracks[n].C)]
                                        + [(t, *viewer_pose(im, cm.M), "reg") for t, im in got])
            dropped = tracks[n].drop_jumps()
            stats["states"][n].update(registered=len(got), asked=len(wants[n]), dropped_jumps=dropped)
            log(f"  footage {LABEL[n]}: {len(got)}/{len(wants[n])} extra frames registered, {dropped} dropped as jumps")
    colours = class_colours(scene)
    tw, th = lenses["A"].tile_wh
    th_max = max(lenses[n].tile_wh[1] for n in names)
    x0s = {"A": margin, "B": W - margin - tw}
    ytile = 284
    covered, total = 0, 0
    with FrameWriter(path) as fw:
        for si, seg in enumerate(segs):
            c = seg["change"]
            nfr = int(round(seg["dur"] * FPS))
            times = {n: seg[n][0] + np.arange(nfr) / FPS for n in names}
            poses = {n: tracks[n].at(times[n]) for n in names}
            alphas = {n: fade_weights(poses[n][2]) for n in names}
            frames = {n: readers[n].frames(list(times[n])) for n in names}
            base = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(base)
            d.text((margin, 72), "THE REAL PHONE VIDEOS · BOXES DRAWN IN 3D FROM EACH FRAME'S CAMERA", font=font("medium", 26), fill=DIM)
            if c is not None:
                d.text((W - margin, 72), f"{si + 1} / {len(segs)}", font=font("medium", 26), fill=DIM, anchor="ra")
                box = pill(d, (margin, 172), f"{c['id']}  {c['type']}", font("bold", 34), CHANGE_RGB[c["type"]], pad=(14, 7))
                d.text((box[2] + 22, 172), c["label"], font=font("bold", 52), fill=TEXT, anchor="ls")
                sentence = c.get("sentence") or ""
            else:
                d.text((margin, 172), "No changes found", font=font("bold", 52), fill=TEXT, anchor="ls")
                sentence = "Every box is an object the system found; nothing was moved, added or removed."
            for n in names:
                d.text((x0s[n], ytile - 18), LABEL[n], font=font("bold", 32), fill=TEXT, anchor="ls")
            y = ytile + th_max + 72
            for line in wrap(d, sentence, font("regular", 36), W - 2 * (margin), max_lines=2):
                d.text((W // 2, y), line, font=font("regular", 36), fill=TEXT, anchor="ms")
                y += 48
            leg_y = H - 56
            lx = margin
            if c is not None:
                d.line((lx, leg_y - 9, lx + 46, leg_y - 9), fill=SOFT, width=4)
                d.text((lx + 58, leg_y), "solid: the object in this video", font=font("regular", 24), fill=DIM, anchor="ls")
                lx += 58 + d.textlength("solid: the object in this video", font=font("regular", 24)) + 40
                for k in range(3):
                    d.line((lx + k * 17, leg_y - 9, lx + k * 17 + 10, leg_y - 9), fill=SOFT, width=4)
                d.text((lx + 58, leg_y), "dashed: where it is in the other video", font=font("regular", 24), fill=DIM, anchor="ls")
            d.text((W - margin, leg_y), "camera per frame: shared COLMAP model; boxes fade where it is unknown",
                   font=font("regular", 24), fill=DIM, anchor="rs")
            for k in range(nfr):
                canvas = base.copy()
                for n in names:
                    tt, img = next(frames[n])
                    R, tv, okk = poses[n][0][k], poses[n][1][k], poses[n][2][k]
                    a = float(alphas[n][k])
                    total += 1
                    covered += a >= 0.999
                    if img is None:
                        tile = np.zeros((lenses[n].tile_wh[1], lenses[n].tile_wh[0], 3), np.uint8)
                    else:
                        tile = lenses[n].undistort(img)
                        tile = render_tile(tile, R, tv, a, lenses[n].K_tile, scene, n, n == "B", c, show_objects or c is None, colours,
                                           object_alpha=0.5 if c is not None else 0.9, max_object_tags=7 if c is not None else 12)
                    pil = Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB))
                    canvas.paste(pil, (x0s[n], ytile))
                    dd = ImageDraw.Draw(canvas)
                    dd.text((x0s[n] + lenses[n].tile_wh[0], ytile - 18), f"{tt:5.1f} s" if tt is not None else "",
                            font=font("regular", 24), fill=DIM, anchor="rs")
                    if a <= 0:
                        pill(dd, (x0s[n] + 12, ytile + lenses[n].tile_wh[1] - 16), "camera pose unknown here: boxes hidden",
                             font("regular", 20), (16, 16, 18), fg=(225, 225, 228), pad=(10, 4), r=6)
                fade = min(1.0, (k + 1) / 8, (nfr - k) / 8)
                if fade < 1:
                    canvas = Image.blend(Image.new("RGB", (W, H), BG), canvas, fade)
                fw.write(canvas)
            seg_stats = {n: {"start_s": round(seg[n][0], 2), "view_score": round(seg[n][1], 3),
                             "boxes_shown": round(float(np.mean(alphas[n] >= 0.999)), 3)} for n in names}
            stats.setdefault("segments", []).append({"change": c["id"] if c else None, "duration_s": seg["dur"], **seg_stats})
            log(f"  footage segment {c['id'] if c else 'objects'}: " + ", ".join(
                f"{LABEL[n]} {seg[n][0]:.1f}-{seg[n][0] + seg['dur']:.1f} s (boxes on {seg_stats[n]['boxes_shown']:.0%})" for n in names))
    stats["boxes_shown_share"] = round(covered / max(1, total), 3)
    return stats


# ---------------------------------------------------------------- part 3: the 3D tour in headless Chrome

# A virtual clock for the page: performance.now, Date.now, timers and requestAnimationFrame only move when
# __vt.step(ms) is called, so the tour plays out the same whatever the machine's speed.
VT_SHIM = r"""
(() => {
  if (window.__vt) return;
  const P = performance, realNow = P.now.bind(P), realRAF = window.requestAnimationFrame.bind(window);
  const realCAF = window.cancelAnimationFrame.bind(window), realST = window.setTimeout.bind(window), realCT = window.clearTimeout.bind(window);
  const realSI = window.setInterval.bind(window), realCI = window.clearInterval.bind(window), realDate = Date.now.bind(Date);
  let on = false, now = 0, dateBase = 0, nextId = 1;
  const timers = new Map(), frames = new Map();
  P.now = () => (on ? now : realNow());
  Date.now = () => (on ? Math.floor(dateBase + now) : realDate());
  window.requestAnimationFrame = (cb) => { if (!on) return realRAF(cb); const id = -(nextId++); frames.set(id, cb); return id; };
  window.cancelAnimationFrame = (id) => { if (id < 0) frames.delete(id); else realCAF(id); };
  window.setTimeout = (cb, ms = 0, ...args) => { if (!on) return realST(cb, ms, ...args); const id = -(nextId++); timers.set(id, { at: now + Math.max(0, +ms || 0), cb, args, every: 0 }); return id; };
  window.clearTimeout = (id) => { if (id < 0) timers.delete(id); else realCT(id); };
  window.setInterval = (cb, ms = 0, ...args) => { if (!on) return realSI(cb, ms, ...args); const every = Math.max(1, +ms || 0), id = -(nextId++); timers.set(id, { at: now + every, cb, args, every }); return id; };
  window.clearInterval = (id) => { if (id < 0) timers.delete(id); else realCI(id); };
  const tick = () => new Promise((r) => realST(r, 0));
  const paint = () => new Promise((r) => realRAF(() => realRAF(r)));
  window.__vt = {
    after: null,
    get on() { return on; },
    get now() { return now; },
    start() { if (on) return; now = realNow(); dateBase = realDate() - now; on = true; },
    async step(ms) {
      const end = now + ms;
      for (let guard = 0; guard < 10000; guard++) {
        let id = null, best = Infinity;
        for (const [k, t] of timers) if (t.at <= end && t.at < best) { best = t.at; id = k; }
        if (id === null) break;
        const t = timers.get(id);
        now = Math.max(now, t.at);
        if (t.every) t.at += t.every; else timers.delete(id);
        try { typeof t.cb === "function" ? t.cb(...t.args) : (0, eval)(String(t.cb)); } catch (e) { console.error(e); }
        await tick();
      }
      now = end;
      const due = [...frames.values()];
      frames.clear();
      for (const cb of due) { try { cb(now); } catch (e) { console.error(e); } }
      await tick();
      if (this.after) { try { this.after(); } catch (e) { console.error(e); } }
      await paint();
    },
  };
})();
"""

# Hide the viewer's panels, add a caption and a Before/After badge, and drive the same visit as ?tour=changes
# (goToChange for every change in turn), captioning each change with its sentence.
TOUR_DRIVER = r"""
(() => {
  const v = window.viewer, names = Object.keys(v.data.splats), L = { A: "Before", B: "After", room: "Room" };
  const COL = { removed: "#ff5a5a", added: "#4cd97b", moved: "#5aa9ff" };
  const FONT = '"Geist", -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Segoe UI", sans-serif';
  // The viewer's glass: frosted, darkened enough for white text over the white walls, a bright rim at the top.
  const GLASS = `background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.025) 42%, rgba(255,255,255,.04)), rgba(20,22,27,.6);
    box-shadow: inset 0 1px 0 rgba(255,255,255,.42), inset 0 -1px 0 rgba(255,255,255,.11), inset 0 0 0 1px rgba(255,255,255,.09),
      inset 0 16px 22px -18px rgba(255,255,255,.3), 0 14px 34px -12px rgba(0,0,0,.6), 0 2px 8px -2px rgba(0,0,0,.35);
    backdrop-filter: blur(22px) saturate(1.7) brightness(.74); -webkit-backdrop-filter: blur(22px) saturate(1.7) brightness(.74);`;
  const st = document.createElement("style");
  st.textContent = `
    #side, #dock, #pip, #note, #stamp, #loading { display: none !important; }
    #wc-cap { position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%); width: max-content; max-width: 960px; box-sizing: border-box;
      padding: 14px 22px 15px; border-radius: 24px; ${GLASS} color: #f6f7f9; font: 600 21px/1.3 ${FONT}; display: none; }
    #wc-cap .k { display: inline-block; margin-right: 11px; padding: 2px 11px; border-radius: 999px; color: #111214; font-size: 15px; font-weight: 700;
      transform: translateY(-2px); box-shadow: inset 0 1px 0 rgba(255,255,255,.45), inset 0 -1px 0 rgba(0,0,0,.12); }
    #wc-cap .n { float: right; margin-left: 22px; color: rgba(246,247,249,.84); font-size: 15px; font-weight: 500; font-variant-numeric: tabular-nums;
      transform: translateY(3px); }
    #wc-cap .s { display: block; margin-top: 4px; color: rgba(246,247,249,.86); font-size: 18px; font-weight: 400; }
    #wc-badge { position: fixed; left: 20px; top: 18px; padding: 8px 20px 8px 15px; border-radius: 999px;
      ${GLASS} color: #f6f7f9; font: 650 20px/1.2 ${FONT}; }
    #wc-badge::before { content: ""; display: inline-block; vertical-align: 2px; width: 10px; height: 10px; margin-right: 10px; border-radius: 50%; background: #eceef2; box-shadow: 0 0 10px rgba(255,255,255,.55); }
    #wc-badge.B { color: #1d1606; background: linear-gradient(180deg, rgba(255,236,190,.55), rgba(255,200,87,0) 60%), rgba(255,200,87,.92);
      box-shadow: inset 0 1px 0 rgba(255,246,222,.9), inset 0 -1px 0 rgba(120,80,0,.25), 0 0 26px -2px rgba(255,200,87,.6), 0 10px 26px -10px rgba(0,0,0,.6); }
    #wc-badge.B::before { background: #1d1606; box-shadow: none; }
    #wc-head { position: fixed; right: 20px; top: 18px; padding: 9px 16px 10px; border-radius: 18px; ${GLASS}
      color: #f6f7f9; font: 500 14px/1.4 ${FONT}; text-align: right; }
    #wc-head span { color: rgba(246,247,249,.84); font-weight: 400; }
  `;
  document.head.append(st);
  const cap = Object.assign(document.createElement("div"), { id: "wc-cap" });
  const badge = Object.assign(document.createElement("div"), { id: "wc-badge" });
  const head = Object.assign(document.createElement("div"), { id: "wc-head" });
  head.innerHTML = "The room in 3D, rebuilt from the two phone videos<br><span>The camera only goes where a phone was</span>";
  document.body.append(cap, badge, head);
  const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
  function setCap(o) {
    if (!o) { cap.style.display = "none"; return; }
    cap.innerHTML = (o.n ? `<span class="n">${esc(o.n)}</span>` : "") + (o.k ? `<span class="k" style="background:${o.kc}">${esc(o.k)}</span>` : "")
      + esc(o.t) + (o.s ? `<span class="s">${esc(o.s)}</span>` : "");
    cap.style.display = "block";
  }
  const S = { done: false, step: "", errors: [] };
  function frame() {
    const n = v.current;
    badge.textContent = L[n] ?? n;
    badge.className = n;
    badge.style.display = names.length > 1 ? "block" : "none";
  }
  const vol = (o) => o.size[0] * o.size[1] * o.size[2];
  // The viewer moves by a fade through black (the view swaps at its midpoint) or a short ride along the path.
  async function arrived(maxMs = 4000) {
    const st = v.state;
    for (let t = 0; st && t < maxMs; t += 1000 / 30) {
      if (!(st.cut && !st.cut.done) && !st.ride) return;
      await new Promise((r) => setTimeout(r, 1000 / 30));
    }
  }
  async function start(o) {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    try {
      await v.ready;
      const changes = v.data.changes ?? [];
      // With changes to show, only the change boxes are drawn: the object boxes would compete with them.
      const objs = document.querySelector("[data-objects]");
      if (changes.length && objs && objs.getAttribute("aria-pressed") === "true") objs.click();
      setCap(changes.length ? { t: `${changes.length} change${changes.length > 1 ? "s" : ""} found`, s: "Each one from the view both videos filmed closest, Before and After in turn" }
                            : { t: "No changes found", s: "A tour of the objects the system found, Before and After" });
      await sleep(o.introMs);
      if (changes.length) {
        for (const [i, c] of changes.entries()) {
          S.step = c.id;
          const done = typeof v.goToChange === "function" ? v.goToChange(c) : Promise.resolve();
          await arrived();
          setCap({ k: `${c.id} · ${c.type}`, kc: COL[c.type] ?? "#fff", t: c.label, s: c.sentence, n: `${i + 1} / ${changes.length}` });
          await done;
          await sleep(o.holdMs);
        }
      } else {
        const cur = v.current, seen = new Set(), pick = [];
        for (const it of [...(v.objectsOf?.[cur] ?? [])].sort((a, b) => vol(b.o) - vol(a.o))) {
          if (seen.has(it.name)) continue;
          seen.add(it.name);
          pick.push(it);
          if (pick.length >= o.maxObjects) break;
        }
        for (const [i, it] of pick.entries()) {
          S.step = it.name;
          const done = v.goToObject(it);
          await arrived();
          setCap({ k: "no change", kc: "#cbd5e1", t: it.name, s: "Found in the 3D model; nothing changed here between Before and After", n: `${i + 1} / ${pick.length}` });
          await done;
          await sleep(1300);
          if (names.length > 1) {
            for (const n of [names[0], names[names.length - 1]]) { await v.show(n); await sleep(1300); }
          }
        }
      }
      setCap(null);
      await sleep(o.outroMs);
    } catch (e) { S.errors.push(String(e && e.stack || e)); }
    S.done = true;
  }
  window.__wc = { S, start, frame };
  window.__vt.after = frame;
  frame();
  return true;
})()
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """Serve a folder on 127.0.0.1 (and the packaged viewer at a virtual path when the room has no index.html)."""

    def __init__(self, root: Path, virtual: dict[str, bytes] | None = None):
        import http.server

        virtual = virtual or {}

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *a, **k):
                super().__init__(*a, directory=str(root), **k)

            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in virtual:
                    body = virtual[path]
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                super().do_GET()

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def find_chrome() -> str | None:
    for c in [os.environ.get("CHROME"), "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium", shutil.which("google-chrome"), shutil.which("chromium"),
              shutil.which("chromium-browser"), shutil.which("chrome")]:
        if c and Path(c).exists():
            return c
    return None


class CDP:
    """A minimal DevTools-protocol client over one page's websocket."""

    def __init__(self, url: str):
        from websockets.sync.client import connect
        self.ws = connect(url, max_size=None, open_timeout=30)
        self.n = 0
        self.console: list[str] = []

    def send(self, method: str, params: dict | None = None, timeout: float = 120) -> dict:
        self.n += 1
        mid = self.n
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=timeout))
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})
            if msg.get("method") == "Runtime.exceptionThrown":
                self.console.append(str(msg["params"]["exceptionDetails"].get("exception", {}).get("description", ""))[:300])
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"].get("type") in ("error", "warning"):
                self.console.append(" ".join(str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"])[:300])

    def eval(self, expr: str, timeout: float = 120):
        r = self.send("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True}, timeout)
        if "exceptionDetails" in r:
            raise RuntimeError(f"page error: {r['exceptionDetails'].get('exception', {}).get('description', r['exceptionDetails'])}"[:500])
        return r.get("result", {}).get("value")

    def close(self):
        self.ws.close()


def render_tour(room_out: Path, scene: dict, path: Path, parts: Path, max_s: float | None = None, log=print) -> dict | None:
    chrome = find_chrome()
    if not chrome:
        log("  tour: skipped (no Chrome or Chromium found; set CHROME=/path/to/chrome)")
        return None
    room_out = room_out.resolve()
    refs = [room_out]
    for p in list(scene.get("splats", {}).values()) + [scene.get("meta", {}).get("frames_dir") or "."]:
        refs.append((room_out / p).resolve().parent if "." in Path(p).name else (room_out / p).resolve())
    root = Path(os.path.commonpath([str(r) for r in refs]))
    rel = room_out.relative_to(root).as_posix()
    page = f"/{rel}/index.html" if rel != "." else "/index.html"
    virtual = {}
    if not (room_out / "index.html").exists():
        from .room3d import VIEWER
        page = page.replace("index.html", "__video_viewer.html")
        virtual[page] = VIEWER.read_text().encode()          # SCENE stays null: the page fetches scene.json next to it
        log("  tour: the room has no index.html; using the packaged viewer on its scene.json")
    srv = _Server(root, virtual)
    port = _free_port()
    profile = Path(tempfile.mkdtemp(prefix="wc-chrome-"))
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                             "--no-first-run", "--no-default-browser-check", "--hide-scrollbars", "--mute-audio",
                             "--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist",
                             "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                             "--disable-backgrounding-occluded-windows", "--window-size=1280,720", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stats: dict = {}
    cdp = None
    try:
        import urllib.request
        ws_url = None
        for _ in range(100):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2).read())
                ws_url = next((t["webSocketDebuggerUrl"] for t in tabs if t.get("type") == "page"), None)
                if ws_url:
                    break
            except OSError:
                pass
            time.sleep(0.2)
        if not ws_url:
            raise RuntimeError("Chrome did not start")
        cdp = CDP(ws_url)
        cdp.send("Runtime.enable")
        cdp.send("Page.enable")
        # 1280x720 CSS pixels at 1.5x: a 1920x1080 picture with the viewer's labels at a size that reads on video.
        cdp.send("Emulation.setDeviceMetricsOverride", {"width": 1280, "height": 720, "deviceScaleFactor": 1.5, "mobile": False})
        cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": VT_SHIM})
        url = f"http://127.0.0.1:{srv.port}{page}"
        cdp.send("Page.navigate", {"url": url})
        t0 = time.time()
        while not cdp.eval("!!(window.viewer && window.viewer.ready)"):
            if time.time() - t0 > 90:
                raise RuntimeError("the viewer did not start: " + "; ".join(cdp.console[-3:]))
            time.sleep(0.3)
        cdp.eval("window.viewer.ready.then(() => true)", timeout=180)
        gl = cdp.eval("(() => { const c = document.createElement('canvas').getContext('webgl2'); const e = c && c.getExtension('WEBGL_debug_renderer_info'); return e ? c.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown'; })()")
        stats["webgl"] = gl
        log(f"  tour: viewer loaded in {time.time() - t0:.1f}s, WebGL renderer: {gl}")
        cdp.eval(TOUR_DRIVER)
        time.sleep(1.5)                     # real time for the splats to be sorted before the clock is frozen
        cdp.eval("window.__vt.start(), true")
        for _ in range(3):
            cdp.eval("window.__vt.step(1000 / 30).then(() => true)")
        cdp.eval("window.__wc.start({ introMs: 2200, holdMs: 1500, outroMs: 600, maxObjects: 4 }), true")
        max_s = max_s or min(150.0, 12.0 + 11.0 * max(4, len(scene.get("changes") or [])))
        nmax = int(max_s * FPS)
        steps = []
        with FrameWriter(path) as fw:
            k = 0
            while k < nmax:
                s = cdp.eval("window.__vt.step(1000 / 30).then(() => ({ done: window.__wc.S.done, step: window.__wc.S.step }))")
                shot = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 93, "optimizeForSpeed": True})
                import base64
                img = cv2.imdecode(np.frombuffer(base64.b64decode(shot["data"]), np.uint8), cv2.IMREAD_COLOR)
                if img.shape[:2] != (H, W):
                    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
                fw.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if not steps or steps[-1][1] != s["step"]:
                    steps.append((k, s["step"]))
                k += 1
                if s["done"]:
                    break
            if k >= nmax:
                log(f"  tour: stopped at the {max_s:.0f} s cap")
        errs = cdp.eval("window.__wc.S.errors")
        stats.update(frames=k, seconds=round(k / FPS, 2), visits=[{"frame": f, "what": w} for f, w in steps if w], errors=errs,
                     console=cdp.console[-5:])
        if errs:
            log(f"  tour: page errors: {errs[:2]}")
    finally:
        if cdp:
            cdp.close()
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        srv.close()
        shutil.rmtree(profile, ignore_errors=True)
    return stats


# ---------------------------------------------------------------- stitching

def _duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                        "stream=nb_read_packets,r_frame_rate", "-of", "json", str(p)], capture_output=True, text=True)
    s = json.loads(r.stdout)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["nb_read_packets"]) * den / num


def stitch(clips: list[Path], out: Path, xfade: float = 0.5) -> float:
    """Join the parts with short crossfades, fade in from and out to black. Returns the length in seconds."""
    durs = [_duration(c) for c in clips]
    args = ["nice", "-n", "5", "ffmpeg", "-v", "error", "-y"]
    for c in clips:
        args += ["-i", str(c)]
    chain, last, off = [], "[0:v]", 0.0
    for i in range(1, len(clips)):
        off += durs[i - 1] - xfade
        chain.append(f"{last}[{i}:v]xfade=transition=fade:duration={xfade}:offset={off:.4f}[x{i}]")
        last = f"[x{i}]"
    total = sum(durs) - xfade * (len(clips) - 1)
    chain.append(f"{last}fade=t=in:st=0:d=0.4,fade=t=out:st={max(0.0, total - 0.6):.4f}:d=0.6,format=yuv420p[v]")
    args += ["-filter_complex", ";".join(chain), "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
             "-r", str(FPS), "-movflags", "+faststart", str(out)]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {r.stderr[-800:]}")
    return total


# ---------------------------------------------------------------- the whole video

def make_video(room_out: Path, run: Path | None, out_mp4: Path, parts_dir: Path | None = None, *, use_claude: bool = True,
               footage: bool = True, tour: bool = True, register: bool = True, reg_fps: float = 10.0,
               objects_on_footage: bool = False, log=print) -> dict:
    """Title card, real footage with tracked 3D boxes, 3D walkthrough, end card -> one mp4 (plus the parts).

    `run` (a `whatchanged run` folder) is optional: without it the facts come from the room's scene.json and the
    footage part is skipped unless the room's own COLMAP work folder is there."""
    room_out, out_mp4 = Path(room_out), Path(out_mp4)
    run = Path(run) if run else None
    if run and not run.exists():
        log(f"Video: {run} not found; going on without the detection run")
        run = None
    parts = Path(parts_dir) if parts_dir else out_mp4.with_name(out_mp4.stem + "_parts")
    parts.mkdir(parents=True, exist_ok=True)
    scene = load_scene(room_out)
    T: dict[str, float] = {}
    report: dict = {"room": str(room_out), "run": str(run) if run else None, "parts": {}}
    t_all = time.time()

    t = time.time()
    facts = change_facts(scene, run)
    (parts / "facts.json").write_text(json.dumps(facts, indent=1))
    summary = summary_line(facts, parts / "summary_cache.json", use_claude, log)
    report["summary"] = summary
    log(f"Video: {facts['change_count']} change(s); summary ({summary['source']}): {summary['text']}")
    title = title_card(facts, summary)
    title.save(parts / "01_title.png")
    clips = [parts / "01_title.mp4"]
    still_clip(title, 3.5, clips[0])
    T["title"] = time.time() - t

    if footage:
        t = time.time()
        log("Video: real footage with 3D boxes")
        try:
            st = render_footage(room_out, run, scene, parts / "02_footage.mp4", parts, register=register, reg_fps=reg_fps,
                                show_objects=objects_on_footage, log=log)
        except Exception as e:  # noqa: BLE001  (the other parts still make a video)
            log(f"  footage: failed ({type(e).__name__}: {e}); left out")
            st = None
        if st is not None:
            clips.append(parts / "02_footage.mp4")
            report["parts"]["footage"] = st
        T["footage"] = time.time() - t

    if tour:
        t = time.time()
        log("Video: 3D walkthrough in headless Chrome")
        try:
            st = render_tour(room_out, scene, parts / "03_tour.mp4", parts, log=log)
        except Exception as e:  # noqa: BLE001
            log(f"  tour: failed ({type(e).__name__}: {e}); left out")
            st = None
        if st is not None:
            clips.append(parts / "03_tour.mp4")
            report["parts"]["tour"] = st
        T["tour"] = time.time() - t

    t = time.time()
    prov = provenance(scene, run)
    end = end_card(scene, facts, prov)
    end.save(parts / "04_end.png")
    clips.append(parts / "04_end.mp4")
    still_clip(end, 4.0, clips[-1])
    T["end"] = time.time() - t

    t = time.time()
    length = stitch(clips, out_mp4)
    T["stitch"] = time.time() - t
    T["total"] = time.time() - t_all
    report.update(video=str(out_mp4), seconds=round(length, 2), clips=[str(c) for c in clips], provenance=prov,
                  timings_s={k: round(v, 1) for k, v in T.items()})
    (parts / "video_report.json").write_text(json.dumps(report, indent=1, default=str))
    log(f"Video: {out_mp4} ({length:.1f} s) in {T['total']:.0f}s  [" + ", ".join(f"{k} {v:.0f}s" for k, v in T.items() if k != "total") + "]")
    return report
