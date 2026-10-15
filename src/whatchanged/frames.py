"""Pick sharp, evenly spaced keyframes from a phone video and keep their real timestamps."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Frame:
    session: str        # "A" or "B"
    index: int          # position within the session
    path: str           # saved JPEG
    t: float            # seconds from the start of the video
    sharpness: float


def video_info(path: str | Path) -> dict:
    """Basic facts about a video, plus HDR and rotation flags from ffprobe when available."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    info = {
        "path": str(path),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration_s"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=color_transfer,avg_frame_rate,r_frame_rate:stream_side_data=rotation",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        stream = json.loads(out.stdout)["streams"][0]
        info["color_transfer"] = stream.get("color_transfer", "unknown")
        info["hdr"] = stream.get("color_transfer") in ("arib-std-b67", "smpte2084")
        def _rate(x):
            a, _, b = (x or "0/1").partition("/")
            return float(a) / float(b or 1) if float(b or 1) else 0.0
        ra, rr = _rate(stream.get("avg_frame_rate")), _rate(stream.get("r_frame_rate"))
        info["variable_frame_rate"] = bool(rr and abs(ra - rr) / rr > 0.01)
        rot = [sd.get("rotation") for sd in stream.get("side_data_list", []) if "rotation" in sd]
        info["rotation"] = rot[0] if rot else 0
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, IndexError, json.JSONDecodeError):
        info.update(color_transfer="unknown", hdr=None, variable_frame_rate=None, rotation=None)
    return info


def check_pair(a: str | Path, b: str | Path) -> list[str]:
    """Warnings a person should see before spending time on 3D."""
    ia, ib = video_info(a), video_info(b)
    warnings = []
    for name, i in (("A", ia), ("B", ib)):
        if i["hdr"]:
            warnings.append(f"{name} is HDR ({i['color_transfer']}); colours may look washed out. Turn HDR Video off next time.")
        if i["variable_frame_rate"]:
            warnings.append(f"{name} has a variable frame rate; timestamps come from the video itself, so this is handled.")
        if i["duration_s"] < 10:
            warnings.append(f"{name} is only {i['duration_s']:.0f} s long; 30-60 s gives better coverage.")
        if i["width"] < i["height"]:
            warnings.append(f"{name} is portrait; landscape keeps more of the room in each frame.")
    if (ia["width"], ia["height"]) != (ib["width"], ib["height"]) and (ia["width"], ia["height"]) != (ib["height"], ib["width"]):
        warnings.append(f"A and B have different resolutions ({ia['width']}x{ia['height']} vs {ib['width']}x{ib['height']}); was the lens or setting changed?")
    return warnings


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(video: str | Path, out_dir: str | Path, session: str, n: int = 24,
                      max_side: int = 1024, candidates_per_bin: int = 5) -> list[Frame]:
    """Split the video into n equal time bins and keep the sharpest of a few candidates in each."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise ValueError(f"no frames in {video}")
    edges = np.linspace(0, total, n + 1).astype(int)
    wanted = {}
    for b in range(n):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        for idx in np.unique(np.linspace(lo, hi - 1, candidates_per_bin).astype(int)):
            wanted[int(idx)] = b

    best: dict[int, tuple[float, np.ndarray, float]] = {}
    pos = 0
    for target in sorted(wanted):
        while pos < target:
            if not cap.grab():
                break
            pos += 1
        ok = cap.grab()
        if not ok:
            break
        pos += 1
        ok, img = cap.retrieve()
        if not ok:
            continue
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        s = _sharpness(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        b = wanted[target]
        if b not in best or s > best[b][0]:
            best[b] = (s, img, t)
    cap.release()

    frames = []
    for i, b in enumerate(sorted(best)):
        s, img, t = best[b]
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        path = out_dir / f"{session}_{i:03d}.jpg"
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames.append(Frame(session=session, index=i, path=str(path), t=round(t, 3), sharpness=round(s, 1)))
    # Drop frames far blurrier than the session's median, but keep at least 80 %.
    med = float(np.median([f.sharpness for f in frames]))
    keep = [f for f in frames if f.sharpness >= 0.25 * med]
    if len(keep) < 0.8 * len(frames):
        keep = sorted(frames, key=lambda f: -f.sharpness)[: int(np.ceil(0.8 * len(frames)))]
        keep.sort(key=lambda f: f.t)
    return keep


def save_manifest(frames: list[Frame], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(f) for f in frames], indent=1))


def load_manifest(path: str | Path) -> list[Frame]:
    return [Frame(**d) for d in json.loads(Path(path).read_text())]
