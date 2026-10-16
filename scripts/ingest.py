#!/usr/bin/env python3
"""Copy the clips to data/raw/{pilot,A,A2,B}.mov with location, audio and all other metadata removed.

  scripts/ingest.py A=~/Downloads/IMG_0102.MOV A2=IMG_0103.MOV B=IMG_0107.MOV [pilot=IMG_0100.MOV]
  scripts/ingest.py A=IMG_0102.MOV+IMG_0104.MOV ...          a take that broke: the clips are joined in order
  scripts/ingest.py --times "pilot=09:05 A=09:20 A2=09:33 B=10:08"   match ~/Downloads clips by recording time
  scripts/ingest.py --times "..." --dry-run                  show the matching and the facts, write nothing

Bare file names are looked up in ~/Downloads. Each clip is reported (codec, size, frame rate, length,
colour transfer, rotation). HDR (HLG or PQ) stops the run unless --allow-hdr. Every output is checked
with ffprobe and by reading its atoms: one video stream, no Apple location, make, model or date tags.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
LABELS = ["pilot", "A", "A2", "B"]
ALIAS = {"A'": "A2", "A\u2032": "A2"}   # the plan's A-prime is A2 here
VIDEO_EXT = {".mov", ".mp4", ".m4v"}
HDR = {"arib-std-b67": "HLG", "smpte2084": "PQ"}
# Bytes that must not appear anywhere outside the video samples of a cleaned file.
FORBIDDEN = [b"ISO6709", b"location", b"\xa9xyz", b"\xa9mak", b"\xa9mod", b"\xa9day", b"com.apple",
             b"mebx", b"iPhone", b"creationdate"]


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"ffprobe cannot read {path}: {r.stderr.strip()}")
    return json.loads(r.stdout)


def _rate(x: str | None) -> float:
    a, _, b = (x or "0/1").partition("/")
    return float(a) / float(b or 1) if float(b or 1) else 0.0


def recorded_at(fmt_tags: dict) -> datetime | None:
    """When recording started, in local time. Apple's creationdate carries the phone's own offset."""
    s = fmt_tags.get("com.apple.quicktime.creationdate")
    if s:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").astimezone()
    s = fmt_tags.get("creation_time")
    if s:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone()
    return None


def facts(path: Path) -> dict:
    """What the pipeline cares about, from ffprobe."""
    d = probe(path)
    v = [s for s in d["streams"] if s.get("codec_type") == "video"]
    if not v:
        sys.exit(f"{path} has no video stream")
    v, tags = v[0], d["format"].get("tags", {})
    rot = [sd.get("rotation") for sd in v.get("side_data_list", []) if "rotation" in sd]
    at = recorded_at(tags)
    return {"path": str(path), "codec": v.get("codec_name"), "width": v.get("width"), "height": v.get("height"),
            "pix_fmt": v.get("pix_fmt"), "avg_fps": round(_rate(v.get("avg_frame_rate")), 3),
            "r_fps": round(_rate(v.get("r_frame_rate")), 3), "duration_s": round(float(d["format"].get("duration", 0)), 2),
            "transfer": v.get("color_transfer", "unknown"), "rotation": rot[0] if rot else 0,
            "dolby_vision": any("DOVI" in sd.get("side_data_type", "") for sd in v.get("side_data_list", [])),
            "streams": len(d["streams"]), "has_location": any("location" in k for k in tags),
            "recorded_at": at.isoformat(timespec="seconds") if at else None, "size_mb": round(path.stat().st_size / 1e6)}


def show(label: str, f: dict) -> None:
    vfr = abs(f["avg_fps"] - f["r_fps"]) / max(f["r_fps"], 1e-9) > 0.01
    hdr = HDR.get(f["transfer"])
    print(f"  {label:5s} {Path(f['path']).name}: {f['codec']} {f['width']}x{f['height']} {f['pix_fmt']}, "
          f"{f['duration_s']:.1f} s, fps avg {f['avg_fps']} / r {f['r_fps']}{' (VARIABLE)' if vfr else ''}, "
          f"transfer {f['transfer']}{' = HDR ' + hdr if hdr else ''}{' + Dolby Vision' if f['dolby_vision'] else ''}, "
          f"rotation {f['rotation']}, {f['size_mb']} MB, recorded {f['recorded_at'] or '?'}")


def resolve(name: str) -> Path:
    p = Path(name).expanduser()
    if not p.exists() and (DOWNLOADS / name).exists():
        p = DOWNLOADS / name
    if not p.exists():
        sys.exit(f"no such clip: {name}")
    return p.resolve()


def match_by_time(times: dict[str, str], clips: list[Path], tolerance_min: float) -> dict[str, list[Path]]:
    """Give each label the clip whose recording start is nearest its written clock time.

    Some phones stamp the end of the recording instead of the start, so a clip also matches at
    (stamp - length). All one-to-one assignments are tried; the smallest total gap wins.
    """
    info = {c: facts(c) for c in clips}
    stamped = [c for c in clips if info[c]["recorded_at"]]
    for c in clips:
        if c not in stamped:
            print(f"  (skipped {c.name}: no recording time in its metadata)")

    def minutes(dt: datetime) -> float:
        return dt.hour * 60 + dt.minute + dt.second / 60

    def gap(label: str, c: Path) -> float:
        h, m = map(int, times[label].split(":"))
        start = minutes(datetime.fromisoformat(info[c]["recorded_at"]))
        return min(abs(h * 60 + m - start), abs(h * 60 + m - (start - info[c]["duration_s"] / 60)))

    labels = list(times)
    if len(stamped) < len(labels):
        sys.exit(f"{len(labels)} times given but only {len(stamped)} clips with a recording time: name them instead")
    best = min(itertools.permutations(stamped, len(labels)), key=lambda p: sum(gap(l, c) for l, c in zip(labels, p)))
    out = {}
    for l, c in zip(labels, best):
        g = gap(l, c)
        print(f"  {l:5s} written {times[l]}  ->  {c.name} (recorded {info[c]['recorded_at'][11:19]}, "
              f"{info[c]['duration_s']:.0f} s long, {g:.1f} min off)")
        if g > tolerance_min:
            sys.exit(f"{l}: the nearest clip is {g:.0f} min from {times[l]}; check the times or name the clips (A=IMG_....MOV)")
        out[l] = [c]
    return out


def atoms(data: bytes, off: int = 0, end: int | None = None, path: str = "") -> list[tuple[str, int, int]]:
    """(path, start, end) of every box, descending into containers; mdat is listed but not opened."""
    end = len(data) if end is None else end
    out = []
    while off + 8 <= end:
        size, typ = int.from_bytes(data[off:off + 4], "big"), data[off + 4:off + 8].decode("latin1")
        hdr = 8
        if size == 1:
            size, hdr = int.from_bytes(data[off + 8:off + 16], "big"), 16
        elif size == 0:
            size = end - off
        if size < hdr:
            break
        p = f"{path}/{typ}"
        out.append((p, off, off + size))
        if typ in ("moov", "trak", "mdia", "minf", "stbl", "udta", "edts", "dinf", "meta", "tref"):
            inner = off + hdr + (4 if typ == "meta" and data[off + hdr + 4:off + hdr + 8] != b"hdlr" else 0)
            out += atoms(data, inner, off + size, p)
        off += size
    return out


def read_boxes(path: Path) -> tuple[list[tuple[str, int, int]], bytes]:
    """Top-level boxes read from the file without loading the (multi-GB) mdat; returns boxes and non-mdat bytes."""
    tops, keep = [], bytearray()
    with open(path, "rb") as f:
        size_all = path.stat().st_size
        off = 0
        while off + 8 <= size_all:
            f.seek(off)
            head = f.read(16)
            size, typ = int.from_bytes(head[:4], "big"), head[4:8].decode("latin1")
            if size == 1:
                size = int.from_bytes(head[8:16], "big")
            elif size == 0:
                size = size_all - off
            if typ != "mdat":
                f.seek(off)
                keep += f.read(size)
            tops.append((typ, off, off + size))
            off += size
    return tops, bytes(keep)


def verify(path: Path) -> list[str]:
    """Problems left in a cleaned file (empty list = clean)."""
    problems = []
    d = probe(path)
    if len(d["streams"]) != 1 or d["streams"][0].get("codec_type") != "video":
        problems.append(f"{len(d['streams'])} streams, expected one video stream")
    left = {k for k in d["format"].get("tags", {}) if k not in ("major_brand", "minor_version", "compatible_brands")}
    left |= {k for s in d["streams"] for k in s.get("tags", {}) if k not in ("handler_name", "vendor_id", "language")}
    if left:
        problems.append(f"tags left: {sorted(left)}")
    _, meta = read_boxes(path)
    boxes = [p for p, _, _ in atoms(meta)]
    extra = [p for p in boxes if p.split("/")[-1] in ("udta", "meta", "keys", "ilst", "\xa9xyz", "loci")]
    if extra:
        problems.append(f"metadata atoms left: {extra}")
    hits = [s.decode("latin1") for s in FORBIDDEN if s in meta]
    if hits:
        problems.append(f"found {hits} outside the video samples")
    return problems


def clean(srcs: list[Path], out: Path) -> float:
    """Video stream only, stream-copied (no re-encode), every tag dropped; clips are joined if several."""
    t = time.time()
    part = out.with_suffix(".part.mov")
    args = ["ffmpeg", "-v", "error", "-y"]
    with tempfile.TemporaryDirectory() as tmp:
        if len(srcs) > 1:
            lst = Path(tmp) / "list.txt"
            lst.write_text("".join(f"file '{s}'\n" for s in srcs))
            args += ["-f", "concat", "-safe", "0", "-i", str(lst)]
        else:
            args += ["-i", str(srcs[0])]
        args += ["-map", "0:v:0", "-c:v", "copy", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
                 "-fflags", "+bitexact", str(part)]
        r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        part.unlink(missing_ok=True)
        sys.exit(f"ffmpeg failed on {[s.name for s in srcs]}:\n{r.stderr.strip()[-1500:]}")
    os.replace(part, out)
    return time.time() - t


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clips", nargs="*", help="LABEL=FILE[+FILE...] (labels: pilot, A, A2, B), or plain files to match with --times")
    p.add_argument("--times", help='written clock times, e.g. "pilot=09:05 A=09:20 A2=09:33 B=10:08"')
    p.add_argument("--since-hours", type=float, default=24, help="with --times and no files: ~/Downloads videos this new")
    p.add_argument("--tolerance-min", type=float, default=10, help="largest gap allowed between a written time and its clip")
    p.add_argument("--out", default=str(REPO / "data" / "raw"))
    p.add_argument("--allow-hdr", action="store_true", help="write HLG/PQ clips anyway (the gates were made on SDR)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--redo", action="store_true", help="rewrite outputs even if they came from the same clips")
    a = p.parse_args(argv)

    named = [c for c in a.clips if "=" in c]
    loose = [resolve(c) for c in a.clips if "=" not in c]
    plan: dict[str, list[Path]] = {}
    for c in named:
        label, _, files = c.partition("=")
        label = ALIAS.get(label, label)
        if label not in LABELS:
            sys.exit(f"unknown label {label!r}; use {', '.join(LABELS)}")
        plan[label] = [resolve(f) for f in files.split("+")]
    if a.times:
        times = {ALIAS.get(k, k): v for k, v in (t.split("=") for t in a.times.replace(",", " ").split())}
        if set(times) - set(LABELS):
            sys.exit(f"unknown label in --times; use {', '.join(LABELS)}")
        times = {k: v for k, v in times.items() if k not in plan}
        if not loose:
            cutoff = time.time() - a.since_hours * 3600
            loose = sorted(f.resolve() for f in DOWNLOADS.iterdir()
                           if f.suffix.lower() in VIDEO_EXT and f.stat().st_mtime > cutoff)
            print(f"Matching {len(times)} written times against {len(loose)} videos in {DOWNLOADS} (last {a.since_hours:g} h)")
        taken = {s for v in plan.values() for s in v}
        plan.update(match_by_time(times, [c for c in loose if c not in taken], a.tolerance_min))
    elif loose:
        sys.exit("plain files need --times to say which is which (or write LABEL=FILE)")
    if not plan:
        sys.exit(__doc__)
    missing = [l for l in ("A", "A2", "B") if l not in plan]
    if missing and not set(plan) <= {"pilot"}:
        print(f"note: no clip for {', '.join(missing)} yet")

    print("Clips:")
    hdr = []
    for label in sorted(plan, key=LABELS.index):
        for src in plan[label]:
            f = facts(src)
            show(label, f)
            if HDR.get(f["transfer"]):
                hdr.append(f"{label} ({src.name}) is HDR {HDR[f['transfer']]} ({f['transfer']})")
            if f["avg_fps"] and abs(f["avg_fps"] - f["r_fps"]) / f["r_fps"] > 0.01:
                print(f"        warning: variable frame rate (avg {f['avg_fps']} vs r {f['r_fps']}); Auto FPS on?")
            if f["rotation"]:
                print(f"        note: rotated {f['rotation']} deg (portrait?); the pipeline expects landscape")
    if hdr:
        print("\n" + "!" * 72 + "\nFAIL: " + "\nFAIL: ".join(hdr) + "\nThe gates were set on SDR footage. Turn HDR off "
              "(Blackmagic: Rec.709 / SDR) and re-film, or pass --allow-hdr.\n" + "!" * 72)
        if not a.allow_hdr:
            sys.exit(2)
    if a.dry_run:
        print("Dry run: nothing written.")
        return

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    manifest = out_dir / "ingest.json"
    if manifest.exists():
        record = json.loads(manifest.read_text())
    for label in sorted(plan, key=LABELS.index):
        out = out_dir / f"{label}.mov"
        src = [x.name for x in plan[label]]
        if (not a.redo and out.exists() and record.get(label, {}).get("sources") == src
                and all(x.stat().st_mtime < out.stat().st_mtime for x in plan[label])):
            # Rewriting would give it a new date, and protocol.sh would then refuse to reuse finished work.
            print(f"  kept {out}: already made from {'+'.join(src)} (--redo to rewrite)")
            continue
        s = clean(plan[label], out)
        problems = verify(out)
        if problems:
            out.unlink()
            sys.exit(f"{label}: metadata survived the clean-up, file removed: {problems}")
        f = facts(out)
        print(f"  wrote {out} in {s:.1f} s: {f['size_mb']} MB, {f['duration_s']:.1f} s, 1 stream, no location or other tags")
        record[label] = {"sources": src, "ingested": datetime.now().isoformat(timespec="seconds"),
                         **{k: v for k, v in facts(plan[label][0]).items() if k not in ("path", "has_location", "streams")},
                         "duration_s": f["duration_s"]}
    manifest.write_text(json.dumps(record, indent=1))
    print(f"Done. Manifest: {manifest}")


if __name__ == "__main__":
    main()
