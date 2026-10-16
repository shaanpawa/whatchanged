#!/usr/bin/env python3
"""The 10-minute check of the pilot, or of take A before anything in the room is moved.

  scripts/quickcheck.py data/raw/pilot.mov
  scripts/quickcheck.py data/raw/A.mov --side-mm 150          marker side, for heights in metres
  scripts/quickcheck.py ~/Downloads/IMG_5849.MOV --out /tmp/qc

One sharp frame per second (ffmpeg, hardware decoding, longest side 1600 px, the same rule as
`whatchanged room`), then COLMAP on those frames with the pipeline's own room3d.run_colmap.
Gates (from the filming review): SDR, steady frame rate, >= 95% of frames posed, camera heights
span >= 1.0 m (>= 0.7x the median height when the marker isn't seen), some frames >= 35 deg down.
Blur, exposure, flicker, look-up share, heading coverage, lens, posing gaps and the floor marker
are reported alongside. Exit 0 = every gate passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from whatchanged import room3d  # noqa: E402
import ingest  # noqa: E402
import marker_scale  # noqa: E402

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def packet_times(path: Path) -> np.ndarray:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return np.sort(np.array([float(x) for x in r.stdout.split() if x.strip() not in ("", "N/A")]))


def decode(path: Path, f: dict, out_dir: Path, prefix: str, fps: float = 1.0, max_side: int = 1600) -> tuple[list[dict], dict]:
    """room3d.extract_frames' rule (sharpest frame per 1/fps window, drop windows under 0.35x the median),
    decoded by ffmpeg instead of OpenCV (several times faster on 4K HEVC), plus per-frame row means for flicker."""
    w, h = f["width"], f["height"]
    if abs(f["rotation"]) in (90, 270):
        w, h = h, w
    s = min(1.0, max_side / max(w, h))
    W, H = round(w * s), round(h * s)
    vfps = f["r_fps"] or 30.0
    window = max(1, round(vfps / fps))
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"decoder": None}
    for hw in (["-hwaccel", "videotoolbox"], []):
        cmd = ["ffmpeg", "-v", "error", "-nostdin", *hw, "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough",
               "-vf", f"scale={W}:{H}:flags=area", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        buf = bytearray(W * H * 3)
        view = memoryview(buf)
        rows, best, i = [], {}, 0
        cur = None                                       # (window, sharpness, image, gray, i)
        while True:
            got = 0
            while got < len(buf):
                n = proc.stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < len(buf):
                break
            img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            gray = cv2.cvtColor(cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            rows.append(gray.mean(1, dtype=np.float32))
            b = i // window
            if cur is not None and cur[0] != b:
                best[cur[0]] = _keep(cur, out_dir, prefix)
            if cur is None or cur[0] != b or sharp > cur[1]:
                cur = (b, sharp, img.copy(), gray, i)
            i += 1
        if cur is not None:
            best[cur[0]] = _keep(cur, out_dir, prefix)
        err = proc.stderr.read().decode(errors="replace")
        proc.wait()
        if i > 0:
            stats["decoder"] = "videotoolbox" if hw else "software"
            break
        log(f"  decoder {'videotoolbox' if hw else 'software'} failed: {err.strip()[-300:]}")
    if not best:
        sys.exit(f"could not decode {path}")
    med = float(np.median([v["sharpness"] for v in best.values()]))
    frames = []
    for b in sorted(best):
        v = best[b]
        tmp = out_dir / v.pop("tmp")
        if v["sharpness"] < 0.35 * med:
            tmp.unlink()
            continue
        name = f"{prefix}_{len(frames):04d}.jpg"
        tmp.rename(out_dir / name)
        frames.append({"name": f"{out_dir.name}/{name}", "video": str(path), "t": round(v["i"] / vfps, 3), **v})
    stats.update(frames_decoded=i, windows=len(best), kept=len(frames), blur_share=1 - len(frames) / len(best),
                 size=[W, H], vfps=vfps, rows=np.stack(rows))
    return frames, stats


def _keep(cur, out_dir: Path, prefix: str) -> dict:
    b, sharp, img, gray, i = cur
    tmp = f"tmp_{prefix}_{b:05d}.jpg"
    cv2.imwrite(str(out_dir / tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"tmp": tmp, "sharpness": round(sharp, 1), "i": i,
            "clipped": float((gray >= 250).mean()), "crushed": float((gray <= 5).mean())}


def flicker(rows: np.ndarray, vfps: float, mains_hz: float = 50.0) -> dict:
    """A heuristic for rolling bands from mains lighting (UK 50 Hz mains = 100 Hz flicker).

    Bands that roll from frame to frame leave the same few-cycles-per-frame sine in every
    frame-to-frame change of the row profile, while camera motion leaves a smooth spectrum. So the
    row spectra of the changes are averaged and compared with a power-law fit; a peak well above
    it at a low cycle count is reported. Separately, frame brightness is checked at the frequency
    100 Hz aliases to at this frame rate (10 Hz at 30 fps). Neither is proof; look at the footage.
    """
    L = np.log(rows.astype(np.float64) + 1)
    D = np.diff(L, axis=0)
    D -= D.mean(1, keepdims=True)
    y = np.linspace(-1, 1, D.shape[1])
    D -= np.outer(D @ y / (y @ y), y)                    # remove each change's tilt (exposure ramps)
    P = (np.abs(np.fft.rfft(D * np.hanning(D.shape[1]), axis=1)) ** 2).mean(0)
    k = np.arange(1, min(60, len(P)))
    fit = np.polyfit(np.log(k), np.log(P[k]), 1)
    ratio = P[k] / np.exp(np.polyval(fit, np.log(k)))
    j = int(np.argmax(ratio[:20]))
    out = {"row_peak_cycles": int(k[j]), "row_peak_ratio": round(float(ratio[j]), 2)}
    f_alias = abs(2 * mains_hz - round(2 * mains_hz / vfps) * vfps)
    m = L.mean(1)
    if len(m) > 128 and 0.5 < f_alias < vfps / 2 - 0.5:
        m = m - np.convolve(m, np.ones(9) / 9, mode="same")
        seg = 128
        S = np.mean([np.abs(np.fft.rfft(m[s:s + seg] * np.hanning(seg))) ** 2 for s in range(0, len(m) - seg, seg // 2)], 0)
        fr = np.fft.rfftfreq(seg, 1 / vfps)
        near = np.abs(fr - f_alias) <= 0.5
        band = (fr > 1) & (fr < vfps / 2 - 0.5)
        out.update(alias_hz=round(f_alias, 2), alias_ratio=round(float(S[near].max() / np.median(S[band])), 2))
    out["suspect"] = out["row_peak_ratio"] > 3.0 or out.get("alias_ratio", 0) > 8.0
    return out


def geometry(dense: Path, frames: list[dict], side_mm: float) -> dict:
    """Registered share, heights, pitch, headings and posing gaps from the COLMAP model."""
    sp = dense / "sparse" / "0"
    imgs = room3d.read_images(sp / "images.bin")
    xyz, _, _ = room3d.read_points(sp / "points3D.bin")
    names = {im["name"] for im in imgs.values()}
    c2w = np.stack([room3d.cam_to_world(im) for im in sorted(imgs.values(), key=lambda v: v["name"])])
    C, fwd = c2w[:, :3, 3], c2w[:, :3, 2]
    up = room3d.level_up(c2w)
    out = {"registered": len(names), "kept": len(frames), "points": len(xyz)}
    try:
        n, d = room3d.find_floor(xyz, up, C)
        up, h = n, C @ n + d
        out["median_height_units"] = float(np.median(h))
    except Exception as e:                                  # too few points under the cameras
        h = C @ up
        out["floor"] = f"not found ({e.__class__.__name__})"
    out["height_span_units"] = float(np.percentile(h, 95) - np.percentile(h, 5))
    m = marker_scale.measure(dense, side_mm, log=lambda *_: None)
    if m["ok"] and m["spread"] <= 0.10:
        out["metres_per_unit"] = m["metres_per_unit"]
        out["marker_spread"] = m["spread"]
    pitch = np.degrees(np.arcsin(np.clip(fwd @ up, -1, 1)))
    out.update(pitch_p5_p50_p95=np.round(np.percentile(pitch, [5, 50, 95]), 1).tolist(),
               down35=float((pitch <= -35).mean()), up10=float((pitch >= 10).mean()))
    level = fwd - np.outer(fwd @ up, up)                    # a reference heading: the first camera not looking straight down
    e1 = level[np.argmax(np.linalg.norm(level, axis=1) > 0.2)]
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    head = np.degrees(np.arctan2(fwd @ e2, fwd @ e1)) % 360
    sectors = np.bincount((head // 30).astype(int), minlength=12)
    hs = np.sort(head)
    out.update(heading_sectors=int((sectors > 0).sum()), heading_gap_deg=round(float(np.max(np.diff(np.r_[hs, hs[0] + 360]))), 0))
    gaps, run = [], None                                    # stretches of the video that did not pose
    for fr in frames:
        if fr["name"] not in names:
            run = [run[0] if run else fr["t"], fr["t"]]
        elif run:
            gaps.append(run)
            run = None
    if run:
        gaps.append(run)
    out["unposed_seconds"] = [[round(a), round(b)] for a, b in gaps]
    models = [p for p in (dense.parent / "sparse").iterdir() if (p / "images.bin").exists()]
    out["models"] = len(models)
    cam = next(iter(room3d.read_cameras(room3d._largest_model(dense.parent / "sparse")[0] / "cameras.bin").values()))
    und = next(iter(room3d.read_cameras(sp / "cameras.bin").values()))
    out["hfov_deg"] = round(float(np.degrees(2 * np.arctan(cam["width"] / 2 / cam["params"][0]))), 1)
    out["undistorted_keeps"] = f"{und['width']}x{und['height']} of {cam['width']}x{cam['height']}"
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip")
    p.add_argument("--out", help="work folder (default out/quickcheck/<clip name>)")
    p.add_argument("--fps", type=float, default=1.0, help="frames kept per second")
    p.add_argument("--side-mm", dest="side_mm", type=float, default=150, help="printed marker's black square side")
    p.add_argument("--no-colmap", dest="no_colmap", action="store_true", help="footage checks only (about a minute)")
    a = p.parse_args(argv)
    clip = Path(a.clip).expanduser().resolve()
    prefix = re.sub(r"[^A-Za-z0-9]+", "", clip.stem) or "clip"
    work = Path(a.out) if a.out else REPO / "out" / "quickcheck" / prefix
    shutil.rmtree(work / "images", ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    gates, notes, times = [], [], {}

    def gate(name: str, ok: bool | None, detail: str) -> None:
        gates.append((name, ok, detail))

    log(f"Quick check of {clip}")
    f = ingest.facts(clip)
    ingest.show(" clip", f)
    pts = packet_times(clip)
    dt = np.diff(pts)
    irregular = float((np.abs(dt - np.median(dt)) > 0.25 * np.median(dt)).mean()) if len(dt) > 10 else 1.0
    hdr = ingest.HDR.get(f["transfer"])
    gate("SDR", not hdr, f"transfer {f['transfer']}" + (f" = HDR {hdr}" if hdr else ""))
    vfr = abs(f["avg_fps"] - f["r_fps"]) / max(f["r_fps"], 1e-9)
    gate("steady fps", vfr <= 0.01 and irregular <= 0.01,
         f"avg {f['avg_fps']} vs r {f['r_fps']} ({vfr:.2%} apart); {irregular:.1%} of frame intervals off by >25%")
    if f["has_location"]:
        notes.append("WARN  location tag present (fine for processing; data/raw copies from ingest.py have none)")

    t = time.time()
    frames, st = decode(clip, f, work / "images" / prefix, prefix, fps=a.fps)
    times["decode"] = time.time() - t
    log(f"Decoded {st['frames_decoded']} frames ({st['decoder']}) in {times['decode']:.0f} s "
        f"({f['duration_s'] / max(times['decode'], 1e-9):.1f}x real time): kept {st['kept']} of {st['windows']} windows")
    (work / "frames.json").write_text(json.dumps({prefix: frames}, indent=1))
    blur = st["blur_share"]
    notes.append(f"{'WARN' if blur > 0.10 else 'ok  '}  blur: {blur:.0%} of {1 / a.fps:g}-s windows had no frame above 0.35x the median sharpness")
    cl, cr = np.array([v["clipped"] for v in frames]), np.array([v["crushed"] for v in frames])
    bad_exp = np.median(cl) > 0.02 or np.median(cr) > 0.05
    notes.append(f"{'WARN' if bad_exp else 'ok  '}  exposure: clipped highlights median {np.median(cl):.1%} (p90 {np.percentile(cl, 90):.1%}), "
                 f"crushed shadows median {np.median(cr):.1%} (p90 {np.percentile(cr, 90):.1%})")
    fl = flicker(st["rows"], st["vfps"])
    notes.append(f"{'WARN' if fl['suspect'] else 'ok  '}  flicker (heuristic): row-change spectrum peak {fl['row_peak_ratio']}x the trend "
                 f"at {fl['row_peak_cycles']} cycles/frame" + (f"; frame brightness at {fl['alias_hz']} Hz (100 Hz alias) "
                                                                f"{fl['alias_ratio']}x the median" if "alias_ratio" in fl else ""))
    det = marker_scale.detector()
    seen = sum(marker_scale.detect(det, cv2.imread(str(work / "images" / fr["name"]), cv2.IMREAD_GRAYSCALE)) is not None
               for fr in frames)
    notes.append(f"{'ok  ' if seen >= 10 else 'WARN'}  floor marker (DICT_4X4_50 id 0) found in {seen}/{len(frames)} frames (want >= 10)")

    geo = {}
    if not a.no_colmap:
        t = time.time()
        log(f"COLMAP on {len(frames)} frames (room3d.run_colmap)")
        try:
            dense = room3d.run_colmap(work, [[fr["name"] for fr in frames]], log=log)
        except RuntimeError as e:
            log(f"COLMAP failed: {str(e).splitlines()[0]}")
            dense = None
        geo = {"registered": 0, "kept": len(frames)}
        if dense is not None:
            try:
                geo = geometry(dense, frames, a.side_mm)
            except Exception as e:                      # still print the gates we have
                log(f"geometry failed: {e!r}")
                geo["registered"] = len(room3d.read_images(dense / "sparse" / "0" / "images.bin"))
        times["colmap"] = time.time() - t
        share = geo["registered"] / max(geo["kept"], 1)
        gate(">= 95% posed", share >= 0.95, f"{geo['registered']}/{geo['kept']} frames ({share:.0%})"
             + (f", {geo['models']} separate models" if geo.get("models", 1) > 1 else ""))
        if "height_span_units" in geo:
            span, mpu, med = geo["height_span_units"], geo.get("metres_per_unit"), geo.get("median_height_units")
            if mpu:
                gate("height span", span * mpu >= 1.0, f"{span * mpu:.2f} m between p5 and p95 (marker scale), "
                     f"median {med * mpu:.2f} m" if med else f"{span * mpu:.2f} m (marker scale)")
            elif med:
                gate("height span", span / med >= 0.7, f"{span / med:.2f}x the median camera height (want >= 0.7x, "
                     f"about 1.0 m at 1.4 m; no marker, so scale-free)")
            else:
                gate("height span", None, "floor not found, so heights can't be judged")
            p5, p50, p95 = geo["pitch_p5_p50_p95"]
            gate(">= 35 deg down", geo["down35"] >= 0.05, f"{geo['down35']:.0%} of frames (want >= 5%); pitch p5 {p5}, "
                 f"median {p50}, p95 {p95} deg")
            notes.append(f"{'ok  ' if geo['up10'] > 0 else 'WARN'}  looking up >= 10 deg: {geo['up10']:.0%} of frames")
            notes.append(f"{'ok  ' if geo['heading_sectors'] >= 9 else 'WARN'}  headings: {geo['heading_sectors']}/12 30-deg sectors, "
                         f"largest gap {geo['heading_gap_deg']:.0f} deg")
            lens = "0.5x ultra-wide" if geo["hfov_deg"] > 90 else "1x main" if geo["hfov_deg"] > 55 else "zoomed?"
            notes.append(f"info  lens: horizontal FOV {geo['hfov_deg']} deg ({lens}); undistorted frames keep {geo['undistorted_keeps']}")
            if geo["unposed_seconds"]:
                notes.append("WARN  not posed (seconds into the clip): "
                             + ", ".join(f"{a_}-{b_}" if a_ != b_ else f"{a_}" for a_, b_ in geo["unposed_seconds"][:12]))
            if mpu:
                notes.append(f"info  marker scale {mpu:.4f} m per COLMAP unit (sides agree to {geo['marker_spread']:.1%})")
        else:
            gate("height span", False, "nothing posed")
            gate(">= 35 deg down", False, "nothing posed")

    total = time.time() - T0
    print("\n" + "=" * 72)
    print(f"QUICK CHECK  {clip.name}  ({f['duration_s']:.0f} s of video, checked in {total / 60:.1f} min)")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL' if ok is False else '????'}  {name:15s} {detail}")
    for n in notes:
        print(f"  {n}")
    passed = all(ok for _, ok, _ in gates)
    print(f"RESULT: {'ALL GATES PASS' if passed else 'SOME GATES FAIL'}" + ("" if not a.no_colmap else " (no COLMAP run)"))
    print("=" * 72)
    st.pop("rows")
    (work / "quickcheck.json").write_text(json.dumps(
        {"clip": str(clip), "facts": f, "gates": [{"gate": g, "pass": ok, "detail": d} for g, ok, d in gates],
         "notes": notes, "decode": st, "flicker": fl, "geometry": geo, "seconds": {**times, "total": total}}, indent=1, default=str))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
