"""whatchanged: two phone videos in, a 3D change report out.

  whatchanged run BEFORE.mp4 AFTER.mp4 --out out/room     full pipeline
  ... --poses colmap                                      cameras from COLMAP, shared with `whatchanged room`
  whatchanged check BEFORE.mp4 AFTER.mp4                  footage warnings, no 3D
  whatchanged report out/room                             rebuild everything after 3D from the cache
  whatchanged demo                                        the bundled example, no model download
  whatchanged room ROOM.MOV --out out/room3d              photoreal 3D room (Gaussian splat) + web viewer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import changes as ch
from .frames import check_pair, extract_keyframes, load_manifest, save_manifest

T0 = time.time()
OBJECT_FRAMES = 24   # keyframes per session for the labelled boxes (was 8: too few views per object on a long walk)
OBJECT_FRAMES_SHARED = 40   # keyframes per video when both videos' views are fused into one object set (~0.4 s each on MPS)


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _object_boxes(out: Path, A, B, enabled: bool, labels: str = "auto"):
    """Labelled 3D boxes for every object in each session (cached in objects.json).

    With ANTHROPIC_API_KEY (and --labels auto or claude), Claude adds the room's own nouns to the
    detector's queries, then verifies and names every box (naming.py). Boxes it drops stay in
    objects.json as "verified": false; the rest of the pipeline only gets the kept ones.
    """
    from . import naming, objects3d
    path = out / "objects.json"
    use_claude = labels in ("auto", "claude") and naming.available()
    if path.exists():
        done = json.loads(path.read_text()).get("stats", {}).get("naming", {})
        if not (use_claude and enabled) or done.get("label_source") == "claude" or done.get("error"):
            return objects3d.load(path)
        log("boxes: the cached boxes carry detector labels only; rebuilding them with Claude's room vocabulary")
    if not enabled:
        return None
    namer, vocab, roles = _claude_vocabulary(out, A, B, labels)
    try:
        devices = dict.fromkeys([objects3d.pick_device(), "cpu"])
    except ImportError as e:
        log(f"warning: object boxes skipped ({e})")
        return None
    for device in devices:                      # Apple's GPU first, the CPU if an op is missing there
        try:
            found = {S.name: objects3d.boxes_for_session(S, n_frames=OBJECT_FRAMES, device=device, vocab=vocab,
                                                         roles=roles, log=log) for S in (A, B)}
            break
        except ImportError as e:
            log(f"warning: object boxes skipped ({e})")
            return None
        except (RuntimeError, NotImplementedError) as e:
            if device == "cpu":
                raise
            log(f"warning: object boxes failed on {device} ({str(e).splitlines()[0]}); trying the CPU")
    boxes = {k: v[0] for k, v in found.items()}
    if namer and not namer.error:
        try:
            boxes = namer.name({"A": A, "B": B}, boxes)
        except Exception as e:  # noqa: BLE001
            log(f"warning: Claude box naming stopped ({type(e).__name__}: {e}); unnamed boxes keep detector labels")
    stats = {"detector": {k: v[1] for k, v in found.items()},
             "naming": namer.stats() if namer else {"label_source": "owlv2",
                                                    "reason": "no ANTHROPIC_API_KEY" if labels in ("auto", "claude") else f"--labels {labels}"}}
    objects3d.write(path, boxes, stats)
    boxes = {k: [b for b in v if b.verified is not False] for k, v in boxes.items()}
    _write_object_graphs(out, boxes)
    return boxes


def _claude_vocabulary(out: Path, A, B, labels: str):
    """The Namer (None without a key or with --labels other than auto/claude), the detector's queries with
    Claude's room nouns added, and their roles."""
    from . import naming, objects3d
    namer, vocab, roles = None, objects3d.VOCAB3D, {}
    if labels in ("auto", "claude") and naming.available():
        namer = naming.Namer(out, log=log)
        try:
            room = namer.vocabulary([A, B])
        except Exception as e:  # noqa: BLE001  (naming is a bonus; the boxes matter more)
            log(f"warning: Claude room vocabulary skipped ({type(e).__name__}: {e})")
            room = []
        vocab = objects3d.VOCAB3D + [o["name"] for o in room if o["name"] not in objects3d.VOCAB3D]
        roles = {o["name"]: o["kind"] for o in room}
    elif labels in ("auto", "claude"):
        log("boxes: detector labels only (no ANTHROPIC_API_KEY in the environment or .env); "
            f"with a key, {naming.MODEL} verifies and names every box")
    return namer, vocab, roles


def _regrow(namer, ctx: dict, boxes: list, views) -> list:
    """Boxes Claude called too small grow to their 2D detections (objects3d.grow_box) and Claude looks
    again: the ones it kept, and the ones it dropped only for holding too little of a real thing ("covers
    only the middle of a long radiator"). Boxes it called too big come in (objects3d.shrink_box): the kept ones,
    and the ones it dropped only for being too big ("spans from the door to the desk": the monitor). A second
    verdict that drops the box or overshoots the other way puts the old box and verdict back; a dropped box
    comes back if Claude now keeps it and it is no longer too big."""
    from . import naming, objects3d
    S, spans, sup = ctx["S"], ctx["spans"], ctx["supports"]
    real = lambda b: b.verified or (b.label_source == "claude" and b.label not in naming.NOT_OBJECTS  # noqa: E731
                                    and "empty" not in b.label)
    small = [b for b in boxes if real(b) and b.fit == "too_small" and b.dets and b.points is not None]
    keys = ("centre", "size", "yaw", "grown", "label", "verified", "part_of", "colour", "fit", "note", "crop",
            "label_source", "model")
    big = [b for b in boxes if b.verified and b.fit == "too_big" and b.dets and b.points is not None]
    # dropped only for being too big ("spans from the door to the desk"): come in to the 2D detections and ask again
    big += [b for b in boxes if b.verified is False and real(b) and b.fit == "too_big" and not b.part_of
            and b.dets and b.points is not None]
    was_dropped = {id(b) for b in big if b.verified is False}
    cache, grown, shrunk = {}, [], []
    for b in small:
        before = {k: getattr(b, k) for k in keys}
        if objects3d.grow_box(S, b, objects3d.free_frames(spans, b), sup, force=True, cache=cache, max_grow=1.5,
                              allow_place=not b.merged_from):
            grown.append((b, before))
    for b in big:
        before = {k: getattr(b, k) for k in keys}
        if objects3d.shrink_box(S, b):
            shrunk.append((b, before))
    if not grown and not shrunk:
        if small or big:
            log(f"boxes: none of the {len(small)} boxes Claude called too small or {len(big)} too big fit their 2D detections better")
        return boxes
    kept_before = [b for b in boxes if b.verified]
    namer.check(S, "shared", [b for b, _ in grown + shrunk], views, tag="_refit")
    back, grew = 0, {id(b) for b, _ in grown}
    for b, before in grown + shrunk:
        # a grown box stays unless it is now dropped or too big; a shrunk kept one only if it now fits; a shrunk
        # dropped one if Claude now keeps it, it is no longer too big (a kept, too-small box beats none), and no
        # kept box of its new name is already there (it would only widen that one when merged)
        if id(b) in was_dropped:
            undo = b.verified is not True or b.fit == "too_big" or any(
                k.label == b.label and naming._touch(k, b, 0.2, 0.4) for k in kept_before)
        else:
            undo = b.verified is False or (b.fit == "too_big" if id(b) in grew else b.fit != "good")
        if undo:
            for k, v in before.items():
                setattr(b, k, v)
            back += 1
    log(f"boxes: {len(grown)} of {len(small)} boxes Claude called too small grew to their 2D detections, "
        f"{len(shrunk)} of {len(big)} too big came in; checked again ({back} put back)")
    boxes, _ = naming.merge(boxes, room_yaw=ctx["room_yaw"])
    return boxes


def _shared_objects(out: Path, A, B, enabled: bool, labels: str = "auto"):
    """ONE set of labelled 3D boxes for both videos, when they share COLMAP cameras (objects3d.boxes_shared):
    each object is fused from both videos' views, so Before and After show the same static objects.
    Cached in objects.json ("shared"). Returns (every object, Claude's dropped ones too; stats), or None."""
    from . import naming, objects3d
    path = out / "objects.json"
    use_claude = labels in ("auto", "claude") and naming.available()
    if path.exists():
        doc = json.loads(path.read_text())
        done = doc.get("stats", {}).get("naming", {})
        if "shared" in doc and (not (use_claude and enabled) or done.get("label_source") == "claude" or done.get("error")):
            return objects3d.load_shared(path), doc.get("stats", {})
        if enabled:
            log("boxes: rebuilding the boxes as one object set for both videos" if "shared" not in doc else
                "boxes: the cached boxes carry detector labels only; rebuilding them with Claude's room vocabulary")
    if not enabled:
        return None
    namer, vocab, roles = _claude_vocabulary(out, A, B, labels)
    try:
        devices = dict.fromkeys([objects3d.pick_device(), "cpu"])
    except ImportError as e:
        log(f"warning: object boxes skipped ({e})")
        return None
    for device in devices:                      # Apple's GPU first, the CPU if an op is missing there
        try:
            boxes, dstats, ctx = objects3d.boxes_shared({"A": A, "B": B}, n_frames=OBJECT_FRAMES_SHARED, device=device,
                                                        vocab=vocab, roles=roles, log=log)
            break
        except ImportError as e:
            log(f"warning: object boxes skipped ({e})")
            return None
        except (RuntimeError, NotImplementedError) as e:
            if device == "cpu":
                raise
            log(f"warning: object boxes failed on {device} ({str(e).splitlines()[0]}); trying the CPU")

    def views(b):                               # show a box only from the videos that saw it
        return objects3d.free_frames(ctx["spans"], b, cap=10 ** 6)
    if namer and not namer.error:
        try:
            boxes = namer.name({"shared": ctx["S"]}, {"shared": boxes}, views={"shared": views}, room_yaw=ctx["room_yaw"])["shared"]
            boxes = _regrow(namer, ctx, boxes, views)
        except Exception as e:  # noqa: BLE001
            log(f"warning: Claude box naming stopped ({type(e).__name__}: {e}); unnamed boxes keep detector labels")
    try:                                        # a second copy of one object, lifted to the wrong place from a few views
        dstats["twins_dropped"] = objects3d.drop_twins(ctx["S"], ctx["spans"], boxes, ctx.get("dets", []), log=log)
    except Exception as e:  # noqa: BLE001  (a check; the boxes matter more)
        log(f"warning: duplicate check skipped ({type(e).__name__}: {e})")
    if namer and not namer.error:
        namer._tally({"shared": boxes})
    for i, b in enumerate(boxes):
        b.id = f"o{i:02d}"
        b.state = "both" if b.verified is not False else ""
    stats = {"detector": {"shared": dstats},
             "naming": namer.stats() if namer else {"label_source": "owlv2",
                                                    "reason": "no ANTHROPIC_API_KEY" if labels in ("auto", "claude") else f"--labels {labels}"}}
    objects3d.write_shared(path, boxes, stats)
    return boxes, stats


def _write_object_graphs(out: Path, boxes: dict) -> None:
    from . import objects3d
    for name, key in (("A", "before"), ("B", "after")):
        (out / f"scene_graph_{key}.json").write_text(json.dumps(objects3d.scene_graph(boxes[name]), indent=1))
        (out / f"markers_{key}.json").write_text(json.dumps(objects3d.marker_array(boxes[name], f"objects_{key}")))


def _detect_and_report(out: Path, labels: str, cfg: ch.Config, objects: bool = True) -> dict:
    from . import floormap, lookahead, naming, report, semantics
    from .geometry import write_ply
    from .recon import Session

    A = Session.load("A", out / "cache" / "A.npz")
    B = Session.load("B", out / "cache" / "B.npz")
    diag = json.loads((out / "cache" / "diag.json").read_text())
    res = ch.detect(A, B, cfg, log)
    baseline = ch.naive_diff(A, B, cfg)
    log(f"Baseline (plain point difference, no visibility): {len(baseline['removed'])} removed, {len(baseline['added'])} added")
    shared = None
    if diag.get("poses_from") == "colmap":      # one camera frame for both videos: one object set for both
        try:
            got = _shared_objects(out, A, B, objects, labels)
        except ValueError as e:
            log(f"warning: no shared object set ({e}); boxing each video on its own")
            got = None
        if got is not None:
            shared, object_stats = got
    if shared is not None:
        from . import objects3d
        objects3d.assign_states(shared, res, log=log)     # Before / After differ only where a change is
        kept = [b for b in shared if b.verified is not False]
        boxes = {n: objects3d.for_state(kept, n) for n in ("A", "B")}
    else:
        boxes = _object_boxes(out, A, B, objects, labels)
    found = semantics.describe(res, A, B, out, labels, log, boxes=boxes)
    if shared is not None:
        objects3d.link_changes(shared, found)
        objects3d.write_shared(out / "objects.json", shared, object_stats)
        _write_object_graphs(out, boxes)
    regions = [r for r in res["regions_A"] if r["reason"] == "hidden"] + res["regions_A"][:3]
    seen_ids = set()
    regions = [r for r in regions if not (id(r) in seen_ids or seen_ids.add(id(r)))]
    log("Robot map: before / after / updated")
    floor = floormap.build(A, B, found, out / "maps")
    g = np.load(out / "maps" / "grids.npz")
    plans = lookahead.plan(A, B, res["regions_A"], g["free_both"], g["origin"], float(g["res"]), out, log=log)
    diag["noise_floor"] = {"A_to_B_sigma": round(res["VA"]["sigma"], 4), "B_to_A_sigma": round(res["VB"]["sigma"], 4)}
    diag["smallest_detectable_change"] = {"removals (judged by B)": res["VA"]["error_budget"], "additions (judged by A)": res["VB"]["error_budget"]}
    diag["config"] = asdict(cfg)
    if (out / "objects.json").exists():
        diag["object_naming"] = json.loads((out / "objects.json").read_text()).get("stats", {}).get("naming", {})
    diag["claude_this_run"] = naming.USAGE.summary()
    if naming.USAGE.n["calls"]:
        log(f"Claude: {naming.USAGE.n['calls']} calls this run, about ${diag['claude_this_run']['cost_usd_est']:.2f}")
    for name, S in (("A", A), ("B", B)):
        p, c = S.points(conf_pct=30, stride=2)
        write_ply(out / f"scene_{name}.ply", p, c)
    summary = {
        "changes": found,
        "cant_tell": [{"reason": r["reason"], "centre_m": np.round(r["centre"], 2).tolist(),
                       "min_m": np.round(r["min"], 2).tolist(), "max_m": np.round(r["max"], 2).tolist(), "voxels": r["n"]}
                      for r in regions],
        "floor_map": floor, "look_next": plans, "baseline_naive": baseline, "diagnostics": diag,
    }
    (out / "changes.json").write_text(json.dumps(summary, indent=1))
    path = report.write(out, A, B, found, regions, floor, diag, plans, boxes=boxes)
    try:
        from . import rrd
        rrd.write(out / "scene.rrd", A, B, found, plans, boxes=boxes)
        log(f"Rerun recording: {out / 'scene.rrd'}")
    except Exception as e:  # the report matters more than the extra recording
        log(f"warning: Rerun export skipped ({e})")
    log(f"Report: {path}")
    for c in found:
        print(f"  {c['id']} {c['type']:8s} {c['sentence']}")
    return summary


def cmd_run(a) -> None:
    from .frames import Frame
    from .recon import reconstruct, reconstruct_colmap

    out = Path(a.out)
    if a.metres_per_unit and a.poses != "colmap":
        sys.exit("--metres-per-unit is metres per COLMAP unit, so it needs --poses colmap")
    (out / "cache").mkdir(parents=True, exist_ok=True)
    for w in check_pair(a.before, a.after):
        log(f"warning: {w}")
    work = None
    if a.poses == "colmap" and a.colmap_from:
        work = Path(a.colmap_from)
        work = work / "work" if (work / "work" / "frames.json").exists() else work
    stale = None if a.force else _stale_cache(out, a.poses, work)
    if stale:
        log(f"3D: {stale}, so the cached reconstruction is redone")
    if a.force or stale or not (out / "cache" / "A.npz").exists():
        if a.poses == "colmap":
            A, B, diag, keys = reconstruct_colmap([a.before, a.after], out, n_frames=a.frames, device=a.device,
                                                  work=work, metres_per_unit=a.metres_per_unit, log=log)
            save_manifest([Frame(**k) for k in keys], out / "frames.json")
        else:
            log(f"Frames: {a.frames} sharp keyframes per video")
            fa = extract_keyframes(a.before, out / "frames", "A", n=a.frames)
            fb = extract_keyframes(a.after, out / "frames", "B", n=a.frames)
            save_manifest(fa + fb, out / "frames.json")
            A, B, diag = reconstruct(fa, fb, device=a.device, log=log)
        (out / "objects.json").unlink(missing_ok=True)
        A.save(out / "cache" / "A.npz")
        B.save(out / "cache" / "B.npz")
        diag["inputs"] = {"before": str(a.before), "after": str(a.after)}
        (out / "cache" / "diag.json").write_text(json.dumps(diag, indent=1))
    else:
        log("3D: using cached reconstruction (pass --force to redo)")
        if a.metres_per_unit:
            cached = json.loads((out / "cache" / "diag.json").read_text()).get("colmap", {}).get("metric_scale_m_per_unit")
            if not cached or abs(cached / a.metres_per_unit - 1) > 0.001:
                log(f"warning: the cached run used {cached} m per COLMAP unit, not {a.metres_per_unit}; pass --force to rescale")
    _detect_and_report(out, a.labels, _cfg(a), not a.no_objects)


def _stale_cache(out: Path, poses: str, work: Path | None) -> str | None:
    """Why the cached 3D in `out` no longer fits the inputs (None if it does, or if there is none).

    With --poses colmap the cache is depth fitted to one COLMAP model. If that model was rebuilt (a room
    moved aside and made again at another frame rate) or another one is given, the old cache would
    put every box and change in the old model's frame, off the new splats.
    """
    path = out / "cache" / "diag.json"
    if not (out / "cache" / "A.npz").exists() or not path.exists():
        return None
    diag = json.loads(path.read_text())
    if diag.get("poses_from", "mapanything") != poses:
        return f"the cache has {diag.get('poses_from', 'mapanything')} poses, not {poses}"
    if poses != "colmap":
        return None
    work = Path(work) if work else out / "colmap"
    was = diag.get("colmap", {}).get("work")
    if was and Path(was).resolve() != work.resolve():
        return f"the cache was fitted to the COLMAP model in {was}, not {work}"
    model = work / "dense" / "sparse" / "0" / "images.bin"
    if model.exists() and model.stat().st_mtime > path.stat().st_mtime:
        return f"the COLMAP model in {work} is newer than the cache"
    return None


def cmd_report(a) -> None:
    _detect_and_report(Path(a.out), a.labels, _cfg(a), not a.no_objects)


def cmd_check(a) -> None:
    ws = check_pair(a.before, a.after)
    print("\n".join(f"warning: {w}" for w in ws) or "Footage looks fine.")


def cmd_demo(a) -> None:
    repo = Path(__file__).resolve().parents[2]
    root = repo / "examples" / "bedroom"
    if not (root / "cache" / "A.npz").exists():
        sys.exit(f"Example cache not found at {root}. Run `whatchanged run` on your own videos instead.")
    a.config = str(repo / "config.frozen.json") if (repo / "config.frozen.json").exists() else None
    _detect_and_report(root, "auto", _cfg(a))


def cmd_relevel(a) -> None:
    """Redo the floor levelling and B-to-A alignment on a cached run (no 3D model needed)."""
    from .recon import Session, level_and_align
    out = Path(a.out)
    A = Session.load("A", out / "cache" / "A.npz")
    B = Session.load("B", out / "cache" / "B.npz")
    diag = json.loads((out / "cache" / "diag.json").read_text())
    if diag.get("poses_from") == "colmap":
        sys.exit("This run's cameras come from COLMAP and are levelled with the splat viewer; "
                 "redo it with `whatchanged run ... --poses colmap --force` instead.")
    level_and_align(A, B, diag, log)
    A.save(out / "cache" / "A.npz")
    B.save(out / "cache" / "B.npz")
    (out / "cache" / "diag.json").write_text(json.dumps(diag, indent=1))
    (out / "objects.json").unlink(missing_ok=True)


def cmd_room(a) -> None:
    from .room3d import build
    build(a.videos, Path(a.out), fps=a.fps, max_side=a.max_side, iterations=a.iters, sh_degree=a.sh,
          camera_height=a.camera_height, detections=a.detections, force=a.force,
          metres_per_unit=a.metres_per_unit, log=log)


def cmd_video(a) -> None:
    from .video import make_video
    room = Path(a.room)
    make_video(room, Path(a.detections) if a.detections else None, Path(a.out) if a.out else room / "whatchanged.mp4",
               use_claude=not a.no_claude, footage=not a.no_footage, tour=not a.no_tour,
               register=not a.no_register, reg_fps=a.reg_fps, log=log)


def cmd_score(a) -> None:
    from .score import markdown, score

    s = score(Path(a.out) / "changes.json", Path(a.key))
    md = markdown(s, a.title or Path(a.out).resolve().name)
    (Path(a.out) / "scorecard.json").write_text(json.dumps(s, indent=1))
    (Path(a.out) / "scorecard.md").write_text(md)
    print(md)
    if a.min_recall is not None and s["recall"] < a.min_recall:
        sys.exit(f"recall {s['recall']} below {a.min_recall}")
    if a.max_false is not None and s["false_alarms"] > a.max_false:
        sys.exit(f"{s['false_alarms']} false alarms, more than {a.max_false}")


def cmd_calibrate(a) -> None:
    """Pick the most sensitive thresholds that report NOTHING on an untouched pair, plus a margin."""
    import itertools
    import subprocess
    from datetime import datetime, timezone

    from .recon import Session
    out = Path(a.out)
    A = Session.load("A", out / "cache" / "A.npz")
    B = Session.load("B", out / "cache" / "B.npz")
    ks, shares, views = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0], [0.5, 0.6, 0.7], [3, 4, 5]
    grid = []
    for k, f, v in itertools.product(ks, shares, views):
        cfg = ch.Config(k_sigma=k, free_share=f, min_views=v)
        r = ch.detect(A, B, cfg, log=lambda *_: None)
        n = len(r["removed"]) + len(r["added"])
        grid.append({"k_sigma": k, "free_share": f, "min_views": v, "false_blobs": n})
        print(f"  k={k:<4} share={f:<4} views={v}: {n} changes", flush=True)
    clean = [g for g in grid if g["false_blobs"] == 0]
    if not clean:
        sys.exit("Even the strictest setting reports changes on the untouched pair; check alignment first.")
    # most sensitive = lowest k, then lowest share, then fewest views; then one step of margin on k
    best = min(clean, key=lambda g: (g["k_sigma"], g["free_share"], g["min_views"]))
    k_next = next((k for k in ks if k > best["k_sigma"]), best["k_sigma"])
    chosen = {"k_sigma": k_next, "free_share": best["free_share"], "min_views": best["min_views"]}
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc = {"config": chosen, "chosen_from": out.name, "rule": "most sensitive setting with 0 changes on the untouched pair, then one step stricter on k_sigma",
           "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "code_commit": sha, "grid": grid}
    Path(a.write).write_text(json.dumps(doc, indent=1))
    print(f"Chosen: {chosen} -> {a.write}. Commit this file BEFORE running the real before/after pair.")


def _cfg(a) -> ch.Config:
    cfg = ch.Config()
    path = getattr(a, "config", None)
    if path:
        frozen = json.loads(Path(path).read_text())["config"]
        for k, v in frozen.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        log(f"Config: {path} (frozen {json.loads(Path(path).read_text()).get('frozen_at', '')})")
    for k in ("voxel", "min_views", "free_share", "k_sigma"):
        v = getattr(a, k, None)
        if v is not None:
            setattr(cfg, k, v)
    return cfg


def _load_dotenv() -> None:
    """Read KEY=VALUE lines from ./.env (git-ignored) without overriding the real environment."""
    import os
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if path.exists():
            for line in path.read_text().splitlines():
                k, sep, v = line.partition("=")
                if sep and k.strip() and not k.strip().startswith("#"):
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


def main(argv=None) -> None:
    _load_dotenv()
    p = argparse.ArgumentParser(prog="whatchanged", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def tuning(sp):
        sp.add_argument("--labels", default="auto", choices=["auto", "claude", "gemini", "siglip", "none"],
                        help="auto: with ANTHROPIC_API_KEY, Claude names the object boxes and the changes")
        sp.add_argument("--voxel", type=float)
        sp.add_argument("--min-views", dest="min_views", type=int)
        sp.add_argument("--free-share", dest="free_share", type=float)
        sp.add_argument("--k-sigma", dest="k_sigma", type=float)
        sp.add_argument("--no-objects", dest="no_objects", action="store_true", help="skip the labelled boxes for every object")
        sp.add_argument("--config", help="frozen thresholds from `whatchanged calibrate`")

    r = sub.add_parser("run", help="full pipeline on two videos")
    r.add_argument("before")
    r.add_argument("after")
    r.add_argument("--out", default="out/run")
    r.add_argument("--frames", type=int, default=32, help="keyframes per video")
    r.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    r.add_argument("--poses", default="mapanything", choices=["mapanything", "colmap"],
                   help="where the cameras come from; colmap shares them with `whatchanged room`, so boxes land on the splats")
    r.add_argument("--colmap-from", dest="colmap_from",
                   help="with --poses colmap: reuse this `whatchanged room BEFORE AFTER` output instead of running COLMAP")
    r.add_argument("--metres-per-unit", dest="metres_per_unit", type=float,
                   help="with --poses colmap: metric scale from the floor marker (scripts/marker_scale.py) instead of MapAnything depth")
    r.add_argument("--force", action="store_true")
    tuning(r)
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("check", help="footage warnings only")
    c.add_argument("before")
    c.add_argument("after")
    c.set_defaults(fn=cmd_check)

    rp = sub.add_parser("report", help="rebuild detection, labels, maps and report from a cached run")
    rp.add_argument("out")
    tuning(rp)
    rp.set_defaults(fn=cmd_report)

    s = sub.add_parser("score", help="compare a run with a hand-written answer key")
    s.add_argument("out")
    s.add_argument("key", help="ground_truth.yaml")
    s.add_argument("--title")
    s.add_argument("--min-recall", dest="min_recall", type=float, help="fail below this recall (for CI)")
    s.add_argument("--max-false", dest="max_false", type=int, help="fail above this many false alarms (for CI)")
    s.set_defaults(fn=cmd_score)

    rl = sub.add_parser("relevel", help="redo floor levelling and alignment on a cached run")
    rl.add_argument("out")
    rl.set_defaults(fn=cmd_relevel)

    cb = sub.add_parser("calibrate", help="set thresholds from an untouched pair (must report nothing)")
    cb.add_argument("out", help="a run of the untouched pair, e.g. out/room-repeat")
    cb.add_argument("--write", default="config.frozen.json")
    cb.set_defaults(fn=cmd_calibrate)

    rm = sub.add_parser("room", help="photoreal 3D room (Gaussian splat) from one or more videos, with a web viewer")
    rm.add_argument("videos", nargs="+", help="one video, or BEFORE and AFTER (posed together, one splat each)")
    rm.add_argument("--out", default="out/room3d")
    rm.add_argument("--fps", type=float, default=4.0, help="frames kept per second of video")
    rm.add_argument("--max-side", dest="max_side", type=int, default=1600, help="longest image side in pixels")
    rm.add_argument("--iters", type=int, default=30000, help="splat training steps (30000: about 12 min on an M4 Pro)")
    rm.add_argument("--sh", type=int, default=3, help="spherical-harmonic degree kept in the .spz (0-3)")
    rm.add_argument("--camera-height", dest="camera_height", type=float, default=1.4, help="median filming height in metres (sets the scale)")
    rm.add_argument("--detections", help="a `whatchanged run` folder on the same videos: metric scale, plus its object and change boxes "
                    "(a --poses colmap run on BEFORE AFTER also lends its COLMAP model)")
    rm.add_argument("--metres-per-unit", dest="metres_per_unit", type=float,
                    help="metric scale from the floor marker (scripts/marker_scale.py); overrides --detections' depth and --camera-height")
    rm.add_argument("--force", action="store_true", help="redo frames, poses and training")
    rm.set_defaults(fn=cmd_room)

    vd = sub.add_parser("video", help="results video (mp4): title, the real footage with tracked 3D boxes, the 3D tour, end card")
    vd.add_argument("room", help="a `whatchanged room A B --detections RUN` folder (scene.json, index.html)")
    vd.add_argument("--detections", help="the `whatchanged run` folder: measured facts for the summary line, and the COLMAP model "
                    "and videos for the footage part (without it the footage part is skipped unless the room has its own COLMAP work folder)")
    vd.add_argument("--out", help="the mp4 to write (default ROOM/whatchanged.mp4; the parts go next to it in <name>_parts/)")
    vd.add_argument("--no-claude", dest="no_claude", action="store_true", help="summary line from the template, no API call")
    vd.add_argument("--no-footage", dest="no_footage", action="store_true", help="leave out the real-footage part")
    vd.add_argument("--no-tour", dest="no_tour", action="store_true", help="leave out the headless-Chrome 3D tour")
    vd.add_argument("--no-register", dest="no_register", action="store_true",
                    help="poses from the model's keyframes only (no colmap image_registrator; boxes fade between sparse keyframes)")
    vd.add_argument("--reg-fps", dest="reg_fps", type=float, default=10.0, help="frames per second registered inside the shown stretches")
    vd.set_defaults(fn=cmd_video)

    d = sub.add_parser("demo", help="run the bundled example from its cache")
    d.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
