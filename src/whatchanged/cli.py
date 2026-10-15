"""whatchanged: two phone videos in, a 3D change report out.

  whatchanged run BEFORE.mp4 AFTER.mp4 --out out/room     full pipeline
  whatchanged check BEFORE.mp4 AFTER.mp4                  footage warnings, no 3D
  whatchanged report out/room                             rebuild everything after 3D from the cache
  whatchanged demo                                        the bundled example, no model download
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


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def _object_boxes(out: Path, A, B, enabled: bool):
    """Labelled 3D boxes for every object in each session (cached in objects.json)."""
    from . import objects3d
    path = out / "objects.json"
    if path.exists():
        return objects3d.load(path)
    if not enabled:
        return None
    try:
        boxes = {S.name: objects3d.boxes_for_session(S, n_frames=8, device="cpu", log=log)[0] for S in (A, B)}
    except ImportError as e:
        log(f"warning: object boxes skipped ({e})")
        return None
    objects3d.write(path, boxes)
    for name, key in (("A", "before"), ("B", "after")):
        (out / f"scene_graph_{key}.json").write_text(json.dumps(objects3d.scene_graph(boxes[name]), indent=1))
        (out / f"markers_{key}.json").write_text(json.dumps(objects3d.marker_array(boxes[name], f"objects_{key}")))
    return boxes


def _detect_and_report(out: Path, labels: str, cfg: ch.Config, objects: bool = True) -> dict:
    from . import floormap, lookahead, report, semantics
    from .geometry import write_ply
    from .recon import Session

    A = Session.load("A", out / "cache" / "A.npz")
    B = Session.load("B", out / "cache" / "B.npz")
    diag = json.loads((out / "cache" / "diag.json").read_text())
    res = ch.detect(A, B, cfg, log)
    boxes = _object_boxes(out, A, B, objects)
    found = semantics.describe(res, A, B, out, labels, log, boxes=boxes)
    regions = [r for r in res["regions_A"] if r["reason"] == "hidden"] + res["regions_A"][:3]
    seen_ids = set()
    regions = [r for r in regions if not (id(r) in seen_ids or seen_ids.add(id(r)))]
    log("Robot map: before / after / updated")
    floor = floormap.build(A, B, found, out / "maps")
    g = np.load(out / "maps" / "grids.npz")
    plans = lookahead.plan(A, B, res["regions_A"], g["free_both"], g["origin"], float(g["res"]), out, log=log)
    diag["noise_floor"] = {"A_to_B_sigma": round(res["VA"]["sigma"], 4), "B_to_A_sigma": round(res["VB"]["sigma"], 4)}
    diag["config"] = asdict(cfg)
    for name, S in (("A", A), ("B", B)):
        p, c = S.points(conf_pct=30, stride=2)
        write_ply(out / f"scene_{name}.ply", p, c)
    summary = {
        "changes": found,
        "cant_tell": [{"reason": r["reason"], "centre_m": np.round(r["centre"], 2).tolist(),
                       "min_m": np.round(r["min"], 2).tolist(), "max_m": np.round(r["max"], 2).tolist(), "voxels": r["n"]}
                      for r in regions],
        "floor_map": floor, "look_next": plans, "diagnostics": diag,
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
    from .recon import reconstruct

    out = Path(a.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    for w in check_pair(a.before, a.after):
        log(f"warning: {w}")
    if a.force or not (out / "cache" / "A.npz").exists():
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
    _detect_and_report(out, a.labels, _cfg(a), not a.no_objects)


def cmd_report(a) -> None:
    _detect_and_report(Path(a.out), a.labels, _cfg(a), not a.no_objects)


def cmd_check(a) -> None:
    ws = check_pair(a.before, a.after)
    print("\n".join(f"warning: {w}" for w in ws) or "Footage looks fine.")


def cmd_demo(a) -> None:
    root = Path(__file__).resolve().parents[2] / "examples" / "bedroom"
    if not (root / "cache" / "A.npz").exists():
        sys.exit(f"Example cache not found at {root}. Run `whatchanged run` on your own videos instead.")
    _detect_and_report(root, "auto", ch.Config())


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


def _cfg(a) -> ch.Config:
    cfg = ch.Config()
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
        sp.add_argument("--labels", default="auto", choices=["auto", "claude", "gemini", "siglip", "none"])
        sp.add_argument("--voxel", type=float)
        sp.add_argument("--min-views", dest="min_views", type=int)
        sp.add_argument("--free-share", dest="free_share", type=float)
        sp.add_argument("--k-sigma", dest="k_sigma", type=float)
        sp.add_argument("--no-objects", dest="no_objects", action="store_true", help="skip the labelled boxes for every object")

    r = sub.add_parser("run", help="full pipeline on two videos")
    r.add_argument("before")
    r.add_argument("after")
    r.add_argument("--out", default="out/run")
    r.add_argument("--frames", type=int, default=24, help="keyframes per video")
    r.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
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

    d = sub.add_parser("demo", help="run the bundled example from its cache")
    d.set_defaults(fn=cmd_demo)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
