"""The guards and overrides around COLMAP, with no COLMAP, model or GPU: videos that did not join stop
before any training or depth inference, the incremental mapper is skipped on big runs, a marker scale
overrides the others, the viewer gets its field of view and photoreal gate, a partly collapsed model stops
before anything is built on it, and splats or depth cached from an older COLMAP model are made again."""

import json
import os
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from whatchanged import cli, objects3d, recon, room3d
from whatchanged.room3d import camera_meta, check_joined, registered_counts, viewer_frame, write_images

CAM = (1600, 900, 1250.0, 1255.0, 800.0, 450.0)   # PINHOLE width, height, fx, fy, cx, cy


def _write_model(sp, c2w, names, pts):
    """A tiny COLMAP model: one PINHOLE camera, these camera-to-world poses, points seen by two images."""
    sp.mkdir(parents=True, exist_ok=True)
    w, h, *params = CAM
    (sp / "cameras.bin").write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, w, h) + struct.pack("<4d", *params))
    imgs = {}
    for i, (T, n) in enumerate(zip(c2w, names), 1):
        R = T[:3, :3].T                                  # COLMAP stores world-to-camera
        x, y, z, qw = Rotation.from_matrix(R).as_quat()
        imgs[i] = {"name": n, "camera_id": 1, "q": (qw, x, y, z), "t": tuple(-R @ T[:3, 3])}
    write_images(sp / "images.bin", imgs)
    track = np.array([[1, 0], [2, 0]], "<u4").tobytes()
    recs = b"".join(struct.pack("<Q3d3BdQ", k, *p, 128, 128, 128, 0.5, 2) + track for k, p in enumerate(pts, 1))
    (sp / "points3D.bin").write_bytes(struct.pack("<Q", len(pts)) + recs)


def _room(s=0.3, seed=1):
    """A tilted COLMAP-like world at scale s: floor, a wall, and 30 cameras circling 1.5 m up."""
    rng = np.random.default_rng(seed)
    a = np.radians(25)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    floor = np.c_[rng.uniform(-2, 2, 3000), np.zeros(3000), rng.uniform(-2, 2, 3000)]
    wall = np.c_[np.full(800, 2.0), rng.uniform(0, 2.5, 800), rng.uniform(-2, 2, 800)]
    pts = (np.r_[floor, wall] @ tilt.T) * s
    cams = []
    for t in np.linspace(0, 2 * np.pi, 30, endpoint=False):
        T = np.eye(4)
        fwd, down = np.array([np.cos(t), 0, np.sin(t)]), np.array([0, -1.0, 0])
        T[:3, :3] = tilt @ np.stack([np.cross(down, fwd), down, fwd], 1)
        T[:3, 3] = tilt @ np.array([0.5 * np.cos(t), 1.5, 0.5 * np.sin(t)]) * s
        cams.append(T)
    return np.stack(cams), pts


def _frames(prefix, n):
    return [{"name": f"{prefix}/{prefix}_{i:04d}.jpg", "video": f"{prefix}.mov", "t": i * 0.25, "sharpness": 1.0}
            for i in range(n)]


# ---------------------------------------------------------------- (1) the videos must join

def test_join_check_stops_when_a_video_is_mostly_missing():
    groups = {"A": [f"A/{i}" for i in range(100)], "B": [f"B/{i}" for i in range(80)]}
    posed = {f"A/{i}" for i in range(97)} | {f"B/{i}" for i in range(30)}
    counts = registered_counts(groups, posed)
    assert counts == {"A": (97, 100), "B": (30, 80)}
    with pytest.raises(SystemExit, match="the videos did not join.*B under 50%.*drop it"):
        check_joined(counts, "drop it")
    check_joined({"A": (97, 100), "B": (40, 80)}, "fine")          # exactly half is enough
    check_joined({"room": (10, 100)}, "one video has nothing to join")


def _fake_colmap(calls, posed_share):
    """Stands in for room3d._colmap: the global mapper poses only the first share of every video."""
    def run(*args, log=print):
        cmd, a = args[0], dict(zip(map(str, args[1::2]), args[2::2]))
        calls.append(cmd)
        if cmd in ("global_mapper", "mapper"):
            names = sorted(p.relative_to(a["--image_path"]).as_posix() for p in a["--image_path"].rglob("*.jpg"))
            share = posed_share if cmd == "global_mapper" else 1.0
            groups = {}
            for n in names:
                groups.setdefault(n.split("/")[0], []).append(n)
            keep = [n for g in groups.values() for n in g[:int(share * len(g))]]
            write_images_at(a["--output_path"] / "0", keep)
        elif cmd == "image_undistorter":
            (a["--output_path"] / "sparse").mkdir(parents=True)
            for f in (a["--input_path"]).glob("*.bin"):
                (a["--output_path"] / "sparse" / f.name).write_bytes(f.read_bytes())
    return run


def write_images_at(sp, names):
    sp.mkdir(parents=True, exist_ok=True)
    write_images(sp / "images.bin", {i: {"name": n, "camera_id": 1, "q": (1.0, 0.0, 0.0, 0.0), "t": (0.0, 0.0, 0.0)}
                                     for i, n in enumerate(names, 1)})


@pytest.mark.parametrize("per_video,incremental", [(150, True), (250, False)])
def test_incremental_mapper_only_on_small_runs(tmp_path, monkeypatch, per_video, incremental):
    groups = []
    for v in "AB":
        (tmp_path / "images" / v).mkdir(parents=True)
        groups.append([f"{v}/{v}_{i:04d}.jpg" for i in range(per_video)])
        for n in groups[-1]:
            (tmp_path / "images" / n).touch()
    calls, lines = [], []
    monkeypatch.setattr(room3d, "_colmap", _fake_colmap(calls, posed_share=0.6))
    room3d.run_colmap(tmp_path, groups, log=lines.append)
    assert ("mapper" in calls) == incremental                     # 300 frames retry; 500 do not
    k = per_video if incremental else int(0.6 * per_video)
    assert f"  per video: A {k}/{per_video}, B {k}/{per_video}" in lines


def test_reconstruct_colmap_stops_before_any_depth_inference(tmp_path, monkeypatch):
    work = tmp_path / "pair" / "work"
    frames = {"A": _frames("A", 40), "B": _frames("B", 40)}
    (work).mkdir(parents=True)
    (work / "frames.json").write_text(json.dumps(frames))
    c2w, pts = _room()
    names = [f["name"] for f in frames["A"][:28]] + [f["name"] for f in frames["B"][:2]]   # B barely made it in
    _write_model(work / "dense" / "sparse" / "0", c2w, names, pts)
    monkeypatch.setattr(recon, "_infer", lambda *a, **k: pytest.fail("MapAnything ran"))
    monkeypatch.setattr(recon, "pick_device", lambda *a, **k: pytest.fail("got as far as picking a device"))
    with pytest.raises(SystemExit, match=r"A 28/40, B 2/40 frames .* did not join"):
        recon.reconstruct_colmap(["A.mov", "B.mov"], tmp_path / "run", work=work, log=lambda *_: None)


# ---------------------------------------------------------------- (3) marker scale, (4) viewer meta

def _room_folder(out, names_per_video, registered):
    """A `whatchanged room` output with frames, a COLMAP model and a trained splat, so build() only rebuilds."""
    work = out / "work"
    frames = {v: _frames(v, n) for v, n in names_per_video.items()}
    (work / "dense" / "images").mkdir(parents=True)
    (work / "frames.json").write_text(json.dumps(frames))
    c2w, pts = _room()
    names = [f["name"] for v, fs in frames.items() for f in fs[:registered[v]]]
    _write_model(work / "dense" / "sparse" / "0", c2w[:len(names)], names, pts)
    for v in frames:
        g = np.zeros(400, dtype=[(k, "<f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                                       "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")])
        g["x"], g["y"], g["z"] = pts[:400].T
        g["opacity"], g["rot_0"] = 3.0, 1.0
        g["scale_0"] = g["scale_1"] = g["scale_2"] = -5.0
        head = "ply\nformat binary_little_endian 1.0\nelement vertex 400\n" + "".join(
            f"property float {k}\n" for k in g.dtype.names) + "end_header\n"
        (work / f"{v}.ply").write_bytes(head.encode() + g.tobytes())
    return out


def test_build_meta_has_fov_gate_and_marker_scale(tmp_path):
    out = _room_folder(tmp_path / "room3d", {"room": 30}, {"room": 30})
    quiet = lambda *_: None  # noqa: E731
    room3d.build(["room.mov"], out, log=quiet)
    meta = json.loads((out / "scene.json").read_text())["meta"]
    assert meta["fov_x_deg"] == pytest.approx(np.degrees(2 * np.arctan(800 / 1250)), abs=0.01)
    assert meta["fov_y_deg"] == pytest.approx(np.degrees(2 * np.arctan(450 / 1255)), abs=0.01)
    assert meta["image_wh"] == [1600, 900] and meta["gate"] == {"R_m": 0.10, "theta_deg": 5}
    assert meta["frame"]["scale_from"] == "camera height 1.4 m" and meta["frame"]["median_camera_height_m"] == pytest.approx(1.4, abs=0.01)
    room3d.build(["room.mov"], out, metres_per_unit=1 / 0.3, log=quiet)   # the marker says 0.3 COLMAP units per metre
    meta = json.loads((out / "scene.json").read_text())["meta"]
    assert meta["frame"]["scale_from"] == "marker"
    assert meta["frame"]["median_camera_height_m"] == pytest.approx(1.5, abs=0.02)


def test_build_stops_before_training_when_the_videos_did_not_join(tmp_path):
    out = _room_folder(tmp_path / "pair", {"A": 20, "B": 20}, {"A": 20, "B": 5})
    for v in "AB":
        (out / "work" / f"{v}.ply").unlink()                    # training would be next
    with pytest.raises(SystemExit, match=r"A 20/20, B 5/20 frames .* did not join"):
        room3d.build(["A.mov", "B.mov"], out, log=lambda *_: None)
    assert not (out / "scene.json").exists()


def test_viewer_frame_names_its_scale():
    c2w, pts = _room()
    assert viewer_frame(c2w, pts, scale=2.0, source="marker")["scale_from"] == "marker"
    assert viewer_frame(c2w, pts, scale=2.0)["scale_from"] == "metric depth"
    assert viewer_frame(c2w, pts, camera_height=1.5)["scale_from"] == "camera height 1.5 m"


def test_camera_meta_from_pinhole():
    m = camera_meta({"model": 1, "width": 1000, "height": 500, "params": (500.0, 250.0, 500.0, 250.0)})
    assert m == {"fov_y_deg": 90.0, "fov_x_deg": 90.0, "image_wh": [1000, 500]}


def test_metres_per_unit_reaches_room_and_run(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(room3d, "build", lambda *a, **k: got.update(room=k["metres_per_unit"]))
    cli.main(["room", "a.mov", "--out", str(tmp_path / "r"), "--metres-per-unit", "0.512"])
    assert got["room"] == 0.512

    class Stop(Exception):
        pass

    def fake(videos, out, **k):
        got["run"] = k["metres_per_unit"]
        raise Stop
    monkeypatch.setattr(cli, "check_pair", lambda *a: [])
    monkeypatch.setattr(recon, "reconstruct_colmap", fake)
    with pytest.raises(Stop):
        cli.main(["run", "a.mov", "b.mov", "--out", str(tmp_path / "x"), "--poses", "colmap", "--metres-per-unit", "0.25"])
    assert got["run"] == 0.25
    with pytest.raises(SystemExit, match="needs --poses colmap"):
        cli.main(["run", "a.mov", "b.mov", "--out", str(tmp_path / "y"), "--metres-per-unit", "0.25"])


# ---------------------------------------------------------------- (2) object boxes: more keyframes, GPU first

def test_object_boxes_use_24_frames_and_fall_back_to_cpu(tmp_path, monkeypatch):
    seen = []

    def boxes(S, n_frames, device, log, **kw):     # kw: vocab and roles from Claude's room list, when there is a key
        seen.append((S.name, n_frames, device))
        if device == "mps":
            raise RuntimeError("op not implemented for 'mps'\nmore detail")
        return [], {}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(objects3d, "pick_device", lambda: "mps")
    monkeypatch.setattr(objects3d, "boxes_for_session", boxes)

    class S:
        def __init__(self, name):
            self.name = name
    out = cli._object_boxes(tmp_path, S("A"), S("B"), True)
    assert out == {"A": [], "B": []} and (tmp_path / "objects.json").exists()
    assert seen == [("A", 24, "mps"), ("A", 24, "cpu"), ("B", 24, "cpu")]


# ---------------------------------------------------------------- (5) a model that partly collapsed

def _walk(n=30, collapsed=(), depth=3.0, rng_seed=0):
    """n cameras along a line looking +z, each seeing 60 points of its own `depth` ahead. Cameras in
    `collapsed` and their points are shrunk 100,000x towards the origin, as the global mapper did on the
    rehearsal clips. Returns COLMAP-style images, points and tracks."""
    rng = np.random.default_rng(rng_seed)
    imgs, xyz, seen = {}, [], []
    for i in range(n):
        c = np.array([0.1 * i, 0.0, 0.0])
        p = c + np.c_[rng.uniform(-1, 1, 60), rng.uniform(-1, 1, 60), rng.uniform(0.8, 1.2, 60) * (depth[i] if np.ndim(depth) else depth)]
        if i in collapsed:
            c, p = c * 1e-5, p * 1e-5
        imgs[i + 1] = {"name": f"A/A_{i:04d}.jpg", "camera_id": 1, "q": (1.0, 0.0, 0.0, 0.0), "t": tuple(-c)}
        xyz.append(p)
        seen += [np.array([i + 1], np.uint32)] * len(p)
    return imgs, np.concatenate(xyz), seen


def test_scale_check_passes_a_normal_walk_with_near_and_far_frames():
    depth = np.full(30, 3.0)
    depth[:3], depth[-3:] = 0.4, 6.0                     # a close-up of the desk, a look across the room
    d = room3d.frame_depths(*_walk(depth=depth))
    assert len(d) == 30 and d["A/A_0000.jpg"] == pytest.approx(0.4, rel=0.1)
    assert room3d.check_scale(d, "hint") == set()


def test_scale_check_stops_on_a_collapsed_model_and_drops_a_few_stray_frames():
    with pytest.raises(SystemExit, match=r"12 of 30 posed frames .* collapsed\); hint"):
        room3d.check_scale(room3d.frame_depths(*_walk(collapsed=range(12))), "hint")
    assert room3d.check_scale(room3d.frame_depths(*_walk(collapsed=(4, 9))), "hint") == {"A/A_0004.jpg", "A/A_0009.jpg"}


def test_run_stops_on_a_collapsed_model_before_any_depth_inference(tmp_path, monkeypatch):
    work = tmp_path / "pair" / "work"
    frames = {"A": _frames("A", 30), "B": _frames("B", 30)}
    work.mkdir(parents=True)
    (work / "frames.json").write_text(json.dumps(frames))
    imgs_a, xyz_a, seen_a = _walk(collapsed=range(10, 30))
    imgs = {**imgs_a, **{i + 30: {**im, "name": im["name"].replace("A", "B")} for i, im in imgs_a.items()}}
    sp = work / "dense" / "sparse" / "0"
    sp.mkdir(parents=True)
    w, h, *params = CAM
    (sp / "cameras.bin").write_bytes(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, w, h) + struct.pack("<4d", *params))
    write_images(sp / "images.bin", imgs)
    pts = np.r_[xyz_a, xyz_a]
    tracks = [np.array([[i, 0]], "<u4") for s in seen_a for i in s] + [np.array([[i + 30, 0]], "<u4") for s in seen_a for i in s]
    (sp / "points3D.bin").write_bytes(struct.pack("<Q", len(pts)) + b"".join(
        struct.pack("<Q3d3BdQ", k, *p, 128, 128, 128, 0.5, 1) + t.tobytes() for k, (p, t) in enumerate(zip(pts, tracks), 1)))
    monkeypatch.setattr(recon, "_infer", lambda *a, **k: pytest.fail("MapAnything ran"))
    monkeypatch.setattr(recon, "pick_device", lambda *a, **k: pytest.fail("got as far as picking a device"))
    with pytest.raises(SystemExit, match=r"COLMAP's model is broken: 20 of 60 posed frames"):
        recon.reconstruct_colmap(["A.mov", "B.mov"], tmp_path / "run", work=work, log=lambda *_: None)


# ---------------------------------------------------------------- (6) caches built on an older COLMAP model

def test_room_retrains_a_splat_when_its_colmap_model_is_newer(tmp_path, monkeypatch):
    out = _room_folder(tmp_path / "room3d", {"room": 30}, {"room": 30})
    ply = out / "work" / "room.ply"
    trained = []

    def train(ds, out_ply, iterations, log=print):
        trained.append(out_ply.name)
        os.utime(out_ply)                                  # as if written again
        return {"steps": iterations}
    monkeypatch.setattr(room3d, "train_splat", train)
    quiet = lambda *_: None  # noqa: E731
    room3d.build(["room.mov"], out, log=quiet)
    assert trained == []                                   # the splat is newer than the model: kept
    t = (out / "work" / "dense" / "sparse" / "0" / "images.bin").stat().st_mtime
    os.utime(ply, (t - 60, t - 60))                        # the model was rebuilt after the splat was trained
    room3d.build(["room.mov"], out, log=quiet)
    assert trained == ["room.ply"]
    room3d.build(["room.mov"], out, log=quiet)
    assert trained == ["room.ply"]


def _cached_run(out, work, poses="colmap"):
    (out / "cache").mkdir(parents=True)
    (out / "cache" / "A.npz").write_bytes(b"")
    diag = {"poses_from": poses}
    if poses == "colmap":
        diag["colmap"] = {"work": str(work.resolve())}
    (out / "cache" / "diag.json").write_text(json.dumps(diag))


def test_run_cache_is_redone_when_its_colmap_model_changed(tmp_path):
    work = tmp_path / "pair" / "work"
    model = work / "dense" / "sparse" / "0" / "images.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")
    t = model.stat().st_mtime
    out = tmp_path / "real"
    _cached_run(out, work)
    assert cli._stale_cache(out, "colmap", work) is None
    assert cli._stale_cache(tmp_path / "nothing-cached", "colmap", work) is None
    os.utime(model, (t + 60, t + 60))                      # the room was rebuilt after this run
    assert "newer than the cache" in cli._stale_cache(out, "colmap", work)
    other = tmp_path / "pair2" / "work"
    assert "not " + str(other) in cli._stale_cache(out, "colmap", other)
    assert "not mapanything" in cli._stale_cache(out, "mapanything", None)
    mapa = tmp_path / "old"
    _cached_run(mapa, None, poses="mapanything")
    assert cli._stale_cache(mapa, "mapanything", None) is None


def test_run_redoes_the_3d_when_the_colmap_model_changed(tmp_path, monkeypatch):
    work = tmp_path / "pair" / "work"
    (work / "frames.json").parent.mkdir(parents=True)
    (work / "frames.json").write_text("{}")
    model = work / "dense" / "sparse" / "0" / "images.bin"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")
    out = tmp_path / "real"
    _cached_run(out, work)
    t = (out / "cache" / "diag.json").stat().st_mtime
    os.utime(model, (t + 60, t + 60))

    class Stop(Exception):
        pass

    def fake(videos, out_, **k):
        assert k["work"] == work
        raise Stop
    monkeypatch.setattr(cli, "check_pair", lambda *a: [])
    monkeypatch.setattr(recon, "reconstruct_colmap", fake)
    with pytest.raises(Stop):
        cli.main(["run", "a.mov", "b.mov", "--out", str(out), "--poses", "colmap", "--colmap-from", str(tmp_path / "pair")])
