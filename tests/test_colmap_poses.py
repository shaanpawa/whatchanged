"""COLMAP poses for the detection run: intrinsics on the network's crop, keyframe picking, and the
two ways the room viewer ties a run's boxes to its splats (shared model, or trimmed clips)."""

import json

import numpy as np
import pytest

from whatchanged.recon import Session, correct_depth, fit_depth, local_correction, model_K, pick_posed_keyframes


def test_model_K_follows_mapanythings_resize_and_crop():
    cropping = pytest.importorskip("mapanything.utils.cropping")
    cam = {"width": 1600, "height": 900, "params": (1250.0, 1255.0, 803.0, 447.5)}
    Kcv = np.array([[1250.0, 0, 802.5], [0, 1255.0, 447.0], [0, 0, 1]])   # COLMAP's +0.5 pixel centres removed
    for wh in ((518, 294), (518, 392)):
        img = np.zeros((900, 1600, 3), np.uint8)
        small, _, K, _ = cropping.rescale_image_and_other_optional_info(img, wh, camera_intrinsics=Kcv)
        left, top = (small.size[0] - wh[0]) // 2, (small.size[1] - wh[1]) // 2   # load_images crops the centre
        K[0, 2] -= left
        K[1, 2] -= top
        mine = model_K(cam, wh)
        assert np.allclose(mine[:2, 2], K[:2, 2], atol=0.6)                # principal point within a pixel
        assert np.allclose(np.diag(mine)[:2], np.diag(K)[:2], rtol=3e-3)   # focal: ours uses the true resized size


def test_keyframes_are_posed_spread_and_sharp():
    frames = [{"name": f"A/A_{i:04d}.jpg", "t": i * 0.25, "sharpness": float(i % 3)} for i in range(40)]
    posed = {f["name"] for f in frames if f["t"] < 3.8 or f["t"] > 6}     # COLMAP lost 2 s in the middle
    keys = pick_posed_keyframes(frames, posed, 5)
    assert all(k["name"] in posed for k in keys)
    assert len(keys) == 4                                                # the slot inside the gap stays empty
    assert [k["sharpness"] for k in keys] == [2.0] * 4
    assert np.all(np.diff([k["t"] for k in keys]) > 1.0)


def test_depth_fit_undoes_a_stretch_and_holds_it_outside_the_points():
    rng = np.random.default_rng(0)
    z = rng.uniform(1.0, 3.0, 800)                        # COLMAP depth at the frame's points
    d = 1.8 * z - 1.1 + rng.normal(0, 0.01, 800)          # network depth: stretched, as on the trial
    d[:20] = 5.0                                          # a few bad matches
    f = fit_depth(d, z, fallback=1.0)
    assert f["kind"] == "affine" and f["res"] < 0.01
    assert np.allclose(correct_depth(np.array([0.9, 2.5]), f), [2.0 / 1.8, 2.0], atol=0.02)
    lo = f["lo"]                                          # below the points' range: ratio frozen, no line
    assert np.isclose(correct_depth(np.array([lo / 2]), f)[0] / (lo / 2), correct_depth(np.array([lo]), f)[0] / lo)
    assert fit_depth(d[:10], z[:10], fallback=0.4)["a"] == 0.4                   # too few points
    assert fit_depth(np.full(300, 2.0), np.full(300, 1.0), fallback=1)["kind"] == "scale"   # no depth range


def test_local_correction_fades_and_marks_unanchored_pixels():
    hw = (60, 80)
    u, v = np.array([10, 11, 10, 11]), np.array([10, 10, 11, 11])     # four points in one corner
    fit = {"kind": "scale", "a": 1.0, "b": 0.0}
    r, w = local_correction(np.ones(4), np.full(4, 1.1), u, v, fit, hw, sigma=3)
    assert abs(r[10, 10] - np.log(1.1)) < 0.03 and abs(r[50, 70]) < 1e-6    # corrected there, untouched far away
    assert w[10, 10] > 2 and w[50, 70] < 1e-6


def _yaw(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _save_session(run, name, poses, times):
    n = len(poses)
    z = np.zeros((n, 4, 4))
    S = Session(name, np.zeros((n, 4, 4, 3), np.uint8), z, z, z > 0, np.tile(np.eye(3), (n, 1, 1)), poses, times,
                [f"frames/{name}_{i:03d}.jpg" for i in range(n)])
    (run / "cache").mkdir(parents=True, exist_ok=True)
    S.save(run / "cache" / f"{name}.npz")


def test_clip_offsets_find_a_trimmed_copy(tmp_path):
    pytest.importorskip("cv2")
    from whatchanged.room3d import clip_offsets

    # Room video: frames every 0.25 s while the phone turns back and forth; COLMAP's world is tilted.
    heading = lambda t: 1.2 * np.sin(t / 3.0) + 0.4 * np.sin(t)          # noqa: E731
    tilt = _yaw(0.3) @ np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    imgs, frames = {}, {"room": []}
    for i, t in enumerate(np.arange(0, 40, 0.25)):
        R = tilt @ _yaw(heading(t))                                      # camera-to-world rotation
        q = _quat(R.T)                                                   # COLMAP stores world-to-camera
        imgs[i] = {"name": f"room/room_{i:04d}.jpg", "camera_id": 1, "q": q, "t": (0.0, 0.0, 0.0)}
        frames["room"].append({"name": imgs[i]["name"], "t": float(t)})
    # Session A: same clock. Session B: a copy trimmed 1.4 s in, so its clock reads 1.4 s less.
    run = tmp_path / "run"
    for name, trim in (("A", 0.0), ("B", 1.4)):
        times = np.linspace(1.0, 36.0, 24)
        poses = np.tile(np.eye(4), (24, 1, 1))
        poses[:, :3, :3] = [_yaw(0.9) @ _yaw(heading(t + trim)) for t in times]   # the run's own levelled frame
        _save_session(run, name, poses, times)
    off = clip_offsets(run, ["room"], frames, imgs, log=lambda *_: None)
    assert off["A"] == 0.0
    assert abs(off["B"] - 1.4) <= 0.15


def _quat(R):
    w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
    return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))


def test_shared_model_boxes_land_exactly(tmp_path):
    pytest.importorskip("cv2")
    from whatchanged.room3d import detection_boxes, shared_alignment

    work = tmp_path / "pair" / "work"
    (work / "dense" / "sparse" / "0").mkdir(parents=True)
    # The run: COLMAP world -> its levelled, metric, z-up world (scale 0.3, some turn, floor shift).
    s, R = 0.3, _yaw(0.7) @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    L = np.eye(4)
    L[:3, :3], L[:3, 3] = s * R, [0.2, -0.1, 0.45]
    run = tmp_path / "run"
    (run / "cache").mkdir(parents=True)
    (run / "cache" / "diag.json").write_text(json.dumps({"colmap": {"work": str(work), "to_levelled": L.tolist(),
                                                                    "metric_scale_m_per_unit": s}}))
    # The viewer: COLMAP world -> y-up (x, z, -y), same scale and floor, its own turn about y and centring.
    a = 0.4
    Y = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    Wm = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Y @ Wm @ (s * R), Y @ Wm @ L[:3, 3] + [1.0, 0.0, -2.0]
    p_colmap = np.array([1.5, -2.0, 4.0])
    p_run = L[:3, :3] @ p_colmap + L[:3, 3]
    align = shared_alignment(run, work / "dense", M, log=lambda *_: None)
    assert align["shared_model"]
    assert abs(align["yaw_deg"] - np.degrees(a)) < 1e-6 and abs(align["matrix"][1, 3]) < 1e-9   # a pure turn, floors equal
    (run / "objects.json").write_text(json.dumps({"sessions": {"A": [
        {"label": "bed", "centre_m": p_run.tolist(), "size_m": [2.0, 1.4, 0.5], "yaw_deg": 0.0}]}}))
    box = detection_boxes(run, ["A"], align)["objects"]["A"][0]
    assert np.allclose(box["centre"], M[:3, :3] @ p_colmap + M[:3, 3], atol=1e-3)
    assert shared_alignment(run, tmp_path / "elsewhere", M, log=lambda *_: None) is None
