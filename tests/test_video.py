"""The results video's offline logic: per-frame poses (interpolate only short calm gaps, fade where unknown),
the summary's number check, box geometry and clipping. No network, no Chrome, no COLMAP."""

import math

import numpy as np
import pytest

from whatchanged.video import (PoseTrack, allowed_numbers, box_corners, change_boxes, change_facts, check_numbers,
                               clip_segment, fade_weights, slerp_R, template_summary, title_card, end_card, view_score)


def yaw(deg):
    """World-to-camera rotation for a camera turned `deg` about the vertical (y) axis."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])


def angle(Ra, Rb):
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


def track(samples):
    return PoseTrack.build([(t, yaw(d), np.array(c, float), src) for t, d, c, src in samples])


# ---------------------------------------------------------------- poses

def test_slerp_halfway():
    R = slerp_R(yaw(0), yaw(8), 0.5)
    assert angle(R, yaw(4)) < 1e-4


def test_interpolates_short_calm_gap():
    tr = track([(1.0, 0, [0, 1.4, 0], "key"), (1.2, 6, [0.1, 1.4, 0], "key")])
    R, t, ok = tr.at([1.1])
    assert ok[0]
    assert angle(R[0], yaw(3)) < 1e-4
    C = -R[0].T @ t[0]
    assert np.allclose(C, [0.05, 1.4, 0], atol=1e-9)


def test_exact_time_uses_the_pose_itself():
    tr = track([(1.0, 0, [0, 1.4, 0], "key"), (5.0, 90, [3, 1.4, 0], "key")])
    R, t, ok = tr.at([1.0 + 0.2 / 30])   # within half a frame of a known pose
    assert ok[0] and angle(R[0], yaw(0)) < 1e-4


def test_no_pose_across_long_gap_fast_turn_big_step_or_outside():
    tr = track([(0.0, 0, [0, 1.4, 0], "key"), (1.0, 0, [0, 1.4, 0], "key"),            # 1 s gap: too long
                (1.1, 30, [0, 1.4, 0], "key"),                                         # 30 degrees in 0.1 s: not calm
                (1.2, 30, [0.6, 1.4, 0], "key")])                                      # 60 cm in 0.1 s: a jump
    _, _, ok = tr.at([0.5, 1.05, 1.15, -0.5, 3.0])
    assert not ok.any()


def test_drop_jumps_keeps_keyframes_and_good_registrations():
    tr = track([(0.0, 0, [0, 1.4, 0], "key"), (0.1, 1, [0.01, 1.4, 0], "reg"), (0.2, 2, [3.0, 1.4, 0], "reg"),
                (0.3, 3, [0.03, 1.4, 0], "reg"), (0.4, 4, [0.04, 1.4, 0], "key")])
    assert tr.drop_jumps() == 1
    assert list(np.round(tr.t, 2)) == [0.0, 0.1, 0.3, 0.4]


def test_fade_ramps_into_unknown_and_drops_flashes():
    ok = np.array([True] * 30 + [False] * 10 + [True] * 30)
    a = fade_weights(ok, fps=30, fade_s=0.25, min_run_s=0.3)
    assert a[0] == 1 and a[15] == 1
    assert (a[30:40] == 0).all()
    assert a[29] < a[25] < 1          # fading out towards the unknown stretch
    assert a[40] < a[45]              # fading back in after it
    flash = np.array([False] * 10 + [True] * 4 + [False] * 10)   # 4 frames (0.13 s) of pose: not worth a flash
    assert not fade_weights(flash, fps=30).any()
    assert (fade_weights(np.ones(12, bool)) == 1).all()


# ---------------------------------------------------------------- geometry

def test_box_corners_follow_the_viewer_yaw():
    b = {"centre": [1.0, 0.5, 2.0], "size": [2.0, 1.0, 0.4], "yaw_deg": 90}
    P = box_corners(b)
    assert np.allclose(P.mean(0), b["centre"])
    # three.js rotation.y = +90 deg takes local +x to world -z, so the 2 m side now runs along z
    assert np.isclose(np.ptp(P[:, 2]), 2.0) and np.isclose(np.ptp(P[:, 0]), 0.4) and np.isclose(np.ptp(P[:, 1]), 1.0)
    assert np.allclose(P[1] - P[0], [0, 0, -2.0], atol=1e-9)


def test_clip_segment_behind_the_camera():
    assert clip_segment(np.array([0, 0, -1.0]), np.array([0, 0, -2.0])) is None
    a, b = clip_segment(np.array([0, 0, -1.0]), np.array([0, 0, 1.0]), near=0.05)
    assert np.isclose(a[2], 0.05) and np.isclose(b[2], 1.0)


def test_change_boxes_solid_where_the_object_is():
    rem = {"type": "removed", "centre": [0, 0, 0], "size": [1, 1, 1]}
    add = {"type": "added", "centre": [0, 0, 0], "size": [1, 1, 1]}
    mov = {"type": "moved", "before": {"centre": [0, 0, 0], "size": [1, 1, 1]}, "after": {"centre": [2, 0, 0], "size": [1, 1, 1]}}
    assert [s for _, s in change_boxes(rem, after=False)] == [True] and [s for _, s in change_boxes(rem, after=True)] == [False]
    assert [s for _, s in change_boxes(add, after=False)] == [False] and [s for _, s in change_boxes(add, after=True)] == [True]
    (now, s1), (then, s2) = change_boxes(mov, after=True)
    assert now["centre"] == [2, 0, 0] and s1 and then["centre"] == [0, 0, 0] and not s2


def test_view_score_prefers_a_box_in_front():
    K = np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1.0]])
    b = {"centre": [0, 0, 2.0], "size": [0.5, 0.5, 0.5]}
    R = np.eye(3)
    ahead = view_score([b], R, np.zeros(3), K, (640, 360))
    behind = view_score([b], R, np.array([0, 0, -4.0]), K, (640, 360))
    assert ahead > 0.3 and behind == 0


# ---------------------------------------------------------------- facts and the summary check

SCENE = {
    "splats": {"A": "A.spz", "B": "B.spz"},
    "meta": {"videos": {"A": "before.mov", "B": "after.mov"}},
    "objects": {"A": [{"label": "bed", "centre": [0, 0, 0], "size": [1, 1, 1]}, {"label": "wall", "verified": False, "centre": [0, 0, 0], "size": [1, 1, 1]}],
                "B": [{"label": "bed", "centre": [0, 0, 0], "size": [1, 1, 1]}]},
    "changes": [
        {"id": "c01", "type": "removed", "label": "office chair", "sentence": "The office chair on the floor is gone; video B saw that spot empty from 6 viewpoints.",
         "centre": [0, 0.5, 0], "size": [0.6, 1.0, 0.6]},
        {"id": "c02", "type": "moved", "label": "wooden chair", "sentence": "The wooden chair moved about 1.4 m.",
         "before": {"centre": [0, 0.4, 0], "size": [0.5, 0.8, 0.5]}, "after": {"centre": [1.4, 0.4, 0.1], "size": [0.5, 0.8, 0.5]},
         "centre": [1.4, 0.4, 0.1], "size": [0.5, 0.8, 0.5]},
    ],
}


def test_facts_and_template():
    f = change_facts(SCENE, None)
    assert f["change_count"] == 2 and f["by_type"] == {"removed": 1, "moved": 1}
    assert f["objects_found"] == {"before": 1, "after": 1}          # the rejected box is not counted
    assert f["changes"][1]["moved_m"] == 1.4
    t = template_summary(f)
    assert t.startswith("2 changes:") and "wooden chair moved 1.4 m" in t
    assert check_numbers(t, f) == []
    none = change_facts({**SCENE, "changes": []}, None)
    assert check_numbers(template_summary(none), none) == []


def test_number_check_rejects_invented_numbers():
    f = change_facts(SCENE, None)
    assert 6.0 in allowed_numbers(f) and 1.4 in allowed_numbers(f)
    assert check_numbers("Two chairs changed: the office chair is gone and the wooden chair moved 1.4 m.", f) == []
    assert check_numbers("The wooden chair moved 1.5 m.", f)              # not a measured number
    assert check_numbers("Three things changed in the room.", f)          # a number word that is not a fact
    assert check_numbers("The wooden chair moved 140 cm.", f)             # converted units are not allowed
    assert check_numbers("Both videos agree: the office chair left.", f) == []
    assert check_numbers("Twenty things changed.", f)                     # number words above twelve count too
    assert check_numbers("Half the room changed.", f)
    g = {**f, "changes": [*f["changes"], {"id": "c03", "type": "added", "object": "double bed", "sentence": ""}]}
    assert check_numbers("A double bed appeared.", g) == []               # a word inside a fact's name is a fact


def test_cards_render():
    f = change_facts(SCENE, None)
    img = title_card(f, {"text": template_summary(f), "source": "template"})
    assert img.size == (1920, 1080)
    end = end_card(SCENE, f, {"shared_model": True, "names_by_claude": False})
    assert end.size == (1920, 1080)
    assert np.asarray(img).std() > 5 and np.asarray(end).std() > 5


@pytest.mark.parametrize("n", [0, 1])
def test_no_change_title(n):
    scene = {**SCENE, "changes": SCENE["changes"][:n]}
    f = change_facts(scene, None)
    assert f["change_count"] == n
    assert title_card(f, {"text": template_summary(f), "source": "template"}).size == (1920, 1080)
