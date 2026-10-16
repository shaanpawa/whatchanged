"""One object set for both videos, on the ray-cast synthetic room: no detector model, no network.

  - both videos' views fuse into one box per object, which both states show;
  - an object under a detected change follows it (removed: Before only, added: After only, moved: both
    places), and every other object stays in both states, however poorly one video saw it;
  - a box whose depth points cover only part of its 2D detections grows to them, and no further;
  - a box whose points are not on the object (a see-through bin) is moved to where its detections are.
"""

import json

import numpy as np
import pytest
from test_traps import A_EYES, B_EYES, ROOM, session

from whatchanged import cli, objects3d
from whatchanged.changes import Blob
from whatchanged.geometry import project
from whatchanged.objects3d import Box3D, Det2D

CAB = ((1.0, 1.0, 0.0), (1.4, 1.3, 1.0), (30, 90, 200))      # a 40 x 30 x 100 cm cabinet on the floor
W, H = 160, 120


def truth(box) -> Box3D:
    lo, hi = np.array(box[0], float), np.array(box[1], float)
    return Box3D("cabinet", 0.9, (lo + hi) / 2, hi - lo, 0.0)


def dets_of(S, box, label="cabinet", whole=False) -> list[Det2D]:
    """What a perfect 2D detector would report: the projected box's outline, in every frame that sees it
    (with whole=True only where most of it is in the picture)."""
    out = []
    for j in range(len(S.depth)):
        uv, z = project(truth(box).corners(), S.poses[j], S.K[j])
        if (z <= 0.1).any():
            continue
        r = np.clip([*uv.min(0), *uv.max(0)], 0, [W - 1, H - 1, W - 1, H - 1])
        if whole and (r[2] - r[0]) * (r[3] - r[1]) < 0.6 * np.prod(uv.max(0) - uv.min(0)):
            continue
        if (r[2] - r[0]) * (r[3] - r[1]) > 20:
            out.append(Det2D(j, r, label, 0.8))
    return out


def blob(lo, hi, kind="removed", step=0.04) -> Blob:
    g = np.stack(np.meshgrid(*[np.arange(a + step / 2, b, step) for a, b in zip(lo, hi)], indexing="ij"), -1).reshape(-1, 3)
    return Blob(kind, g, np.zeros((len(g), 3), np.uint8), 0.9, 3.0, 5)


def obj(label, lo, hi, seen=("A", "B")) -> Box3D:
    lo, hi = np.array(lo, float), np.array(hi, float)
    return Box3D(label, 0.8, (lo + hi) / 2, hi - lo, 0.0, [0], {label: 0.8}, verified=True, label_source="claude",
                 seen_by={n: [0, 1] for n in seen})


@pytest.fixture(scope="module")
def rooms():
    return session("A", ROOM + [CAB], A_EYES), session("B", ROOM + [CAB], B_EYES)


# ---------------------------------------------------------------- states follow the changes

def test_no_change_means_the_same_objects_in_both_states():
    shared = [obj("bed", (3.0, 2.0, 0.0), (3.9, 2.9, 0.5)),
              obj("lamp", (0.2, 0.2, 0.0), (0.4, 0.4, 1.5), seen=("A",))]       # B never saw it well
    objects3d.assign_states(shared, {"removed": [], "added": [], "moves": []}, log=lambda *_: None)
    a, b = objects3d.for_state(shared, "A"), objects3d.for_state(shared, "B")
    assert [o.label for o in a] == [o.label for o in b] == ["bed", "lamp"]
    assert all(o.state == "both" for o in shared)


def test_removed_object_is_before_only_and_a_mug_does_not_take_the_desk():
    chair = obj("wooden chair", (1.0, 1.0, 0.0), (1.4, 1.3, 0.9), seen=("A",))
    desk = obj("desk", (2.0, 0.0, 0.0), (3.2, 0.6, 0.75))
    lamp = obj("lamp", (0.2, 0.2, 0.0), (0.4, 0.4, 1.5), seen=("A",))         # seen by one video, no change: static
    res = {"removed": [blob((1.02, 1.02, 0.1), (1.38, 1.28, 0.85)),             # the chair, partly
                       blob((2.5, 0.2, 0.75), (2.6, 0.3, 0.85))],               # a mug off the desk
           "added": [], "moves": []}
    shared = [chair, desk, lamp]
    objects3d.assign_states(shared, res, log=lambda *_: None)
    assert (chair.state, chair.change) == ("A", "c01")
    assert desk.state == lamp.state == "both"
    assert [o.label for o in objects3d.for_state(shared, "A")] == ["wooden chair", "desk", "lamp"]
    assert [o.label for o in objects3d.for_state(shared, "B")] == ["desk", "lamp"]


def test_a_static_object_behind_a_removed_one_stays():
    chair = obj("wooden chair", (1.0, 1.0, 0.0), (1.4, 1.3, 0.9), seen=("A",))
    radiator = obj("radiator", (0.9, 1.2, 0.1), (1.6, 1.6, 0.5))             # too deep a box, reaching the chair
    pillow = obj("pillow", (1.05, 1.05, 0.45), (1.35, 1.25, 0.6), seen=("A",))  # on the chair, gone with it
    res = {"removed": [blob((1.02, 1.02, 0.05), (1.38, 1.28, 0.85))], "added": [], "moves": []}
    objects3d.assign_states([chair, radiator, pillow], res, log=lambda *_: None)
    assert chair.state == pillow.state == "A" and radiator.state == "both"


def test_added_object_is_after_only():
    case = obj("suitcase", (1.7, 1.0, 0.0), (2.2, 1.4, 0.7), seen=("B",))
    res = {"removed": [], "added": [blob((1.72, 1.02, 0.05), (2.18, 1.38, 0.68), "added")], "moves": []}
    objects3d.assign_states([case], res, log=lambda *_: None)
    assert case.state == "B" and objects3d.for_state([case], "A") == []


def test_moved_object_is_one_object_at_two_places():
    old = obj("wooden chair", (1.0, 1.0, 0.0), (1.4, 1.3, 0.9), seen=("A",))
    new = obj("wooden chair", (2.5, 1.0, 0.0), (2.8, 1.4, 0.9), seen=("B",))    # turned 90 degrees
    r, a = blob((1.02, 1.02, 0.1), (1.38, 1.28, 0.85)), blob((2.52, 1.02, 0.1), (2.78, 1.38, 0.85), "added")
    fit = {"fitness": 0.8, "turn_deg": 90.0, "turn_determined": True}
    shared = [old, new]
    objects3d.assign_states(shared, {"removed": [r], "added": [a], "moves": [(0, 0, 0.2, fit)]}, log=lambda *_: None)
    assert old.state == "moved" and new.state == "merged" and old.change == "c01"
    before, after = objects3d.for_state(shared, "A"), objects3d.for_state(shared, "B")
    assert len(before) == len(after) == 1
    assert np.allclose(before[0].centre, old.centre) and np.allclose(after[0].centre, new.centre)
    # seen at one end only: carried along by the change (shift and turn)
    lone = obj("wooden chair", (1.0, 1.0, 0.0), (1.4, 1.3, 0.9), seen=("A",))
    objects3d.assign_states([lone], {"removed": [r], "added": [a], "moves": [(0, 0, 0.2, fit)]}, log=lambda *_: None)
    moved = objects3d.for_state([lone], "B")[0]
    assert np.allclose(moved.centre[:2], a.centre[:2] + (lone.centre[:2] - r.centre[:2]) @ np.array([[0, 1], [-1, 0]]), atol=0.02)
    assert abs(np.degrees(moved.yaw)) == pytest.approx(90, abs=1)


def test_viewer_boxes_come_from_the_shared_set(tmp_path):
    pytest.importorskip("cv2")
    from whatchanged.room3d import detection_boxes
    chair = obj("wooden chair", (1.0, 1.0, 0.0), (1.4, 1.3, 0.9), seen=("A",))
    bed = obj("bed", (3.0, 2.0, 0.0), (3.9, 2.9, 0.5))
    objects3d.assign_states([chair, bed], {"removed": [blob((1.02, 1.02, 0.1), (1.38, 1.28, 0.85))], "added": [],
                                           "moves": []}, log=lambda *_: None)
    chair.id, bed.id = "o00", "o01"
    objects3d.write_shared(tmp_path / "objects.json", [chair, bed])
    doc = json.loads((tmp_path / "objects.json").read_text())
    assert [o["label"] for o in doc["sessions"]["B"]] == ["bed"] and len(doc["shared"]) == 2
    out = detection_boxes(tmp_path, ["A", "B"], {"matrix": np.eye(4), "yaw_deg": 0.0})["objects"]
    assert [o["id"] for o in out["A"]] == ["o00", "o01"] and [o["id"] for o in out["B"]] == ["o01"]
    assert out["A"][1]["centre"] == out["B"][0]["centre"]              # the static bed is the same box in both


# ---------------------------------------------------------------- fuller boxes

def _partial(S, box, zmax):
    """The depth points of a box, but only its lower part (plain or white tops give no depth)."""
    p, _ = S.points(conf_pct=0, stride=1)
    lo, hi = np.array(box[0]) - 0.01, np.array(box[1]) + 0.01
    p = p[((p >= lo) & (p <= hi)).all(1) & (p[:, 2] < zmax)]
    return p


def test_box_grows_to_its_2d_detections_and_no_further(rooms):
    A, B = rooms
    C, spans = objects3d.combine({"A": A, "B": B})
    pts = _partial(C, CAB, zmax=0.45)
    c, s, yaw = objects3d.fit_box_at(pts, 0.0)
    b = Box3D("cabinet", 0.9, c, s, yaw, [0], {"cabinet": 0.9}, pts, dets=dets_of(C, CAB))
    assert b.size[2] < 0.5
    assert objects3d.grow_box(C, b, list(range(len(C.depth))), supports=[0.0])
    t = truth(CAB)
    assert b.size[2] == pytest.approx(1.0, abs=0.12)                               # the top found again
    assert np.all(b.size[:2] <= np.sort(t.size[:2])[::-1] + 0.12)                   # no ballooning
    assert np.linalg.norm(b.centre[:2] - t.centre[:2]) < 0.08
    assert b.grown["iou"][1] > b.grown["iou"][0]


def test_box_that_covers_its_detections_is_left_alone(rooms):
    A, _ = rooms
    pts = _partial(A, CAB, zmax=2.0)
    c, s, yaw = objects3d.fit_box_at(pts, 0.0)
    b = Box3D("cabinet", 0.9, c, s, yaw, [0], {"cabinet": 0.9}, pts, dets=dets_of(A, CAB))
    size = b.size.copy()
    assert not objects3d.grow_box(A, b, list(range(len(A.depth))), supports=[0.0])
    assert np.allclose(b.size, size)


def test_too_big_box_comes_in_to_its_detections(rooms):
    A, B = rooms
    C, _ = objects3d.combine({"A": A, "B": B})
    t = truth(CAB)
    pts = _partial(C, CAB, zmax=2.0)
    floor = np.random.default_rng(0).uniform([1.0, 1.3, 0.0], [1.4, 1.7, 0.02], (200, 3))   # bled-in floor in front
    b = Box3D("cabinet", 0.9, t.centre + [0, 0.2, 0], t.size + [0, 0.4, 0], 0.0, [0], {"cabinet": 0.9},
              np.concatenate([pts, floor]), dets=dets_of(C, CAB))
    assert objects3d.shrink_box(C, b)
    assert b.size[1] == pytest.approx(0.3, abs=0.08) and np.linalg.norm(b.centre - t.centre) < 0.06


def test_box_off_its_detections_is_moved_onto_them(rooms):
    A, B = rooms
    C, _ = objects3d.combine({"A": A, "B": B})
    t = truth(CAB)
    pts = _partial(C, CAB, zmax=2.0) + [0.4, 0.3, 0.0]        # depth seen through the object: the wall behind
    b = Box3D("bin", 0.9, t.centre + [0.4, 0.3, 0.0], t.size.copy(), 0.0, [0], {"bin": 0.9}, pts, dets=dets_of(C, CAB))
    assert objects3d.coverage(C, b, b.dets)[1] < 0.5
    assert objects3d.grow_box(C, b, list(range(len(C.depth))), supports=[0.0])
    assert b.grown["mode"] == "placed" and np.linalg.norm(b.centre - t.centre) < 0.1


# ---------------------------------------------------------------- the whole shared pass

def test_both_videos_fuse_into_one_object_set(rooms, monkeypatch):
    A, B = rooms

    def detect(S, frames, vocab, kind, device, log=print):   # (a lone box cut off by the image edge lifts the wall)
        return [d for d in dets_of(S, CAB, whole=True) if d.frame in set(frames)], {"dets_2d": 0}
    monkeypatch.setattr(objects3d, "detect", detect)
    boxes, stats, ctx = objects3d.boxes_shared({"A": A, "B": B}, n_frames=40, log=lambda *_: None)
    cabs = [b for b in boxes if b.label == "cabinet"]
    assert len(cabs) == 1 and set(cabs[0].seen_by) == {"A", "B"}
    assert stats["frames_per_session"] == {"A": len(A_EYES), "B": len(B_EYES)}
    assert np.linalg.norm(cabs[0].centre[:2] - truth(CAB).centre[:2]) < 0.08
    objects3d.assign_states(boxes, {"removed": [], "added": [], "moves": []}, log=lambda *_: None)
    assert [b.label for b in objects3d.for_state(boxes, "A")] == [b.label for b in objects3d.for_state(boxes, "B")]


def test_colmap_runs_box_once_for_both_videos(tmp_path, monkeypatch, rooms):
    A, B = rooms
    seen = []

    def boxes_shared(sessions, n_frames, device, log, **kw):
        seen.append((list(sessions), n_frames, device))
        C, spans = objects3d.combine(sessions)
        b = Box3D("cabinet", 0.9, truth(CAB).centre, truth(CAB).size, 0.0, [0, len(A.depth)], {"cabinet": 0.9},
                  seen_by={"A": [0], "B": [0]})
        return [b], {"boxes": 1}, {"S": C, "spans": spans, "supports": np.array([0.0]), "room_yaw": 0.0}
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(objects3d, "pick_device", lambda: "cpu")
    monkeypatch.setattr(objects3d, "boxes_shared", boxes_shared)
    monkeypatch.setattr(cli, "log", lambda *_: None)
    shared, stats = cli._shared_objects(tmp_path, A, B, True, "auto")
    assert seen == [(["A", "B"], cli.OBJECT_FRAMES_SHARED, "cpu")]
    assert [b.id for b in shared] == ["o00"] and stats["naming"]["label_source"] == "owlv2"
    doc = json.loads((tmp_path / "objects.json").read_text())
    assert doc["shared"][0]["seen_by"] == {"A": [0], "B": [0]}
    assert doc["sessions"]["A"] == doc["sessions"]["B"]
    again, _ = cli._shared_objects(tmp_path, A, B, True, "auto")             # cached: no second detection pass
    assert len(seen) == 1 and again[0].label == "cabinet"


def test_too_small_boxes_grow_and_a_worse_second_look_puts_them_back(rooms, monkeypatch):
    A, B = rooms
    C, spans = objects3d.combine({"A": A, "B": B})
    ctx = {"S": C, "spans": spans, "supports": np.array([0.0]), "room_yaw": 0.0}
    monkeypatch.setattr(cli, "log", lambda *_: None)

    class Namer:                                   # Claude's second look, as a fixed verdict
        def __init__(self, fit):
            self.fit, self.asked = fit, 0

        def check(self, S, key, boxes, views=None, tag=""):
            for x in boxes:
                x.fit, self.asked = self.fit, self.asked + 1

    def small():
        pts = _partial(C, CAB, zmax=0.45)
        c, s, yaw = objects3d.fit_box_at(pts, 0.0)
        return Box3D("cabinet", 0.9, c, s, yaw, [0, len(A.depth)], {"cabinet": 0.9}, pts, dets=dets_of(C, CAB),
                     verified=True, label_source="claude", fit="too_small")
    ok = Namer("good")
    out = cli._regrow(ok, ctx, [small()], None)
    assert ok.asked == 1 and out[0].fit == "good" and out[0].size[2] == pytest.approx(1.0, abs=0.12)
    b = small()
    height = b.size[2]
    out = cli._regrow(Namer("too_big"), ctx, [b], None)
    assert out[0].fit == "too_small" and out[0].size[2] == pytest.approx(height) and not out[0].grown


def _verdict(verified, fit):
    class Namer:                                   # Claude's second look, as a fixed verdict
        asked = 0

        def check(self, S, key, boxes, views=None, tag=""):
            for x in boxes:
                x.verified, x.fit = verified, fit
                Namer.asked += 1
    return Namer()


def test_a_box_dropped_as_too_big_comes_in_and_is_asked_again(rooms, monkeypatch):
    A, B = rooms
    C, spans = objects3d.combine({"A": A, "B": B})
    ctx = {"S": C, "spans": spans, "supports": np.array([0.0]), "room_yaw": 0.0}
    monkeypatch.setattr(cli, "log", lambda *_: None)
    t = truth(CAB)
    floor = np.random.default_rng(0).uniform([1.0, 1.3, 0.0], [1.4, 1.7, 0.02], (200, 3))

    def big(label="cabinet"):                      # "spans from the door to the desk": Claude dropped it as too big
        return Box3D(label, 0.9, t.centre + [0, 0.2, 0], t.size + [0, 0.4, 0], 0.0, [0, len(A.depth)], {"cabinet": 0.9},
                     np.concatenate([_partial(C, CAB, zmax=2.0), floor]), dets=dets_of(C, CAB), verified=False,
                     label_source="claude", fit="too_big")
    keep = _verdict(True, "too_small")
    out = cli._regrow(keep, ctx, [big()], None)
    assert keep.asked == 1 and out[0].verified is True and out[0].fit == "too_small"
    assert out[0].size[1] == pytest.approx(0.3, abs=0.08) and out[0].grown["mode"] == "shrunk"
    b = big()
    size = b.size.copy()
    out = cli._regrow(_verdict(False, "too_big"), ctx, [b], None)          # still no: the old box and verdict stay
    assert out[0].verified is False and out[0].fit == "too_big" and np.allclose(out[0].size, size) and not out[0].grown
    wall = _verdict(True, "good")
    out = cli._regrow(wall, ctx, [big("wall")], None)                      # not a thing: never asked again
    assert wall.asked == 0 and out[0].verified is False
    there = Box3D("cabinet", 0.9, t.centre.copy(), t.size.copy(), 0.0, [0], {"cabinet": 0.9}, _partial(C, CAB, zmax=2.0),
                  verified=True, label_source="claude", fit="good")      # already boxed right there: it stays dropped
    b = big()
    out = cli._regrow(_verdict(True, "good"), ctx, [there, b], None)
    assert b.verified is False and np.allclose(b.size, size) and len(out) == 2
    assert np.allclose(there.size, t.size) and there.verified is True


GHOST = ((2.4, 1.7, 0.0), (2.8, 2.0, 1.0))           # where a few close-up frames put a second cabinet: empty floor


def _kept(box, label="cabinet", seen=("A", "B"), frames=(0,)):
    lo, hi = np.array(box[0], float), np.array(box[1], float)
    return Box3D(label, 0.9, (lo + hi) / 2, hi - lo, 0.0, list(frames), {label: 0.9}, verified=True,
                 label_source="claude", seen_by={n: [0] for n in seen})


def test_a_second_copy_lifted_to_the_wrong_place_is_dropped(rooms):
    A, B = rooms
    C, spans = objects3d.combine({"A": A, "B": B})
    ghost_dets = [Det2D(d.frame, d.box, "cabinet", 0.8) for d in dets_of(C, (*GHOST, None))[:2]]   # its two close-ups
    real, ghost = _kept(CAB), _kept(GHOST)
    n = objects3d.drop_twins(C, spans, [real, ghost], dets_of(C, CAB) + ghost_dets, log=lambda *_: None)
    assert n == 1 and ghost.verified is False and real.verified is True
    assert "second copy of the cabinet" in ghost.note
    # the same pair when the second one is only in the other video (added there, or moved): no shared video, both stay
    real, ghost = _kept(CAB, seen=("A",), frames=(0,)), _kept(GHOST, seen=("B",), frames=(len(A.depth),))
    assert objects3d.drop_twins(C, spans, [real, ghost], dets_of(C, CAB) + ghost_dets, log=lambda *_: None) == 0
    assert ghost.verified is True


def test_two_real_objects_of_one_name_both_stay():
    second = ((2.4, 1.7, 0.0), (2.8, 2.0, 1.0), (30, 90, 200))
    A, B = session("A", ROOM + [CAB, second], A_EYES), session("B", ROOM + [CAB, second], B_EYES)
    C, spans = objects3d.combine({"A": A, "B": B})
    one, two = _kept(CAB), _kept(second)
    dets = dets_of(C, CAB) + dets_of(C, second)
    assert objects3d.drop_twins(C, spans, [one, two], dets, log=lambda *_: None) == 0
    assert one.verified is True and two.verified is True


# ---------------------------------------------------------------- changes the object detector finds

MOVED = ((2.4, 0.4, 0.0), (2.8, 0.7, 1.0), (30, 90, 200))      # the cabinet, after it was carried across the room


def _in(C, spans, name, dets):
    return [d for d in dets if spans[name][0] <= d.frame < spans[name][1]]


def test_object_seen_steadily_in_one_video_and_absent_where_the_other_looked_is_a_move():
    A, B = session("A", ROOM + [CAB], A_EYES), session("B", ROOM + [MOVED], B_EYES)
    C, spans = objects3d.combine({"A": A, "B": B})
    dets = _in(C, spans, "A", dets_of(C, CAB)) + _in(C, spans, "B", dets_of(C, MOVED))
    before, after = _kept(CAB, seen=("A",)), _kept(MOVED, seen=("B",))
    found = objects3d.object_changes(C, spans, [before, after], dets, list(range(len(C.depth))), log=lambda *_: None)
    assert {(c[0] is before, c[1]) for c in found} == {(True, "removed"), (False, "added")}
    assert all(c[2] >= 3 and c[3] == 0 for c in found)          # the other video showed the spot and found nothing
    from whatchanged import changes as ch
    res = {"removed": [], "added": [], "moves": []}
    assert ch.add_object_changes(res, found, A, B, ch.Config(), log=lambda *_: None) == 2
    (i, j, _, fit), = res["moves"]
    assert res["removed"][i].source == res["added"][j].source == "objects" and fit["turn_deg"] is None
    assert np.allclose(res["removed"][i].centre[:2], before.centre[:2], atol=0.05)


def test_object_both_videos_detect_is_no_change_and_neither_is_one_the_other_never_looked_at():
    A, B = session("A", ROOM + [CAB], A_EYES), session("B", ROOM + [CAB], B_EYES)
    C, spans = objects3d.combine({"A": A, "B": B})
    box = _kept(CAB)
    assert objects3d.object_changes(C, spans, [box], dets_of(C, CAB), list(range(len(C.depth))), log=lambda *_: None) == []
    # only A's frames went through the detector: B never looked, so nothing can be said about B
    only_a = list(range(*spans["A"]))
    assert objects3d.object_changes(C, spans, [box], _in(C, spans, "A", dets_of(C, CAB)), only_a, log=lambda *_: None) == []


def test_detector_found_removal_says_how_it_was_found():
    from whatchanged.semantics import check_sentence, sentence
    c = {"id": "c01", "type": "removed", "label": "speaker", "colour": "red", "size_m": [0.1, 0.1, 0.2], "bottom_m": 0.75,
         "on_before": "desk", "evidence": {"free_views": 4, "free_share": 1.0, "noise_margin": 0.0, "by": "objects"}}
    c["sentence"] = sentence(c)
    assert c["sentence"] == ("The red speaker on the desk is gone; video B looked at that spot from 4 viewpoints "
                             "and did not find it.")
    assert check_sentence(c) == []


def test_claude_keeps_only_the_detector_changes_it_sees_are_gone(tmp_path, monkeypatch):
    from test_naming import Fake

    from whatchanged import naming
    A, B = session("A", ROOM + [CAB], A_EYES), session("B", ROOM + [MOVED], B_EYES)
    C, spans = objects3d.combine({"A": A, "B": B})
    dets = _in(C, spans, "A", dets_of(C, CAB)) + _in(C, spans, "B", dets_of(C, MOVED))
    before, after = _kept(CAB, seen=("A",)), _kept(MOVED, seen=("B",))
    found = objects3d.object_changes(C, spans, [before, after], dets, list(range(len(C.depth))), log=lambda *_: None)
    # Claude sees the cabinet gone from its old place, but (say) still sees something at the new one in video A
    fake = Fake(lambda text: {"object": True, "there": "absent" if "video A of a room" in text else "present",
                              "name": "blue cabinet"})
    monkeypatch.setattr(naming, "available", lambda: True)
    kept = naming.confirm_changes(tmp_path, C, spans, found, log=lambda *_: None, client=fake)
    assert [c[1] for c in kept] == ["removed"] and len(fake.calls) == 2
    assert all(len(call["messages"][0]["content"]) >= 3 for call in fake.calls)   # its own view(s) and the other's
    # answers are cached: asking again costs nothing
    again = Fake(lambda text: {"object": False, "there": "present", "name": ""})
    assert [c[1] for c in naming.confirm_changes(tmp_path, C, spans, found, log=lambda *_: None, client=again)] == ["removed"]
    assert again.calls == []
    # another machine decodes the frames a shade differently: the answers are keyed by what was asked, not pixels
    from dataclasses import replace
    shifted = replace(C, images=np.clip(C.images.astype(int) + 1, 0, 255).astype(np.uint8))
    assert [c[1] for c in naming.confirm_changes(tmp_path, shifted, spans, found, log=lambda *_: None, client=again)] == ["removed"]
    assert again.calls == []
    # without a key the cached verdicts still hold (the CI replay); with nothing cached, every change is kept
    monkeypatch.setattr(naming, "available", lambda: False)
    assert [c[1] for c in naming.confirm_changes(tmp_path, C, spans, found, log=lambda *_: None)] == ["removed"]
    assert naming.confirm_changes(tmp_path / "fresh", C, spans, found, log=lambda *_: None) is None


def test_an_object_mostly_seen_after_is_shown_from_the_after_video():
    spans = {"A": (0, 10), "B": (10, 30)}
    chair = _kept(CAB, frames=(3, 12, 14, 16, 18, 20, 22))          # one stray sighting before, six after
    assert set(objects3d.free_frames(spans, chair, cap=10 ** 6)) == set(range(30))
    assert set(objects3d.free_frames(spans, chair, cap=10 ** 6, min_share=0.25)) == set(range(10, 30))
    both = _kept(CAB, frames=(3, 5, 12, 14))
    assert set(objects3d.free_frames(spans, both, cap=10 ** 6, min_share=0.25)) == set(range(30))
