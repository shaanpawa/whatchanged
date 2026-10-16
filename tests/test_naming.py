"""Claude's box naming with a fake client: no network, no key, no detector model.

The room vocabulary is cleaned and cached, every box is verified and renamed (the detector's label
kept as owl_label), dropped boxes are recorded rather than deleted, parts and duplicates merge,
refusals and a bad key leave the detector's labels, and without a key the pipeline says so once.
"""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_traps import A_EYES, BOX, ROOM, SUITCASE, session  # the ray-cast synthetic room

from whatchanged import cli, naming, objects3d, semantics
from whatchanged.objects3d import Box3D

BOX_C, BOX_S = np.array([1.2, 1.15, 0.2]), np.array([0.4, 0.3, 0.4])   # BOX from test_traps
FLOOR_C, FLOOR_S = np.array([2.6, 0.6, 0.15]), np.array([0.4, 0.4, 0.3])  # empty floor


class Fake:
    """Stands in for anthropic.Anthropic(): answers from the prompt text, records every request."""

    def __init__(self, answer, stop="end_turn"):
        self.answer, self.stop, self.calls = answer, stop, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        out = self.answer(kw["messages"][0]["content"][-1]["text"])
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(stop_reason=self.stop, usage=SimpleNamespace(input_tokens=2000, output_tokens=400),
                               content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=json.dumps(out))])


VERDICTS = {
    "heater": {"keep": True, "name": "Radiator", "part_of": "", "colour": "white", "fit": "good", "reason": "white panel radiator"},
    "refrigerator": {"keep": False, "name": "floor", "part_of": "", "colour": "", "fit": "too_big", "reason": "empty floor"},
}
ROOM_LIST = {"objects": [{"name": "radiator", "kind": "object"}, {"name": "tv bench", "kind": "surface"},
                         {"name": "rug", "kind": "flat"}]}


def answer(text):
    owl = re.search(r'labelled "([^"]+)"', text)
    return VERDICTS[owl.group(1)] if owl else ROOM_LIST


def boxes():
    return {"A": [Box3D("heater", 0.4, BOX_C.copy(), BOX_S.copy(), 0.0, [0, 3], {"heater": 0.4}, owl_label="heater"),
                  Box3D("refrigerator", 0.3, FLOOR_C.copy(), FLOOR_S.copy(), 0.0, [1], {"refrigerator": 0.3},
                        owl_label="refrigerator")]}


@pytest.fixture(scope="module")
def room():
    return session("A", ROOM + [BOX], A_EYES)


# ---------------------------------------------------------------- vocabulary

def test_vocabulary_is_cleaned_and_capped():
    raw = {"objects": [{"name": "Radiator", "kind": "object"}, {"name": "radiator", "kind": "object"},
                       {"name": "wall", "kind": "object"}, {"name": "TV bench!", "kind": "surface"},
                       {"name": "a very long name for a thing", "kind": "object"}, {"name": "rug", "kind": "flat"},
                       {"name": "lamp", "kind": "sparkly"}] + [{"name": f"thing {i}", "kind": "object"} for i in range(50)]}
    v = naming.parse_vocabulary(raw)
    assert [o["name"] for o in v[:4]] == ["radiator", "tv bench", "rug", "lamp"]
    assert [o["kind"] for o in v[:4]] == ["object", "surface", "flat", "object"]
    assert len(v) == 40 and "wall" not in [o["name"] for o in v]
    assert naming.parse_vocabulary(None) == []


def test_vocabulary_is_one_low_effort_call_then_cached(tmp_path, room):
    fake = Fake(answer)
    v = naming.Namer(tmp_path, log=lambda *_: None, client=fake).vocabulary([room, room])
    assert [o["name"] for o in v] == ["radiator", "tv bench", "rug"]
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5-5" and call["output_config"]["effort"] == "low"
    assert call["fallbacks"] == "default" and call["betas"] == ["server-side-fallback-2026-07-01"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert len(call["messages"][0]["content"]) == 2            # one contact sheet + the question
    again = naming.Namer(tmp_path, log=lambda *_: None, client=fake).vocabulary([room, room])
    assert again == v and len(fake.calls) == 1


def test_surface_nouns_keep_their_support_plane():
    pts = np.c_[np.zeros((50, 2)), np.r_[np.full(25, 0.01), np.linspace(0.2, 0.5, 25)]]
    assert len(objects3d._strip_supports(pts, "tv bench", np.array([]))) == 25          # unknown: floor-level points go
    support = objects3d.FLAT_OR_SUPPORT | {"tv bench"}
    assert len(objects3d._strip_supports(pts, "tv bench", np.array([]), support=support)) == 50


# ---------------------------------------------------------------- views

def test_views_see_the_whole_box_unoccluded_from_new_directions():
    S = session("A", ROOM + [BOX, SUITCASE], A_EYES)
    b = Box3D("box", 0.5, BOX_C, BOX_S, 0.0)
    shares = {j: naming.visible_share(S, j, b, naming.volume_samples(b)) for j in range(len(S.depth))}
    assert min(v for v in shares.values() if v is not None) < 0.5    # the suitcase hides it from some cameras
    views = naming.pick_views(S, b)
    assert 2 <= len(views) <= 3
    H, W = S.depth.shape[1:]
    for j, uv in views:
        assert shares[j] >= 0.5
        assert (uv >= 0).all() and (uv[:, 0] < W).all() and (uv[:, 1] < H).all()
    rays = [(BOX_C - S.poses[j][:3, 3]) / np.linalg.norm(BOX_C - S.poses[j][:3, 3]) for j, _ in views]
    assert np.degrees(np.arccos(np.clip(rays[0] @ rays[1], -1, 1))) >= 20
    crop = naming.view_crop(S, *views[0])
    assert max(crop.size) <= 768 and np.array(crop).reshape(-1, 3).tolist().count([255, 0, 255]) > 20


# ---------------------------------------------------------------- verify and rename

def test_boxes_are_renamed_and_dropped_ones_recorded(tmp_path, room):
    fake = Fake(answer)
    n = naming.Namer(tmp_path, log=lambda *_: None, client=fake)
    n.vocabulary([room])
    out = n.name({"A": room}, boxes())
    rad, floor = out["A"]
    assert (rad.label, rad.owl_label, rad.label_source, rad.verified) == ("radiator", "heater", "claude", True)
    assert (rad.colour, rad.fit, rad.model) == ("white", "good", "claude-opus-5-5")
    assert floor.verified is False and floor.owl_label == "refrigerator"
    box_calls = [c for c in fake.calls if c["output_config"]["effort"] == "medium"]
    assert len(box_calls) == 2
    assert "radiator, tv bench, rug" in box_calls[0]["messages"][0]["content"][-1]["text"]
    assert (tmp_path / rad.crop).exists()
    s = n.stats()
    assert s["label_source"] == "claude" and s["kept"] == 1 and s["dropped"] == 1 and s["calls"] == 3
    assert s["cost_usd_est"] == pytest.approx(3 * (2000 * 4 + 400 * 20) / 1e6)

    objects3d.write(tmp_path / "objects.json", out, {"naming": s})
    raw = json.loads((tmp_path / "objects.json").read_text())["sessions"]["A"]
    assert [(o["label"], o["owl_label"], o["verified"]) for o in raw] == [("radiator", "heater", True), ("floor", "refrigerator", False)]
    assert [b.label for b in objects3d.load(tmp_path / "objects.json")["A"]] == ["radiator"]
    assert len(objects3d.load(tmp_path / "objects.json", rejected=True)["A"]) == 2
    assert "floor" not in (tmp_path / "objects_A.txt").read_text()

    again = naming.Namer(tmp_path, log=lambda *_: None, client=fake)        # second run: all from the cache
    again.vocabulary([room])
    assert [b.label for b in again.name({"A": room}, boxes())["A"]] == ["radiator", "floor"]
    assert len(fake.calls) == 3 and again.stats()["cached"] == 3


def test_refusal_keeps_the_detector_label(tmp_path, room):
    n = naming.Namer(tmp_path, log=lambda *_: None, client=Fake(answer, stop="refusal"))
    b = n.name({"A": room}, boxes())["A"][0]
    assert (b.label, b.label_source, b.verified) == ("heater", "owlv2", None)
    assert n.stats()["refused"] == 2 and n.stats()["label_source"] == "owlv2"


def test_bad_key_stops_naming_and_keeps_detector_labels(tmp_path, room):
    anthropic = pytest.importorskip("anthropic")
    import httpx2
    bad = anthropic.AuthenticationError("invalid x-api-key", body=None, response=httpx2.Response(
        401, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")))
    fake, logs = Fake(lambda _: bad), []
    n = naming.Namer(tmp_path, log=logs.append, client=fake, workers=1)
    assert n.vocabulary([room]) == [] and n.error == "AuthenticationError 401"
    n2 = naming.Namer(tmp_path, log=logs.append, client=fake, workers=1)
    out = n2.name({"A": room}, {"A": boxes()["A"] + boxes()["A"] + boxes()["A"]})
    assert [b.label for b in out["A"]] == ["heater", "refrigerator"] * 3
    assert len(fake.calls) <= 3                       # the vocabulary, the first box, at most one already in flight
    assert sum("unavailable" in m for m in logs) == 2


# ---------------------------------------------------------------- merging

def named(label, centre, size, keep=True, part_of="", source="claude", fit="good"):
    return Box3D(label, 0.5, np.array(centre, float), np.array(size, float), 0.0, [0], {label: 0.5},
                 owl_label=label, label_source=source, verified=keep if source == "claude" else None,
                 part_of=part_of, fit=fit)


def test_parts_and_duplicates_merge_by_geometry():
    bs = [named("bed", (0, 0, 0.3), (2.0, 1.4, 0.6)),
          named("bed base", (0, -0.75, 0.15), (2.0, 0.08, 0.3), keep=False, part_of="bed"),    # side panel just outside
          named("pillow", (0.6, 0.3, 0.65), (0.5, 0.35, 0.12)),
          named("pillow", (0.75, 0.35, 0.66), (0.45, 0.35, 0.12)),
          named("wooden chair", (3, 0, 0.45), (0.5, 0.5, 0.9)),
          named("wooden chair", (4.5, 0, 0.45), (0.5, 0.5, 0.9)),                              # another chair
          named("sofa", (0.2, 0.2, 0.3), (1.0, 1.0, 0.6), source="owlv2"),                     # unchecked: left alone
          named("door", (-2, 2, 1.0), (0.4, 0.05, 2.0), keep=False, part_of="wardrobe"),
          named("panel", (-1.6, 2, 1.0), (0.4, 0.05, 2.0), keep=False, part_of="wardrobe"),
          named("handle", (5, 5, 1.0), (0.05, 0.05, 0.1), keep=False, part_of="cupboard"),
          named("seat", (4.95, 0, 0.45), (0.4, 0.4, 0.1), keep=False, part_of="chair"),          # by the 2nd chair
          named("headphones", (1, 3, 0.8), (0.12, 0.08, 0.15)),
          named("headphones", (1.05, 3, 0.7), (0.45, 0.35, 0.45), fit="too_big"),
          named("radiator", (0, 5, 0.3), (0.6, 0.12, 0.4)),
          named("radiator", (3, 5, 0.3), (0.3, 0.12, 0.4), keep=False, part_of="radiator")]       # another one
    out, folded = naming.merge(bs)
    assert folded == 4
    assert [(b.label, b.verified) for b in out] == [
        ("bed", True), ("pillow", True), ("wooden chair", True), ("wooden chair", True), ("sofa", None),
        ("wardrobe", True), ("cupboard", True), ("seat", False), ("headphones", True), ("radiator", True),
        ("radiator", True)]
    bed = out[0]
    assert sorted(bed.merged_from) == ["bed", "bed base"]
    assert bed.centre[1] - bed.size[1] / 2 == pytest.approx(-0.79, abs=0.02)     # the refit takes in the panel
    assert out[5].fit == out[6].fit == "too_small" and out[6].size[2] == pytest.approx(0.1)   # parts stand in for wholes
    assert out[8].size.max() < 0.2 and sorted(out[8].merged_from) == ["headphones", "headphones"]   # too-big box adds no size
    assert out[10].fit == "too_small" and out[10].centre[0] == 3       # a radiator part far from the other radiator


# ---------------------------------------------------------------- the pipeline

def _fake_detector(seen):
    def boxes_for_session(S, **kw):
        seen.append(kw)
        return boxes()["A"], {"boxes": 2}
    return boxes_for_session


def test_without_a_key_boxes_keep_detector_labels(tmp_path, monkeypatch, room):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not naming.available()
    seen, logs = [], []
    monkeypatch.setattr(objects3d, "pick_device", lambda: "cpu")
    monkeypatch.setattr(objects3d, "boxes_for_session", _fake_detector(seen))
    monkeypatch.setattr(cli, "log", logs.append)
    B = session("B", ROOM + [BOX], A_EYES)
    out = cli._object_boxes(tmp_path, room, B, True, "auto")
    assert [b.label for b in out["A"]] == ["heater", "refrigerator"]
    assert seen[0]["vocab"] == objects3d.VOCAB3D and seen[0]["roles"] == {}
    doc = json.loads((tmp_path / "objects.json").read_text())
    assert doc["stats"]["naming"] == {"label_source": "owlv2", "reason": "no ANTHROPIC_API_KEY"}
    assert doc["sessions"]["A"][0]["owl_label"] == "heater" and doc["sessions"]["A"][0]["label_source"] == "owlv2"
    assert sum("ANTHROPIC_API_KEY" in m for m in logs) == 1


def test_with_a_key_the_pipeline_names_boxes_once(tmp_path, monkeypatch, room):
    fake, seen = Fake(answer), []
    make = naming.Namer
    monkeypatch.setattr(naming, "available", lambda: True)
    monkeypatch.setattr(naming, "Namer", lambda out, log: make(out, log=log, client=fake))
    monkeypatch.setattr(objects3d, "pick_device", lambda: "cpu")
    monkeypatch.setattr(objects3d, "boxes_for_session", _fake_detector(seen))
    monkeypatch.setattr(cli, "log", lambda *_: None)
    B = session("B", ROOM + [BOX], A_EYES)
    out = cli._object_boxes(tmp_path, room, B, True, "auto")
    assert "radiator" in seen[0]["vocab"] and "heater" in seen[0]["vocab"]     # Claude's nouns join, none lost
    assert seen[0]["roles"]["tv bench"] == "surface"
    assert [b.label for b in out["A"]] == ["radiator"] and [b.label for b in out["B"]] == ["radiator"]
    doc = json.loads((tmp_path / "objects.json").read_text())
    assert doc["stats"]["naming"]["label_source"] == "claude" and doc["stats"]["naming"]["model"] == "claude-opus-5-5"
    assert [o["verified"] for o in doc["sessions"]["B"]] == [True, False]
    n = len(fake.calls)
    again = cli._object_boxes(tmp_path, room, B, True, "auto")                # cached objects.json: no new calls
    assert [b.label for b in again["A"]] == ["radiator"] and len(fake.calls) == n


def test_change_labeller_hears_the_checked_box_name():
    rad = named("radiator", BOX_C, BOX_S)
    pts = BOX_C + np.random.default_rng(0).uniform(-0.1, 0.1, (50, 3))
    name, _, _, checked = semantics._boxes_context(pts, 0.0, [rad])
    assert name == checked == "radiator"
    # the same hint for every labeller, so a keyless replay finds Claude's cached answer
    assert '"radiator"' in semantics._hint(checked, "claude") and semantics._hint(checked, "siglip") == semantics._hint(checked, "claude")
    owl = named("heater", BOX_C, BOX_S, source="owlv2")
    assert semantics._boxes_context(pts, 0.0, [owl])[3] == ""
