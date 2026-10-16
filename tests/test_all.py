"""`whatchanged all`: every step, in order, each in its own process, stopping at the first failure."""

import subprocess
from types import SimpleNamespace

import pytest

from whatchanged import cli


def _run(monkeypatch, argv, fail_at=None):
    calls = []

    def fake(cmd):
        calls.append(cmd[3:])                       # drop "python -m whatchanged.cli"
        return SimpleNamespace(returncode=1 if len(calls) == fail_at else 0)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(cli, "auto_frames", lambda videos: 29)
    cli.main(["all", "a.mp4", "b.mp4", "--out", "o", *argv])
    return calls


def test_all_runs_room_then_changes_then_viewer_video_and_score(monkeypatch):
    calls = _run(monkeypatch, ["--key", "key.yaml"])
    assert [c[0] for c in calls] == ["room", "run", "room", "video", "score"]
    room, run, viewer, video, score = calls
    assert room[1:4] == ["a.mp4", "b.mp4", "--fps"] and "--detections" not in room
    assert run[run.index("--poses") + 1] == "colmap" and run[run.index("--colmap-from") + 1] == "o/room"
    assert run[run.index("--frames") + 1] == "29" and run[run.index("--out") + 1] == "o/run"
    assert viewer[viewer.index("--detections") + 1] == "o/run"
    assert video == ["video", "o/room", "--detections", "o/run"] and score == ["score", "o/run", "key.yaml"]


def test_all_stops_at_the_first_failed_step(monkeypatch):
    with pytest.raises(SystemExit, match="step 2"):
        _run(monkeypatch, ["--no-video"], fail_at=2)
