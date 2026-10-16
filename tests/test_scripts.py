"""The filming-night scripts, with no COLMAP run and no model: ingest.py strips every tag (location
included) and stops on HDR, and morning.sh's dry run passes its knobs to the right steps."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import ingest  # noqa: E402

has_ffmpeg = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not has_ffmpeg, reason="ffmpeg not installed")
TAGS = {"com.apple.quicktime.location.ISO6709": "+12.3456+065.4321+001.000/", "com.apple.quicktime.make": "Apple",
        "com.apple.quicktime.model": "iPhone 17", "com.apple.quicktime.creationdate": "2026-10-16T09:20:00+0100"}


def _clip(path: Path, hdr: bool = False) -> Path:
    """Two seconds of test pattern with sound and the tags an iPhone writes, location included."""
    raw = path.with_suffix(".raw.mov")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30", "-f", "lavfi",
                    "-i", "sine=frequency=440", "-t", "2", "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", "-c:a", "aac",
                    str(raw)], check=True)
    meta = [x for k, v in TAGS.items() for x in ("-metadata", f"{k}={v}")]
    colour = ["-color_trc", "arib-std-b67", "-color_primaries", "bt2020"] if hdr else []
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-map", "0", "-c", "copy", *colour, *meta,
                    "-movflags", "+write_colr+use_metadata_tags", str(path)], check=True)
    raw.unlink()
    return path


@needs_ffmpeg
def test_ingest_keeps_the_video_and_drops_location_sound_and_tags(tmp_path):
    src = _clip(tmp_path / "IMG_0102.MOV")
    assert b"ISO6709" in src.read_bytes()                  # the test clip really carries a location
    ingest.main([f"A={src}", "--out", str(tmp_path / "raw")])
    out = tmp_path / "raw" / "A.mov"
    assert ingest.verify(out) == []
    d = ingest.probe(out)
    assert [s["codec_type"] for s in d["streams"]] == ["video"]
    assert not any(k.startswith("com.apple") for k in d["format"].get("tags", {}))
    data = out.read_bytes()
    assert all(s not in data for s in (b"ISO6709", b"+12.3456", b"iPhone", b"com.apple"))
    rec = json.loads((tmp_path / "raw" / "ingest.json").read_text())["A"]
    assert rec["sources"] == ["IMG_0102.MOV"] and "12.3456" not in json.dumps(rec)
    # a second ingest of the same clip keeps the file, so morning.sh can reuse the work built from it
    before = out.stat().st_mtime_ns
    ingest.main([f"A={src}", "--out", str(tmp_path / "raw")])
    assert out.stat().st_mtime_ns == before


@needs_ffmpeg
def test_ingest_stops_on_hdr_unless_allowed(tmp_path):
    src = _clip(tmp_path / "IMG_0103.MOV", hdr=True)
    assert ingest.facts(src)["transfer"] == "arib-std-b67"
    with pytest.raises(SystemExit) as e:
        ingest.main([f"A={src}", "--out", str(tmp_path / "raw")])
    assert e.value.code == 2 and not (tmp_path / "raw" / "A.mov").exists()
    ingest.main([f"A={src}", "--out", str(tmp_path / "raw"), "--allow-hdr"])
    assert ingest.verify(tmp_path / "raw" / "A.mov") == []


def _dry_run(tmp_path, seconds=None, **env) -> dict[str, str]:
    """morning.sh --dry-run on placeholder clips (or test-pattern clips this many seconds long): the command
    line of every step, by step name."""
    if not all(shutil.which(t) for t in ("ffmpeg", "ffprobe", "colmap")) or not (REPO / ".venv/bin/whatchanged").exists():
        pytest.skip("morning.sh's preflight needs ffmpeg, colmap and the .venv install")
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for c, s in zip(("A", "A2", "B"), seconds or (None,) * 3):
        if s is None:
            (raw / f"{c}.mov").touch()
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=2", "-t", str(s),
                            "-c:v", "mpeg4", str(raw / f"{c}.mov")], check=True)
    r = subprocess.run(["bash", str(REPO / "scripts" / "morning.sh"), "--dry-run"], cwd=REPO, capture_output=True, check=False,
                       text=True, env={**os.environ, "RAW": str(raw), "OUT": str(tmp_path / "out"),
                                       "CONFIG": str(tmp_path / "config.json"), "GT": str(tmp_path / "no_key.yaml"), **env})
    assert r.returncode == 0, r.stdout + r.stderr
    steps = {}
    for line in r.stdout.splitlines():
        if ">>> " in line:
            name, _, cmd = line.split(">>> ", 1)[1].partition(": ")
            steps[name] = cmd
    return steps


def test_morning_dry_run_lists_every_step_in_order(tmp_path):
    steps = _dry_run(tmp_path)
    assert list(steps) == ["cut-A", "cut-B", "fast-room", "fast-run", "fast-viewer", "fast-video", "pair-room",
                           "repeat-room", "repeat-run", "calibrate", "repeat-report", "real-run", "pair-viewer",
                           "pair-video"]
    assert "--labels auto" in steps["fast-run"] and "--labels auto" in steps["repeat-run"]
    assert "--config" in steps["real-run"] and "--poses colmap" in steps["real-run"]


def test_morning_label_knobs_reach_only_their_runs(tmp_path):
    steps = _dry_run(tmp_path, FAST_LABELS="siglip", REPEAT_LABELS="siglip")
    for s in ("fast-run", "repeat-run", "repeat-report"):
        assert "--labels siglip" in steps[s], s
    assert "--labels" not in steps["real-run"]            # the real pair always gets Claude when a key is set


def test_a_shell_that_only_mentions_a_heavy_job_is_not_waited_for(tmp_path):
    """morning.sh waits for other heavy jobs. A wrapper shell or log watcher whose command line names one
    (zsh -c "... bin/whatchanged run ...") must not count, or the run waits forever; the job itself must."""
    script = (REPO / "scripts" / "morning.sh").read_text()
    heavy = next(line for line in script.splitlines() if line.startswith("HEAVY="))
    start = script.index("heavy_pids() {")
    func = script[start:script.index("\n}\n", start) + 3]
    marker = f"bin/whatchanged run {tmp_path.name}"          # unique, so other jobs on this machine don't count
    shell = subprocess.Popen(["/bin/sh", "-c", f"sleep 30; : {marker}"])
    job = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", marker])
    try:
        import time
        time.sleep(0.5)
        r = subprocess.run(["bash", "-c", f"{heavy}\n{func}\nheavy_pids"], capture_output=True, text=True, check=False)
        pids = set(r.stdout.split())
        assert str(job.pid) in pids
        assert str(shell.pid) not in pids
    finally:
        shell.kill()
        job.kill()


def test_morning_keyframes_follow_the_shortest_clip_and_match_between_runs(tmp_path):
    def frames(steps):
        got = {s: steps[s].split("--frames ")[1].split()[0] for s in ("repeat-run", "real-run")}
        assert got["repeat-run"] == got["real-run"]       # thresholds frozen on the repeat must fit the real pair
        return int(got["real-run"])
    assert frames(_dry_run(tmp_path / "long", seconds=(170, 100, 240))) == 40      # 0.4 a second of the 100 s clip
    assert frames(_dry_run(tmp_path / "longer", seconds=(170, 180, 240))) == 64    # capped
    assert frames(_dry_run(tmp_path / "unreadable")) == 24
    assert frames(_dry_run(tmp_path / "set", FRAMES="50")) == 50
