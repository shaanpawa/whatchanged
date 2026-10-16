# scripts: filming and running a before/after pair

Run from the repo root. Clips go to `data/raw/` and results to `out/` (both git-ignored).

| When | Command | Takes |
|---|---|---|
| Test clip copied over | `.venv/bin/python scripts/ingest.py pilot=IMG_0100.MOV` | seconds |
| | `.venv/bin/python scripts/quickcheck.py data/raw/pilot.mov` | under 1 min |
| A and A2 filmed, **before moving anything** | `.venv/bin/python scripts/ingest.py A=IMG_0102.MOV A2=IMG_0103.MOV` | seconds |
| | `.venv/bin/python scripts/quickcheck.py data/raw/A.mov` | about 5 min for 5 min of video |
| B filmed | `scripts/protocol.sh A=IMG_0102.MOV A2=IMG_0103.MOV B=IMG_0107.MOV` | fast pair, then the full runs |

Bare file names are looked up in `~/Downloads`. Instead of names you can give the clock times you
wrote down, `--times "pilot=09:05 A=09:20 A2=09:33 B=10:08"` (add `--dry-run` to see the matching).
A take that broke into two clips: `A=IMG_0102.MOV+IMG_0104.MOV` joins them in order.

**ingest.py** copies the video stream only (no re-encode, no audio, no Apple location, make, model or
date tags; checked with ffprobe and by reading the file's atoms) and prints codec, size, frame rate
(average vs nominal), length, colour transfer and rotation. HDR (HLG or PQ) stops it: re-film in SDR,
or pass `--allow-hdr`. An output already made from the same clip is kept (`--redo` rewrites it).

**quickcheck.py** keeps the sharpest frame per second (ffmpeg hardware decoding, about 12x real time
on 4K HEVC), runs COLMAP on them with the pipeline's own `room3d.run_colmap`, and prints PASS/FAIL for:
SDR, steady frame rate, at least 95% of frames posed, camera heights spanning at least 1.0 m (0.7x the
median height if the marker isn't seen), at least 5% of frames 35 degrees or more down. It also reports
blur, exposure, a flicker heuristic, looking-up share, headings, lens, floor marker, and **which seconds
did not pose**: re-film that pass while the room is untouched. `--no-colmap`: footage checks only.
The trial video (45 s, HLG, 1x) takes 21 s and fails SDR, height span and the 35-degree gate, as it
should. It also posed only 17/39 frames at 1 fps but 71/74 at `--fps 2`, so the protocol runs at 2 fps.

**protocol.sh** runs one heavy step at a time (and first waits for any other COLMAP, MapAnything
or splat job):

1. ingest (skipped if `data/raw/{A,A2,B}.mov` exist and no clips are given)
2. fast pair: first 90 s of A and B, `room --fps 2 --iters 7000`, marker, `run --poses colmap`,
   `room --detections`, `video` → `out/fast/report.html`, `out/fast-room/index.html` and
   `out/fast-room/whatchanged.mp4`, the first real result
3. full: `room A B --fps 2` (model + splats) → `room A A2 --fps 2 --iters 2000` → `run A A2` →
   `calibrate` (writes `config.frozen.json`: commit it) → `report A A2` on the frozen config →
   `run A B --config config.frozen.json` → `room A B --detections out/real` → `video` →
   `score` when an answer key is given (`GT=path/to/ground_truth.yaml`)
4. report: paths, the change lists (the untouched A vs A2 must show 0) and step times

Each step logs to `out/protocol/NN-step.log` (times in `out/protocol/timings.tsv`); a failed step prints
`STOP` and the log to read. Resume with `--from fast|full|report` (finished work is reused, seconds per
cached step); `--only fast` runs one stage; `--dry-run` prints the commands. A folder built from older
clips stops the run with the `mv` to type. If calibration fails (the untouched pair still shows a
change), `SKIP_CALIBRATE=1 scripts/protocol.sh --from full` gets A vs B on the default thresholds.
Knobs: `FPS FRAMES SIDE_MM FAST_S ITERS GT ...` (top of the script). `FAST_LABELS=siglip REPEAT_LABELS=siglip`
keeps Claude's calls (and cost) for the real pair only.

Afterwards, the results go into the repo by hand: the run folders into `examples/` (what to keep and how
to strip local paths: [examples/README.md](../examples/README.md)) and the video, GIF and screenshot into
`docs/media/` ([docs/media/README.md](../docs/media/README.md)).

**marker_scale.py** `out/pair --side-mm 150` triangulates the floor marker's corners (ArUco DICT_4X4_50,
id 0) from every registered undistorted frame and prints metres per COLMAP unit, the spread across the
four sides, the reprojection error, and how far MapAnything's scale is from it. protocol.sh passes it
as `--metres-per-unit` whenever the marker is found. Measure the printed square and set `SIDE_MM`.
`--check desk=123.4cm` prints a tape distance in model units; `--selftest` checks the maths on a
synthetic square (0.2% off; a camera with a 1.7-degree pose error is dropped).
