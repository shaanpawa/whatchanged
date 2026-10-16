# whatchanged

**Two phone videos of a room in, a robot's map update out.** `whatchanged` rebuilds both
videos in one 3D frame, works out what was moved, added or removed, says so in plain
sentences built from the measurements, updates the robot's floor map, and tells it where to
look for anything it couldn't check.

<!-- HERO: GIF of the report + links (live report · 3D in Rerun · 60 s video) -->

## Results

<!-- Filled from `whatchanged score` on the real recordings. No number here is hand-typed. -->

## Quickstart

```bash
git clone https://github.com/<you>/whatchanged && cd whatchanged
uv sync                      # Python 3.12, PyTorch, MapAnything (Apache-2.0 weights, 4.9 GB on first run)
uv run whatchanged demo      # rebuild the bundled example from its cached 3D, under a minute, no download
uv run whatchanged run before.mp4 after.mp4 --out out/myroom
open out/myroom/report.html
```

It runs on a Mac (CPU, about 2 minutes for 2 × 24 frames on an M4 Pro), on Linux with or
without an NVIDIA GPU, and on GitHub's free runners. No API key is needed; set
`GEMINI_API_KEY` for sharper object names.

**No install at all:** fork this repo, open *Actions → run on your own videos*, paste two
video links, and download the report from the finished run.

| Command | What it does |
|---|---|
| `whatchanged check A.mp4 B.mp4` | Warns about HDR, portrait, short or mismatched footage before any 3D |
| `whatchanged run A.mp4 B.mp4 --out DIR` | Full pipeline |
| `whatchanged report DIR` | Re-run everything after the 3D step from the cache (seconds) |
| `whatchanged score DIR ground_truth.yaml` | Compare with a hand-written answer key |

## The room in 3D (photoreal)

```bash
brew install colmap                      # camera poses (CPU is fine)
uv sync --extra room                     # msplat: Gaussian-splat training on the Mac GPU (Metal)
uv run whatchanged room room.mov --out out/room3d
python -m http.server -d out/room3d      # then open http://localhost:8000
```

One video in, a walkable photoreal copy of the room out, in a few minutes on an M-series Mac:

1. **Frames.** The sharpest frame in every quarter second; windows that are all motion blur are
   left out rather than teaching the model blur.
2. **Poses.** COLMAP: SIFT features, matches between neighbours in time plus a strided net of
   all-pairs that closes the loop when the walk returns to the door, then the global mapper.
3. **Splat.** [msplat](https://github.com/rayanht/msplat) trains a 3D Gaussian splat on the GPU.
   Faint splats, giant blobs and floaters outside the filmed space are dropped, and the result is
   written as a compressed `.spz` (about 10× smaller than PLY).
4. **Level and scale.** Up is the direction every camera's sideways axis is perpendicular to (a
   handheld phone has almost no roll), the floor is the lowest substantial layer under the cameras,
   and the scale comes from MapAnything's metric depth when a detection run is given
   (`--detections out/room`), otherwise from the filming height (`--camera-height`).
5. **Viewer.** `index.html` uses [Spark](https://sparkjs.dev) on three.js: *Fly* (drag + WASD),
   *Orbit*, and *Tour*, which replays the walk along the video's own camera path at the phone's
   field of view.

Two videos (`whatchanged room before.mov after.mov`) are posed together in **one** COLMAP model,
so both splats share a frame, and each video gets its own splat: training one splat on both would
turn every moved object into a ghost. With `--detections` the viewer also draws the labelled object
boxes and the changes (red removed, green added, blue moved with a ghost and an arrow), each with
its sentence.

For the boxes to sit on the right things, run the detection on the same cameras as the splats:

```bash
uv run whatchanged run before.mov after.mov --out out/pair-run --poses colmap
uv run whatchanged room before.mov after.mov --out out/pair --detections out/pair-run
```

The room then trains on the detection run's COLMAP model (or, the other way round,
`run --poses colmap --colmap-from out/pair` reuses the room's), so the boxes need no fitting at
all. Given a run on other copies of the footage (a self-split, a trimmed clip), the viewer fits
the two camera paths instead, finding each clip's start from the phone's turns, and leaves the
boxes out if fewer than 60% of the cameras agree to 15 cm.

The splats are for looking at the room. Change detection runs on the measured geometry below, not
on the splats: a splat is only as good as the views it was trained on, and what changed is
exactly where the two videos' views differ.

## What you get

| File | For |
|---|---|
| `report.html` | People: 3D viewer, one card per change with before/after crops, floor map, look-next |
| `changes.json` | Programs: every change with type, label, sentence, 3D box, evidence, video timestamps |
| `maps/map_{before,after,updated}.yaml/.pgm` | Robots: Nav2 `map_server` occupancy maps, plus `costmap_updated.png` |
| `viewpoints.json` | Robots: where to stand and which way to face to check each uncertain region |
| `scene.rrd` | Engineers: [Rerun](https://rerun.io) recording with both scans, every camera and the changes |

## How it works

1. **Keyframes.** About 24 sharp frames per video, each with its real timestamp.
2. **One shared 3D frame.** [MapAnything](https://github.com/facebookresearch/map-anything)
   runs once over frames from *both* videos to put every camera in one metric frame. The depth
   used for change testing comes from separate before-only and after-only passes, aligned to that
   frame, so the network never sees both versions of a changed spot at once. The floor is fitted
   and becomes z = 0. A static-alignment check reports how well walls from the two videos agree.
   With `--poses colmap` the cameras and intrinsics come instead from the COLMAP model the splats
   use (both videos in one bundle adjustment), and MapAnything only supplies depth. Each frame's
   depth is fitted to the COLMAP points it saw (scale and offset, then a local correction around
   the points), and depth with no COLMAP point nearby is left out, so a plain white wall becomes
   *can't tell* rather than a guess. The metric scale is MapAnything's depth over COLMAP's,
   corrected for the network's own focal-length guess. On the trial room that guess was 2.3×
   too short: uncorrected, the phone sat at 0.72 m; corrected, at 1.39 m, with the bed top
   at 0.53 m and the desk at 0.70 m.
3. **Did the other video see this spot?** Space is cut into 3 cm voxels. For every voxel seen in
   one video, each camera of the other video that can see it compares its own depth there:
   further away means *free* (it saw through where the thing was), about the same means *seen*,
   nearer means *occluded*. A spot is a change only with enough free votes; too few votes means
   **can't tell**, split into *hidden* and *not filmed*. This is the brief's "viewpoint, occlusion,
   or incomplete observations" line, made explicit.
4. **The room sets its own threshold.** Most of what both videos saw did not change, so the
   spread of depth disagreement over shared space is the noise floor. The tolerance is a fixed
   multiple of it, so every room runs with the same settings.
5. **Objects and moves.** Changed voxels are grouped into objects. A removal and an addition that
   match in size and colour are one *moved* object, reported with its distance.
6. **Names and sentences.** A vision model (Gemini, or SigLIP 2 offline) only names the object and
   what it sits on. Sentences are templates filled with the measured numbers, and a check fails the
   run if a sentence contains a number that isn't in the change's facts.
7. **Map update.** The new map patches the old one only where the second video looked. Cells it
   never saw keep their old value and are marked *stale*, instead of being overwritten as free.
8. **Where to look next.** For each region it couldn't check, it scores standing spots on free
   floor by how much of the region a camera there would see past the current obstacles.

## Design choices and trade-offs

<!-- short, honest; filled after the real runs -->

## Evaluation

- `tests/test_traps.py` renders a synthetic room by ray casting and checks, on every push, that a
  viewpoint change alone reports nothing, a removed box is *removed*, a box hidden behind a new
  obstacle is **not** called removed, and a moved box is *moved* with the right distance.
- The real recordings are scored against an answer key written before the first run, including
  an untouched repeat recording that must report nothing.
- Trial room, untouched, split into two overlapping clips (`--poses colmap`): 0 changes (the plain
  point difference reports 40). Static surfaces agree to 0.8 cm median. The noise floor is 1.0–5.3%
  of distance (with MapAnything's own poses: 2.5–7.8%).

## Limitations

<!-- filled from the real runs, with pictures of the failures -->

## Related work

- Fehr et al., *TSDF-based change detection for consistent long-term dense reconstruction*, ICRA 2017: unexplored space is not empty space.
- Wu et al., *SceneDiff*, arXiv 2512.16908: multiview object change detection benchmark and method.
- Weihs et al., *Visual Room Rearrangement*, CVPR 2021: an agent must find what changed in a room.
- Hornung et al., *OctoMap*, 2013: free / occupied / unknown as first-class map states.

## How this was built

Coding agents (Claude Code) wrote most of the code from a plan I directed. The capture, the
answer keys, the design decisions and the checks of every output are mine.

## Licences

Code: MIT. MapAnything code and the `map-anything-apache` weights: Apache-2.0. SigLIP 2: Apache-2.0.
