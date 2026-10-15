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
