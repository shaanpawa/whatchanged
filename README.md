# whatchanged

**Two phone videos of a room in; out comes what was moved, added or removed, in 3D and in plain words,
plus the update to a robot's map.**

[![ci](https://github.com/shaanpawa/whatchanged/actions/workflows/ci.yml/badge.svg)](https://github.com/shaanpawa/whatchanged/actions/workflows/ci.yml)

<!-- HERO. Fill after the real run (see docs/media/README.md for the exact commands):
     docs/media/hero.gif         8 s loop of the footage part of the results video, 800 px wide, about 5 MB
     docs/media/whatchanged.mp4  the full results video from `whatchanged video`
     docs/media/walkthrough.png  one frame of the 3D walkthrough with a change selected
     For a video that plays inline, edit this file on github.com, drag whatchanged.mp4 into the editor and
     paste the https://github.com/user-attachments/assets/... line it gives you on its own line below. -->
<p align="center">
  <a href="docs/media/whatchanged.mp4"><img src="docs/media/hero.gif" width="100%" alt="The real before and after footage with tracked 3D boxes on what changed, then the 3D walkthrough"></a>
</p>

**[Results video](docs/media/whatchanged.mp4)** ·
**[Walk the room in your browser](https://shaanpawa.github.io/whatchanged/bedroom-3d/)** ·
**[Change report](https://shaanpawa.github.io/whatchanged/bedroom/report.html)** ·
**[Design note](docs/DESIGN.md)**

## What it does

The brief asks for three things. Each maps to one stage.

| The brief | whatchanged |
|---|---|
| A shared 3D representation from the two recordings | Both videos are posed together in **one COLMAP model**. MapAnything gives metric depth for every keyframe, pinned to COLMAP's points. Floor at z = 0, metres. Each video also gets a photoreal Gaussian splat in that same frame. |
| Identify and visualise what was moved, added or removed, in 3D | A **visibility vote** per 3 cm voxel: did the other video see through where this was? Too few votes means **can't tell**, kept apart from "unchanged". A removal and an addition with the same shape are one **move**, with its distance and turn. Shown as coloured 3D boxes in a walkthrough of the real room, on the real footage, in a report and in Rerun. |
| Describe the changes with labels or language | Every object gets a 3D box (OWLv2 detections fused in 3D). **Claude Opus 5.5** checks and names each box from crops. Sentences are templates filled with the measured numbers, and a check fails the run if a number is not in the facts. |

For a robot coming back to a room it mapped before, it also writes the **Nav2 map update** (cells the second
video never saw are marked stale, not free) and **where to stand next** to check anything it could not see.

<p align="center"><img src="docs/media/walkthrough.png" width="100%" alt="The 3D walkthrough: the photoreal room, labelled object boxes, a removed object in red and its sentence"></p>

## Results

<!-- RESULTS. Filled from the real run, not typed by hand:
     the table is out/real/scorecard.md (`whatchanged score out/real examples/bedroom/ground_truth.yaml`),
     the sentences are out/real/changes.json, the repeat line is out/repeat/changes.json,
     the scale line is out/pair/marker_scale.json and `scripts/marker_scale.py --check`. -->

My bedroom, filmed on an iPhone 17 (no LiDAR): before (A), an untouched repeat (A2) and after (B). The
answer key was written before the first run.

| # | What really changed | Result | Detected | Move error |
|---|---|---|---|---|
| 1 | _filled from scorecard.md_ | | | |

- **Sentences it wrote:** _filled from changes.json_
- **Untouched repeat (A vs A2, nothing moved):** _N_ changes. The plain point difference reports _N_.
- **Scale:** the printed floor marker gives _X_ m per COLMAP unit; MapAnything's depth gives _Y_ (_Z_% off).
  A tape-measured _edge_ of _L_ cm comes out at _M_ cm.
- **Claude:** _N_ calls, about $_C_ for the pair, all cached.

On a trial video of the same room (one walk split into two overlapping clips, nothing changed): **0 changes**,
where the plain point difference reports 40. Static surfaces of the two clips agree to 0.8 cm (median).

## Quickstart

Needs macOS on Apple Silicon for the photoreal room (Linux works for detection, see below),
[uv](https://docs.astral.sh/uv/), and COLMAP 4 and ffmpeg (`brew install colmap ffmpeg`).

```bash
git clone https://github.com/shaanpawa/whatchanged && cd whatchanged
uv sync --extra room --extra claude   # Python 3.12, PyTorch, MapAnything, msplat
uv run whatchanged demo               # rebuild the bundled example from its cached 3D
```

On your own two videos (MapAnything fetches 4.9 GB of weights on the first run):

```bash
uv run whatchanged check before.mov after.mov            # HDR, portrait, short clips
uv run whatchanged run before.mov after.mov --poses colmap --out out/run
uv run whatchanged room before.mov after.mov --detections out/run --out out/room
uv run whatchanged video out/room --detections out/run   # out/room/whatchanged.mp4
open out/run/report.html                                 # the change report
python3 -m http.server -d out/room 8000                  # the walkthrough, at localhost:8000
```

`scripts/morning.sh` runs the whole protocol one heavy step at a time: ingest (strips location tags), a
fast first pass, the untouched repeat, calibration, the real pair, the viewer, the video and the score.
See [scripts/README.md](scripts/README.md).

**Claude is optional.** Put `ANTHROPIC_API_KEY=...` in a git-ignored `.env`. Without it no API is called:
boxes keep OWLv2's labels and changes are named by SigLIP 2 on the CPU.

**Without a Mac or COLMAP:** `whatchanged run before.mp4 after.mp4` (MapAnything's own poses) runs on Linux,
CPU or CUDA. Or fork the repo, open *Actions → run on your own videos*, paste two video links and download
the report from the finished run.

| Command | What it does |
|---|---|
| `check A B` | Footage warnings before any 3D |
| `run A B --out DIR` | Detection pipeline (`--poses colmap` shares cameras with `room`) |
| `room A B --out DIR --detections RUN` | One Gaussian splat per video, walkthrough viewer with the boxes |
| `video ROOM --detections RUN` | Results video: title, real footage with tracked boxes, 3D tour, end card |
| `report DIR` | Redo everything after the 3D step from the cache (seconds) |
| `calibrate DIR` | Thresholds from an untouched pair, frozen to `config.frozen.json` |
| `score DIR key.yaml` | Recall, precision (95% intervals), traps, move error against an answer key |

## How it works

```mermaid
flowchart TB
  A["Video A<br/>before"] --> COL
  B["Video B<br/>after"] --> COL
  subgraph S3D["One shared 3D frame"]
    COL["COLMAP<br/>both videos, one model"]
    DEP["MapAnything depth<br/>pinned to COLMAP points"]
    SPL["Gaussian splat per video<br/>for viewing only"]
  end
  subgraph CHG["What changed: geometry"]
    VOTE["Visibility vote<br/>per 3 cm voxel"]
    CLS["removed, added, moved<br/>or can't tell"]
  end
  subgraph SEM["What it is: semantics"]
    OWL["OWLv2 boxes<br/>fused in 3D"]
    CL["Claude Opus 5.5<br/>checks and names"]
    SEN["Sentences from<br/>measured facts"]
  end
  subgraph OUT["Outputs"]
    VIEW["Walkthrough<br/>and video"]
    ROBOT["Nav2 map update<br/>look-next, ROS markers"]
    DATA["changes.json<br/>report, Rerun"]
  end
  COL --> DEP
  COL --> SPL
  DEP --> VOTE --> CLS
  DEP --> OWL --> CL
  CLS --> SEN
  CL --> SEN
  SPL --> VIEW
  SEN --> VIEW
  CLS --> ROBOT
  SEN --> DATA
```

1. **Poses.** The sharpest frame in each short window of both videos goes into one COLMAP model (SIFT,
   neighbours in time plus a sparse all-pairs net that closes loops, global mapper). If either video does not
   join the model, it stops there, before any depth or training, and says which.
2. **Depth.** MapAnything runs on each video's keyframes separately, so it never sees both versions of a
   changed spot. Each frame's depth is fitted to the COLMAP points it saw; depth with no point nearby (a
   plain white wall) is dropped, so it becomes *can't tell* instead of a guess. Scale comes from the
   network's metric depth, corrected for its own focal-length guess, or from a printed floor marker.
3. **Vote.** For every voxel one video saw, each camera of the other video compares its own depth there:
   further means *free* (it saw through where the thing was), equal means *seen*, nearer means *occluded*.
   Removals are A judged by B; additions are B judged by A.
4. **Tolerance from the room.** Most of what both videos saw did not change, so the spread of their depth
   disagreement, per distance band, is the noise floor. The tolerance is 3 times it. Thresholds are then
   frozen on the untouched pair before the real pair is run.
5. **Objects.** OWLv2 finds objects in up to 40 keyframes per video, with the room's own nouns added to its
   queries. Detections from both videos are fused into **one** set of 3D boxes, so Before and After show the
   same static objects and differ only where a change is.
6. **Names and sentences.** Claude judges each box (see below). Each change is named from its before and
   after crops, told the name of the box it sits in so the two agree. Its sentence is a template filled with
   the measurements, for example *"The wooden chair moved about 1.2 m from beside the window to beside the
   bed and turned about 90°."*
7. **Map and next view.** The updated occupancy map patches the old one only where video B looked. For each
   region it could not check, it scores standing spots on free floor by how much of the region a camera there
   would see past the current obstacles.

Why each choice, with the numbers behind it: [docs/DESIGN.md](docs/DESIGN.md).

## Outputs for robots

All in one levelled metric frame: metres, z up, floor at z = 0, yaw about +z.

| File | What it is | Format |
|---|---|---|
| `maps/map_{before,after,updated}.yaml/.pgm` | Occupancy maps. *Updated* changes only cells video B saw; cells it never saw keep their old value and are listed as stale (`map_update.json`) | Nav2 `map_server` |
| `viewpoints.json` | Where to stand and which way to face to settle each *can't tell* region, with the share it would see | x, y, yaw, tilt |
| `markers_{before,after}.json` | One box and one label per object | `visualization_msgs/MarkerArray` (rosbridge JSON) |
| `scene_graph_{before,after}.json` | Objects with *on* and *near* relations | JSON, frame `map` |
| `objects_{A,B}.txt` | The object layout | SpatialLM `Bbox(class,x,y,z,angle_z,scale_x,scale_y,scale_z)` |
| `objects.json` | One object set for both videos: state (both, before only, after only, moved), who saw it, Claude's verdict | JSON |
| `changes.json` | Every change: type, name, sentence, 3D box, evidence (free views, share, noise margin), video timestamps, *can't tell* regions, smallest detectable change | JSON |
| `scene.rrd` | Both scans, every camera, the boxes and the changes | [Rerun](https://rerun.io) |
| `A.spz`, `B.spz`, `scene.json` | A Gaussian splat per video in the shared frame, and the camera path | SPZ, for a simulator or the viewer |

## Where Claude is used, and where it is not

Claude Opus 5.5 is a **judge**. Geometry places every box, decides every change and measures every number.

| Claude does | Claude does not |
|---|---|
| List the room's objects from one contact sheet, so OWLv2 looks for "radiator" instead of forcing it into "heater" | Estimate cameras, depth or scale |
| Look at up to three crops of each 3D box with its wireframe drawn in: keep or drop, a plain name, part of what, colour, and whether the box fits | Draw, place or size a box (a box it calls too small or too big is refitted by geometry and shown to it again) |
| Name each change from before and after crops, told the name of the box at that spot so the two agree | Decide what changed, where, or how far it moved |
| Write the video's one-line summary, rejected if it contains a number that is not in the measured facts | Write the change sentences (templates over measured facts) |

Every answer is structured JSON, cached on disk by image hash and prompt, so a rerun costs nothing and gives
the same result. On the trial room: 85 calls, about $1.11. Without a key, everything runs locally.

## Evaluation

- **Unit tests (on every push, about 10 s, no models or keys).** A synthetic room rendered by ray casting checks the
  brief's traps: a viewpoint change alone reports nothing, a removed box is *removed*, a box hidden behind a
  new obstacle is **not** called removed, a moved and turned box is *moved* with its distance and turn.
  Also: COLMAP guards, the shared object set, Claude naming with a fake client (refusals, bad key, merges),
  and the video's pose interpolation.
- **Untouched repeat.** A second recording with nothing moved (A2) must report nothing. Thresholds are
  calibrated on it and frozen to `config.frozen.json`, with the time and code commit, before A vs B is run.
- **Answer key and tape.** The real pair is scored against `examples/bedroom/ground_truth.yaml`, written
  before the first run, with 95% Wilson intervals, an occlusion trap, and move distances checked against a
  tape measure and a printed ArUco marker.
- **Baseline.** Every run also reports the plain point difference (a point is changed if the other video has
  nothing within 5 cm). On the untouched trial it reports 40 changes; the visibility vote reports 0.
- **CI** rebuilds the committed example from its cache, checks it finds the same changes, scores it, and
  checks the untouched repeat reports nothing.

## Limitations

- **Phone video only.** No LiDAR or IMU. Scale comes from the network's metric depth or a printed marker.
- **Plain walls are blind spots.** Depth is kept only where COLMAP has points (about half the pixels on the
  trial), so a change against a white wall is *can't tell*, not found.
- **Small things are missed.** On the trial the smallest depth gap it trusts is about 9 to 13 cm at 2 to 3 m
  from the camera and 25 to 31 cm nearer or further. Every run writes its own figure to `changes.json`.
  A mouse moved on the desk is an expected miss.
- **Splats are sharp only near filmed views.** Held-out frames on the walked path render at about 30 dB
  PSNR; a few centimetres and degrees off the path they blur. So the viewer is a walkthrough along the filmed
  path, not free flight.
- **Fast turns lose poses.** COLMAP leaves gaps where the phone turned quickly; the footage overlay fades its
  boxes out there rather than guess.
- **Time.** _Filled from out/morning/timings.tsv._ COLMAP time grows faster than the frame count (6.8 times
  longer for 2.4 times the frames on the trial). The photoreal room and the video need an Apple Silicon Mac.
- **Lighting.** Both videos should be filmed under the same lights. A large lighting change can look like a
  change in depth.

## Related work

- Fehr et al., *TSDF-based change detection for consistent long-term dense reconstruction*, ICRA 2017: unexplored space is not empty space.
- Hornung et al., *OctoMap*, Autonomous Robots 2013: free, occupied and unknown as first-class map states.
- Weihs et al., *Visual Room Rearrangement*, CVPR 2021: an agent must find what changed in a room.
- Wu et al., *SceneDiff*, arXiv 2512.16908: a benchmark and method for multiview object change detection.
- Mao et al., *SpatialLM*, arXiv 2506.07491: the 3D box layout format and the boxes-on-real-footage look.
- Keetha et al., *MapAnything*, arXiv 2509.13414, and Schönberger and Frahm, *Structure-from-Motion Revisited* (COLMAP), CVPR 2016.

## How this was built

Coding agents (Claude Code) wrote most of the code from a plan I directed. The capture, the answer keys, the
design decisions and the checks of every output are mine. The experiments behind each decision are in
[docs/DESIGN.md](docs/DESIGN.md).

## Licences

Code: MIT. Models and libraries, all usable commercially:
MapAnything code and `map-anything-apache` weights (Apache-2.0), OWLv2 (Apache-2.0), SigLIP 2 (Apache-2.0),
COLMAP (BSD), msplat (Apache-2.0), Spark (MIT), three.js (MIT), Rerun (MIT or Apache-2.0).
Claude Opus 5.5 is called through the Anthropic API under Anthropic's terms; no weights are shipped.
