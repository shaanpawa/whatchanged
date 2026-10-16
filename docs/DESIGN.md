# Design note

A robot that comes back to a room it has mapped needs three answers: what is different, where exactly, and
what it can no longer vouch for. This note gives the key choices behind `whatchanged`, the trade-offs, and
the measurements that decided them. Unless it says otherwise, a number comes from a trial video of my
bedroom (iPhone 17, 45 s, one walk split into two overlapping clips with nothing moved), from experiments
run on it during development (their outputs are too big for the repo), or from the unit tests.

Five rules run through it:

1. **Geometry measures, models describe.** Cameras, depth, where things are and how far they moved are
   measured. Neural models give depth, names and a second witness to what changed, and are checked against
   the measurements and each other: Claude can veto a change, never create one.
2. **"Can't tell" is an answer.** Space the other video never saw is not reported as unchanged or as removed.
3. **Nothing invented.** No generated pixels or shapes in anything a robot would read.
4. **Runs on a laptop.** CPU or an Apple GPU, no CUDA needed, and only weights with commercial-use licences.
5. **Settings are frozen before the real test.** Thresholds come from an untouched repeat, then the before and
   after pair is run and scored against an answer key. The real pair broke this, and section 5 says how.

## 1. One COLMAP model for both videos; MapAnything only for depth

**First version.** MapAnything ran once over keyframes from both videos and returned every camera in one
metric frame. It worked: 0 false changes on the untouched trial, static surfaces 1.5 cm apart.

**What went wrong.** On this footage MapAnything guesses a focal length about 2.3 times too short (208 px
against COLMAP's 471 px at 518 px wide), so its field of view is far too wide. The Apache checkpoint ignores the intrinsics it is given,
so the guess bends its poses and makes its metric depth about 2 times too short: taken at face value, it
put the phone 0.72 m above the floor. The photoreal splats need COLMAP poses anyway, and with two separate
camera frames the object boxes landed 30 to 40 cm off the splats.

**Now.** The sharpest frames of both videos go into one COLMAP model: SIFT, matches between neighbours in
time plus a strided all-pairs net that closes loops and ties the videos together, then the global mapper.
MapAnything runs per video, so it never sees both versions of a changed spot, and supplies depth only.

- A plain per-frame depth scale was not enough: the ratio to COLMAP's points spread by 34% across frames,
  because the network stretches depth. Each frame now gets an affine fit to its own COLMAP points (2.1%
  median misfit), then a local correction spread 18 px from each point.
- Depth with no COLMAP point nearby is dropped (about half the pixels survive). A white wall without points
  had sat up to 40% too near; video B then "saw through" it and gave 5 false changes. Dropped, it becomes
  *can't tell*.
- Metric scale is MapAnything's depth times the ratio of COLMAP's focal length to the network's. That moved
  the camera from 0.72 m to 1.39 m, with the bed top at 0.53 m and the desk at 0.70 m. A printed ArUco
  marker on the floor (`scripts/marker_scale.py`) can replace it.

| Untouched trial, 24 keyframes per clip | MapAnything poses | COLMAP poses |
|---|---|---|
| False changes (plain point difference) | 0 (63) | 0 (40) |
| Static surfaces, median distance | 1.5 cm | 0.8 cm |
| Noise floor, by distance band | 2.5 to 7.8% | 1.0 to 5.3% |
| Boxes on the splats | 30 to 40 cm off (separate model) | exact: all 48 cameras agree to 0.07 mm |

**Trade-off.** COLMAP is slow on a CPU and grows faster than the frame count (127 to 305 frames took 6.8
times longer). It loses the pose on fast turns (5 gaps of 1 to 6.6 s on the trial) and needs texture. A guard
stops the run before any depth or training if either video has under half its frames in the model.
`--poses mapanything` stays for machines without COLMAP and for the fork-and-run GitHub workflow.

## 2. A visibility vote with "can't tell"

The obvious method, a point is changed if the other video has nothing within 5 cm, fires on every
difference in viewpoint: 40 false changes on the untouched trial. The brief asks to separate real changes
from viewpoint, occlusion and incomplete observation, so the test is about what each camera could see.

Space is cut into 3 cm voxels. For every voxel one video saw, each camera of the other video that looks at
it compares its own depth there: further away is *free* (it saw through where the thing was), about the same
is *seen*, nearer is *occluded*. A voxel is changed with at least 3 free votes and a free share of at least
60%. "Free" must hold over a 5 px neighbourhood, so an object's outline seen from a new angle does not look
like it vanished, and at least 3 of the voxel's own frames must agree a surface was there. Votes count only
within 3.5 m. Too few votes is *can't tell*, split into *hidden* (the other video's view was blocked) and
*not filmed*. This follows Fehr et al. (ICRA 2017) and OctoMap: unknown is not free.

*Can't tell* is used, not hidden away: it feeds the stale cells of the map update and the look-next
viewpoints (section 10).

## 3. The room sets its own threshold

There is no fixed depth tolerance. Most of what both videos saw did not change, so the robust spread
(1.4826 × median absolute deviation) of their depth disagreement over co-visible voxels is the noise floor.
It is measured in 6 distance bands, because depth error grows with distance, and the tolerance is 3 times it
(never under 3 cm). Every room runs with the same settings, and every run writes the smallest change it can
detect at each distance to `changes.json`. On the trial that was about 9 to 13 cm at 2 to 3 m and 25 to 31 cm
in the nearest and farthest bands. A mouse moved on a desk is below it; an answer key can mark such a change
`expected: miss` so it is not counted against recall.

`whatchanged calibrate` then searches the vote settings on the untouched repeat (A vs A2), takes the most
sensitive setting that reports nothing, and steps one notch stricter. The result goes to
`config.frozen.json` with the time and the code commit, before A vs B is run.

## 4. Moves are matched by shape

A removal and an addition are one moved object when their shapes match: the removed voxels are registered onto
the added ones by a turn about the vertical and a shift (from 12 starting turns), and at least half of each
shape must land within 1.5 voxels of the other. Pairs are chosen by the Hungarian method on shape fit, with
colour weighing in only lightly. The turn is reported only when the best fit clearly beats every fit more than
45° away, so a round bin is not said to have turned. Moves under 15 cm without a turn are not moves.

## 5. A second witness: the object detector

The vote rests on depth, and depth has one point of failure. On the first pass over the real pair, COLMAP's
model came out subtly wrong for the after video: its focal length 3% off, and B's depth fitted B's own COLMAP
points only to 19% (median) instead of about 2%. The noise floor rose to 16 to 39% of the distance and the
vote found nothing at all. Even two halves of the same video disagreed by up to 29%, so the alignment was not
the problem. Rebuilding the model fixed it (2.0% misfit, noise floor 1.9 to 3.8%), and every run now warns
when a video's depth fits its own points worse than 8%. The vote also struggles with things whose depth is
thin: a mesh bin you can see through, a speaker 10 cm across.

The detector needs no depth of that quality. For every object box and each video, it counts the frames that
should show the box (in front of the camera, its outline in the picture, not hidden behind nearer depth) and
how many have a detection of that kind of thing on it. An object found in at least 2 and half of one video's
frames, and in at most 20% of at least 3 frames of the other, is not there in the other video. Two of the
same name, one only before and one only after, at least 15 cm apart, are one move. The far end of a move is
often filmed only once or twice (the floor beside a bedside table), so a box of the same name at least 15 cm
away, which the other video detects at least once and more often than this one (in at most 40% of this one's
views), is taken as that end, for Claude to confirm.

A detector miss looks exactly like a removal, so each candidate goes to Claude with its own video's view and
the other video's view of the same spot, the same wireframe drawn in both. Is it one real object, and is it
still there? Only "absent" is kept. On the real pair it proposed 11 candidates; Claude kept 7 (both ends of
three moves, and the speaker's old place) and turned down 4: a window blind, books, a desk mat and a black
box behind the TV that was there in both videos. The colour search below adds a twelfth, which Claude also
confirmed. Each change's evidence says which witness found it.

**Turned in place.** The office chair was turned around where it stood. The vote saw the space its back had
filled go empty and called it removed, but the detector still finds the chair in that spot in both videos. A
removal or addition that lies inside an object both videos still detect steadily where it is, and is a real
part of it (at least a quarter of its longest side), is now reported as that object turned or shifted in
place. Supports (desks, beds, tables, shelves) never count, so a mug gone from a desk stays a removal.

**Following an object by its own colour.** The speaker's new place behind the TV never got a box of its own:
it is 10 cm across, mostly hidden, and its sightings merged with a black box beside it that is there in both
videos (Claude rightly said that box was there before). So an object that left, with no other end
confirmed, is looked for by its colour. The colour is learnt, not given: the strong pixels in the middle of
its box in the frames that detected it, kept only when most of them share one hue (a grey or black object
has none, and is skipped). In the other video, each detection of its kind is cut down to pixels of that hue,
so the black box beside it no longer drags the place off, and put in 3D from their depth. At least 2 frames
must agree within 25 cm, and the first video, where it looked at that place, must have seen that colour at
that depth in at most 20% of its looks. Then Claude confirms, as for any detector change. For the speaker:
5 after frames agree on one spot on the TV bench, 1.0 m from the desk; the before video saw no such colour
there in 8 of 8 looks; Claude confirmed. That made it 5 of 5.

These rules were written after earlier passes over the real pair, so it is not a held-out result for them;
the unit tests (including a blue object followed by its colour, and a grey one that is skipped) and the
trial are. They are only as good as the detector and the colour: a grey or black object that ends up hidden
stays "no longer at its old spot". The camera flakiness also shows in a fresh build: in 2 of 3 COLMAP builds
of this pair the after video's focal length came out 3 to 10% off, and the result got worse. The warning
catches it.

## 6. One object set for both videos

With a box set per video, each video boxed the room on its own (23 boxes in one clip of the untouched trial,
22 in the other), so switching between Before and After made static boxes appear and vanish though nothing had
changed. With shared COLMAP cameras there is one object set: OWLv2 runs on up to 40 keyframes per video,
detections from both videos are fused in 3D, and each box records which video saw it. A box differs between
states only where it overlaps a detected change: removed is Before only, added is After only, moved is one
object at two places.

Depth is sparse exactly where boxes matter (white radiators, chair backs, a mesh bin seen through), so a box
grows toward the outline of its 2D detections, each face only as far as raises its agreement with them, and
furniture is turned to line up with the walls when it is within 12°. On the trial, 75 boxes were checked:
44 kept, 6 dropped, 15 merged into others.

## 7. Claude as judge, not oracle

Vision-language models are good at "what is this" and poor at metric 3D. A robot's map also has to be
reproducible. So Claude Opus 5.5 never draws a box or decides a change. It answers fixed JSON schemas about
pictures that already have the geometry drawn in:

- **Room vocabulary** (one call, effort low): a contact sheet of 12 frames gives the room's own nouns,
  which join OWLv2's text queries. Alone, OWLv2 called the window blind a "refrigerator" and the radiator a
  "heater", and boxed the office chair as a "desk".
- **Box verdicts** (one call per box, effort medium): up to three crops from views that see the whole box
  unoccluded, ideally at least 20° apart, wireframe drawn in. Keep or drop, name, part of what, colour, fit.
  Boxes called too small or too big are refitted by geometry and shown again; a worse second answer puts the
  old box back. Overlapping boxes that are one thing are merged by geometry.
- **Detector-found changes** (one call each, effort medium): the object in its own video and the other
  video's view of the same spot, wireframe drawn in both. Real object? Still there? Only "absent" is kept
  (section 5).
- **Change names**: before and after crops, plus the name of the box at that spot, so the change and the
  box say the same word.
- **Video summary line** (effort low): thrown away if it contains any number not in the measured facts.

The change sentences are not written by a model. They are templates filled with the measured size, distance
and turn, and the run fails if a sentence holds a number that is not in its change's facts.

Every answer is cached by image hash and prompt version, so a rerun is free and identical. The trial took 85
calls, about $1.11. Without a key, boxes keep OWLv2's labels and changes are named by SigLIP 2 on the CPU.

## 8. Splats for looking, geometry for detecting

The brief asks for a clear, compelling presentation, and a room should look like the room, so each video
gets a 3D Gaussian splat (msplat on the Mac GPU, about 12 min for 30k steps on 127 frames) in the shared
COLMAP frame. Detection never uses the splats: a splat is only as good as the views it was trained on, and
what changed is exactly where the two videos' views differ. Each video gets its own splat, because one splat
trained on both turns every moved object into a ghost. Floaters outside the filmed space are dropped and the
result is written as compressed `.spz`, about 10 times smaller than PLY.

## 9. A locked walkthrough, not free flight

The first viewer let you fly anywhere. It looked photoreal on the walked path and smeared everywhere else.
Measured on the trial by rendering held-out frames:

| Rendered view | PSNR |
|---|---|
| Held-out frames on the walked path | about 30 dB |
| Within 2 cm and 3° of a filmed view | about 24 dB |
| 4 cm and 5° off | about 17 dB |
| The return walk, trained on the outbound walk | 20.3 dB median |
| 40 cm off the path | broken |

The trial walk saw only 42% of the floor plan from 3 or more directions 20° apart, never framed 90% of the
wall above 1.8 m, and saw 0.7% of the floor from steeper than 45°. Denser frames from the same video
(8 per second) gained only 0.8 dB. No renderer fixes views that were never filmed, so the viewer is a
walkthrough: the camera stands only where the phone stood, with the phone's own lens, and moves along the
walk with Play, scroll, the arrow keys or the timeline. Before and After swap at the same spot, or the
nearest spot both videos filmed. Choosing a change goes to the best view filmed in both and blinks between
them. The real phone frame can be shown beside the render.

The same cameras put the boxes on the real footage for the results video (SpatialLM's look, done with
COLMAP instead of ARKit). Extra frames are registered into the model. Only short, calm gaps between poses
are interpolated; where the pose is unknown the boxes fade out rather than be drawn from a guessed camera.

## 10. Robot outputs: stale, not free

The updated occupancy map (Nav2 `map_server` format) changes only cells video B actually saw. Cells it never
saw keep their old value and are listed as stale, instead of being overwritten as free or unknown. For each
*can't tell* region, the look-next planner tries standing spots on a 25 cm grid of floor both videos saw as
free, and scores each by the share of the region a camera there would see past the current obstacles. On the
trial, its best spot for a hidden region would see 71% of it; the nearest camera of video B turned toward it
would see 1%.

## 11. Capture is the biggest lever

A what-if on the trial's own camera path showed the 0.5x ultra-wide lens would raise the well-covered floor
from 29% to 59% and cut the never-framed wall from 34% to 3%. So the filming protocol asks for: the
ultra-wide lens, SDR, locked exposure and white balance, about 5 minutes per state at three heights, a
printed scale marker, an untouched repeat (A2) on a different route, and a check of A
(`scripts/quickcheck.py`) before anything is moved. `scripts/ingest.py` copies the
video stream only, so the Apple location, make, model and date tags never reach the repo.

## Next

- **Feed-forward splats for views off the filmed path.** Models such as InfiniSplat v2 and AnySplat turn a
  few dozen photos into Gaussians in one pass and hold up better between and beyond the filmed views. They
  need a CUDA GPU, so they would replace the Mac's per-room training on a Linux machine, for viewing only;
  detection does not use the splats.
- **One camera calibration for both videos** when they come from the same phone and lens, to stop the
  flaky focal length above (`WHATCHANGED_ONE_CAMERA=1` switches it on; not yet tested enough to be the default).
- **An untouched repeat (A2) for every pair,** to calibrate the thresholds before the real run, as the
  protocol asks.

## What I tried and dropped

| Tried | What happened | Decision |
|---|---|---|
| Free-flying splat viewer | 17 dB a few centimetres off the filmed path, broken at 40 cm | Locked walkthrough (section 9) |
| AI-generated models of changed objects (Meshy and Tripo via Higgsfield, 48 credits) | The chair looked right and stayed sharp from any angle, but its outline matched held-out real frames at only 0.36 to 0.66 IoU, against 0.83 to 0.86 on the frames it was made from; cutting the real chair out of the splat left a hole | Dropped: shapes a robot would read must be measured |
| Unreal Engine or Blender as the viewer | Unreal's splat plugins are Windows-only; neither is something a reviewer can run; neither adds detail the capture lacks | Browser viewer (three.js and Spark) |
| Generative fixers and world models (Difix3D+, Marble) | They fill unfilmed views with invented pixels | Dropped for the same reason |
| VGGT, Pi3, MASt3R-SLAM, running SpatialLM | Non-commercial licences; SpatialLM and MASt3R-SLAM also need CUDA | MapAnything (Apache-2.0) and COLMAP |
| Plain point difference | 40 false changes on the untouched trial | Kept as the baseline every run reports |
| A TSDF mesh fused from the depth | Median 1.3 cm from COLMAP's points (90% within 5.3 cm), but bumpy walls and holes where nothing was filmed | A prototype for a collision mesh, not in the pipeline yet |

## Numbers at a glance

| Measure | Trial (nothing moved) | Real pair (five things moved) |
|---|---|---|
| Frames posed by COLMAP | 166 of 168 (A), 161 of 163 (B) | 138 of 140 (A), 169 of 169 (B) |
| Depth fitted to COLMAP points | 2.1% median misfit, 52% of pixels anchored | 2.1% (A) and 2.0% (B), 41 to 45% anchored |
| Metric scale correction for the network's focal guess | 2.27 times | 1.19 times (0.5x lens) |
| Camera height after levelling | 1.39 m median | 1.48 m median |
| Static surfaces, A against B | 0.8 cm median | 2.1 cm median |
| Noise floor by distance band | 1.0 to 5.3% | 1.9 to 3.8% |
| Changes reported (plain point difference) | 0 (40) | 5, no false alarms (50) |
| Of the five real changes | | all 5 found: 4 moves and 1 turn in place |
| Objects kept after Claude's check | 44 | 50 |
| Claude cost | 85 calls, about $1.11 | about 100 calls, about $1.20 per full run |
| Unit tests | 94, about 15 s, on every push in CI, no models, no keys | |
