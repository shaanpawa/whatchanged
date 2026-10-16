# Examples

The real before and after pair from my bedroom, so the results can be checked without filming anything.
`whatchanged demo` rebuilds `bedroom/` from its cached 3D into `out/demo` in about a minute, with no model
download and no API key, and CI does the same on every push.

| Folder | What | Made by |
|---|---|---|
| `bedroom/` | Change detection on A (before) vs B (after): report, `changes.json`, maps, object boxes, scorecard | `whatchanged run A B --poses colmap` |
| `bedroom/ground_truth.yaml` | What I really changed, from my note when filming | by hand |
| `bedroom-3d/` | The photoreal walkthrough: `index.html`, `scene.json`, `A.spz`, `B.spz`, `frames/` | `whatchanged room A B --detections bedroom` |

**Inputs.** The two videos (73 s and 85 s) are in the
[example-inputs release](https://github.com/shaanpawa/whatchanged/releases/tag/v1.0), as 1600 × 900 copies
with all metadata removed. They were filmed in 4K at 60 fps with the 0.5x lens; the pipeline read 1080p,
30 fps copies. Their keyframes are in `bedroom/frames/`, at the size the pipeline used them.

No untouched repeat (A2) was filmed for this pair, so it ran on the default thresholds. `bedroom-repeat/` is
where one goes (`whatchanged run A A2 --poses colmap`); CI checks it reports nothing once it is there.

## What a run folder keeps

Enough to replay everything after the 3D step:

- `cache/A.npz`, `cache/B.npz`, `cache/diag.json`: keyframes, depth, cameras; `cache/dets2d.json`: every 2D detection
- `frames/`: the keyframes (the change crops are cut from them)
- `objects.json`, `naming_cache.json`, `label_cache.json`: boxes and Claude's cached answers
- the outputs: `changes.json`, `report.html`, `scorecard.md`, `maps/`, `viewpoints.json`, `scene_graph_*.json`,
  `markers_*.json`, `objects_*.txt`, `scene.rrd`, `crops/`, `object_crops/`, `lookahead/`

Left out: `colmap/` and `work/` (hundreds of MB), `scene_*.ply` (rebuilt by the replay) and logs. Paths
in the committed files are relative to the repo, and CI fails if a local home folder appears in any file.
