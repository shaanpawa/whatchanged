# Examples

The real before and after pair from my bedroom, so the results can be checked without filming anything.
`whatchanged demo` rebuilds `bedroom/` from its cached 3D in seconds, with no model download and no API key,
and CI does the same on every push.

| Folder | What | Made by |
|---|---|---|
| `bedroom/` | Detection run on A (before) vs B (after) | `whatchanged run A B --poses colmap` (`out/real`) |
| `bedroom/ground_truth.yaml` | What I really changed, written before the first run | by hand |
| `bedroom-repeat/` | The untouched pair, A vs A2: must report nothing | `whatchanged run A A2 --poses colmap` (`out/repeat`) |
| `bedroom-3d/` | The photoreal walkthrough: `index.html`, `scene.json`, `A.spz`, `B.spz`, `frames/` | `whatchanged room A B --detections out/real` (`out/pair`) |

The input videos are not in the repo (about 5 minutes of 4K each). Their frames are, at the size the
pipeline used them.

## What a run folder keeps

Enough to replay everything after the 3D step:

- `cache/A.npz`, `cache/B.npz`, `cache/diag.json`: keyframes, depth, cameras
- `frames/`: the keyframes (the change crops are cut from them)
- `objects.json`, `naming_cache.json`, `label_cache.json`: boxes and Claude's cached answers
- the outputs: `changes.json`, `report.html`, `maps/`, `viewpoints.json`, `scene_graph_*.json`,
  `markers_*.json`, `objects_*.txt`, `scene.rrd`, `crops/`, `object_crops/`, `lookahead/`

Left out on purpose (`.gitignore` drops most of them): `colmap/` and `work/` (hundreds of MB),
`scene_*.ply` (rebuilt by the replay), `*.log` (they print local folder names).

## Before committing

`cache/diag.json` and `changes.json` record the COLMAP folder and the frame paths as absolute paths, which
include the local user name. Make them relative, then check nothing is left. CI runs the same check.

```bash
for d in examples/bedroom examples/bedroom-repeat; do
  for f in "$d/cache/diag.json" "$d/changes.json" "$d/report.html"; do
    [ -f "$f" ] && sed -i '' "s#$PWD/##g" "$f"
  done
done
git grep --untracked -nIE '/Users/[A-Za-z]|sk-ant-[A-Za-z0-9_-]{16}' -- examples docs || echo clean
du -sh examples/*
```
