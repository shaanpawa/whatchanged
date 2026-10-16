# docs/media

The pictures the top-level README shows. All three come from the real run; none is made by hand.

| File | What | Made from |
|---|---|---|
| `whatchanged.mp4` | The full results video | `whatchanged video out/pair --detections out/real` |
| `hero.gif` | An 8 s loop of the real footage with the tracked 3D boxes (800 px, about 5 MB) | Part 2 of the video |
| `walkthrough.png` | One frame of the 3D walkthrough with a change selected | Part 3 of the video |

From the repo root, after `scripts/morning.sh` (pick `-ss` so the clip starts on a change):

```bash
cp out/pair/whatchanged.mp4 docs/media/whatchanged.mp4
ffmpeg -y -ss 2 -t 8 -i out/pair/whatchanged_parts/02_footage.mp4 \
  -vf "fps=10,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
  -loop 0 docs/media/hero.gif
ffmpeg -y -ss 6 -i out/pair/whatchanged_parts/03_tour.mp4 -frames:v 1 docs/media/walkthrough.png
ls -la docs/media    # keep each file under 50 MB, and the GIF under 10 MB
```

For a video that plays inside the README on GitHub, edit `README.md` on github.com, drag `whatchanged.mp4`
into the editor, and keep the `https://github.com/user-attachments/assets/...` line it inserts on its own
line under the hero.
