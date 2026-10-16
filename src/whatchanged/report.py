"""One self-contained report.html: 3D viewer, change cards, floor map, diagnostics."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

# The same colours as the 3D walkthrough and the video.
TYPE_COLOURS = {"moved": "#5aa9ff", "added": "#4cd97b", "removed": "#ff5a5a", "unsure": "#ffc857"}


def _b64_img(path: str | Path | None, max_side: int = 360) -> str:
    if not path or not Path(path).exists():
        return ""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _b64_png(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode() if path.exists() else ""


def _cloud(points: np.ndarray, colors: np.ndarray, n: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if len(points) > n:
        idx = rng.choice(len(points), n, replace=False)
        points, colors = points[idx], colors[idx]
    return {"p": base64.b64encode(points.astype(np.float32).tobytes()).decode(),
            "c": base64.b64encode(colors.astype(np.uint8).tobytes()).decode(), "n": int(len(points))}


def _evidence(c: dict) -> str:
    """One line on why the change was called: the depth vote, or the detector looking and not finding it."""
    ev = c["evidence"]
    other = "B" if c["type"] == "removed" else "A"
    if ev.get("by") == "objects":
        found = "none" if ev["free_share"] >= 0.999 else f"{1 - ev['free_share']:.0%}"
        where = (f"video {other} showed its spot in {ev['free_views']} frames" if c["type"] != "moved"
                 else f"each video showed the other's spot in at least {ev['free_views']} frames")
        return f"{where}; the object detector found it in {found} of them"
    return (f"{other if c['type'] != 'moved' else 'B'} saw it free in {ev['free_views']} views · free share {ev['free_share']:.0%} · "
            f"{ev['noise_margin']:.1f}× past the noise tolerance")


def _fmt_t(t):
    return "" if t is None else f"{int(t // 60):02d}:{t % 60:04.1f}"


def write(out_dir: Path, A, B, changes: list[dict], regions: list[dict], floor: dict, diag: dict,
          plans: list[dict] | None = None, title: str = "What changed in the room?", max_points: int = 120_000,
          boxes: dict | None = None) -> Path:
    plans = plans or []
    boxes = boxes or {}
    pa, ca = A.points(conf_pct=30, stride=2)
    pb, cb = B.points(conf_pct=30, stride=2)
    data = {
        "A": _cloud(pa, ca, max_points, 0), "B": _cloud(pb, cb, max_points, 1),
        "camsA": A.poses[:, :3, 3].round(3).tolist(), "camsB": B.poses[:, :3, 3].round(3).tolist(),
        "changes": [{k: c[k] for k in ("id", "type", "label", "centre_m", "size_m", "yaw_deg")}
                    | ({"before": {k: c["before"][k] for k in ("centre_m", "size_m", "yaw_deg")}} if c["type"] == "moved" else {})
                    for c in changes],
        "regions": [{"reason": r["reason"], "min": np.round(r["min"], 3).tolist(), "max": np.round(r["max"], 3).tolist()}
                    for r in regions],
        "colours": TYPE_COLOURS,
        "objects": {k: [b.to_dict() for b in v] for k, v in boxes.items()},
        "views": [{"id": p["id"], "cam": p["stand_at_m"] + [p["camera_height_m"]], "target": p["region_centre_m"]} for p in plans],
    }
    counts = {t: sum(c["type"] == t for c in changes) for t in ("moved", "added", "removed")}

    cards = []
    for c in changes:
        times = " / ".join(f"{k} {_fmt_t(v)}" for k, v in c["times"].items() if v is not None)
        extra = ""
        if c["type"] == "moved":
            turn = c["move"]["turn_deg"]
            extra = f" · moved {c['move']['distance_m']:.2f} m" + (f" · turned {abs(turn):.0f}°" if turn is not None else " · turn not determined")
        cards.append(f"""
      <article class="card" data-id="{c['id']}" tabindex="0">
        <div class="row"><span class="chip c-{c['type']}">{c['type']}</span><span class="name">{html.escape(c['label'])}</span><span class="cid">{c['id']}</span></div>
        <p class="sent">{html.escape(c['sentence'])}</p>
        <div class="thumbs"><figure><img src="{_b64_img(c['crops']['before'])}" alt="before"><figcaption>before (A)</figcaption></figure>
        <figure><img src="{_b64_img(c['crops']['after'])}" alt="after"><figcaption>after (B)</figcaption></figure></div>
        <p class="ev">{_evidence(c)}{extra}{' · ' + times if times else ''}</p>
      </article>""")
    if not cards:
        cards.append('<p class="muted">No changes found.</p>')

    reg_rows = "".join(
        f"<li><span class='chip c-unsure'>{'hidden' if r['reason'] == 'hidden' else 'not filmed'}</span> "
        f"around ({r['centre'][0]:.1f}, {r['centre'][1]:.1f}) m, {r['n']} voxels</li>" for r in regions[:6]) or "<li>None.</li>"
    look = "".join(
        f"""<article class="card" data-id="{p['id']}" tabindex="0"><div class="row"><span class="chip c-unsure">look next</span>
        <span class="name">{'hidden spot' if p['reason'] == 'hidden' else 'unfilmed spot'}</span><span class="cid">{p['id']}</span></div>
        <img class="prev" src="{_b64_img(p['preview'], 480)}" alt="predicted view">
        <p class="sent">Stand at ({p['stand_at_m'][0]:.2f}, {p['stand_at_m'][1]:.2f}) m, camera at {p['camera_height_m']:.1f} m,
        facing {p['face_yaw_deg']}°, about {p['distance_m']:.1f} m from the spot.</p>
        <p class="ev">predicted to see {p['predicted_visible']:.0%} of it · nearest camera of B would see {p['naive_visible']:.0%}</p></article>"""
        for p in plans) or '<p class="muted">Nothing left to check.</p>'
    al = diag.get("alignment_B_to_A_m", {})
    home = str(Path.home())  # the report gets published: show paths relative to the home folder
    diag_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v).replace(home, '~'))}</td></tr>"
                        for k, v in diag.items())

    page = TEMPLATE.replace("__TITLE__", html.escape(title)) \
        .replace("__COUNTS__", f"{counts['moved']} moved · {counts['added']} added · {counts['removed']} removed · {len(regions)} can't-tell regions") \
        .replace("__CARDS__", "".join(cards)) \
        .replace("__MAP__", _b64_png(out_dir / "maps" / "map_update.png")) \
        .replace("__FLOOR__", html.escape(json.dumps(floor))) \
        .replace("__BLOCKED__", f"{floor.get('newly_blocked_m2', 0):.2f}") \
        .replace("__FREED__", f"{floor.get('newly_free_m2', 0):.2f}") \
        .replace("__STALE__", f"{floor.get('stale_m2', 0):.2f}") \
        .replace("__REGIONS__", reg_rows) \
        .replace("__LOOK__", look) \
        .replace("__ALIGN__", f"{al.get('p50', 0) * 100:.1f} cm (median), {al.get('p90', 0) * 100:.1f} cm (90th percentile)") \
        .replace("__DIAG__", diag_rows) \
        .replace("__DATA__", json.dumps(data))
    path = out_dir / "report.html"
    path.write_text(page)
    return path


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="dark">
<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400..700&display=swap">
<style>
/* The same glass as the 3D walkthrough and the video: frosted, darkened, a bright rim along the top edge. */
:root{color-scheme:dark;--bg:#0b0b0c;--ink:#f6f7f9;--ink-2:rgba(246,247,249,.8);--amber:#ffc857;
--moved:#5aa9ff;--added:#4cd97b;--removed:#ff5a5a;--unsure:#ffc857;
--glass-fill:rgba(22,24,29,.6);--glass-sheen:linear-gradient(180deg,rgba(255,255,255,.08),rgba(255,255,255,.02) 42%,rgba(255,255,255,.03));
--glass-filter:blur(22px) saturate(1.7) brightness(.74);
--rim:inset 0 1px 0 rgba(255,255,255,.34),inset 0 -1px 0 rgba(255,255,255,.08),inset 0 0 0 1px rgba(255,255,255,.08);
--drop:0 14px 34px -14px rgba(0,0,0,.7),0 2px 8px -2px rgba(0,0,0,.35);--fill-1:rgba(255,255,255,.07);--fill-2:rgba(255,255,255,.13);
--font:"Geist",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;--mono:ui-monospace,SFMono-Regular,Menlo,monospace;--ease:cubic-bezier(.2,.8,.2,1)}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font:15px/1.5 var(--font);-webkit-font-smoothing:antialiased;background:var(--bg);
background-image:radial-gradient(60vw 40vw at 10% -5%,rgba(255,200,87,.09),transparent 70%),radial-gradient(60vw 45vw at 100% 40%,rgba(90,169,255,.07),transparent 70%);background-attachment:fixed}
.glass{background:var(--glass-sheen),var(--glass-fill);box-shadow:var(--rim),var(--drop);-webkit-backdrop-filter:var(--glass-filter);backdrop-filter:var(--glass-filter)}
:focus-visible{outline:2px solid var(--amber);outline-offset:2px}
a{color:inherit}
.top{position:sticky;top:12px;z-index:5;margin:12px auto 0;width:min(1200px,calc(100% - 24px));padding:14px 14px 14px 22px;border-radius:24px;
display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:12px 24px}
.mark{margin:0 0 2px;color:var(--ink-2);font-size:.8rem}
h1{margin:0;font-size:clamp(1.35rem,3.2vw,1.85rem);font-weight:650;letter-spacing:-.02em;line-height:1.15;text-wrap:balance}
h2{margin:0;font-size:1.2rem;font-weight:650;letter-spacing:-.01em}
.counts{margin:4px 0 0;color:var(--ink-2);font-size:.9rem;font-variant-numeric:tabular-nums}
.go{display:flex;flex-wrap:wrap;gap:8px}
.btn{display:inline-flex;align-items:center;gap:9px;height:44px;padding:0 20px 0 16px;border-radius:999px;text-decoration:none;font-weight:600;font-size:.95rem;
background:linear-gradient(180deg,rgba(255,255,255,.14),rgba(255,255,255,.03) 55%,rgba(255,255,255,.06)),rgba(22,24,29,.5);
box-shadow:inset 0 1px 0 rgba(255,255,255,.6),inset 0 -1px 0 rgba(255,255,255,.16),inset 0 0 0 1px rgba(255,255,255,.12),var(--drop);
transition:transform 150ms var(--ease),background-color 200ms var(--ease),box-shadow 200ms var(--ease)}
.btn:hover{background-color:rgba(40,43,50,.6)}.btn:active{transform:scale(.97)}
.btn svg{width:18px;height:18px;fill:currentColor;flex:none}
.btn.primary{color:#1d1606;background:linear-gradient(180deg,rgba(255,236,190,.55),rgba(255,200,87,0) 60%),rgba(255,200,87,.92);
box-shadow:inset 0 1px 0 rgba(255,246,222,.9),inset 0 -1px 0 rgba(120,80,0,.25),0 0 22px -4px rgba(255,200,87,.55),var(--drop)}
.btn.primary:hover{box-shadow:inset 0 1px 0 rgba(255,246,222,.9),inset 0 -1px 0 rgba(120,80,0,.25),0 0 30px -2px rgba(255,200,87,.7),var(--drop)}
.btn[hidden]{display:none}
.wrap{max-width:1200px;margin:0 auto;padding:20px clamp(12px,3vw,24px) 64px;display:grid;gap:36px}
section>h2{margin-bottom:4px}section>.muted{margin:0 0 14px;max-width:75ch}
.viewer{position:relative;border-radius:24px;overflow:hidden;height:min(70vh,640px);min-height:360px;background:#0b0b0c;scroll-margin-top:120px;
box-shadow:inset 0 0 0 1px rgba(255,255,255,.08),var(--drop)}
#gl{width:100%;height:100%;display:block}
.tools{position:absolute;left:12px;top:12px;display:flex;flex-wrap:wrap;gap:4px;padding:4px;border-radius:999px}
.tools button{appearance:none;border:0;font:inherit;font-size:.82rem;font-weight:500;color:var(--ink);background:transparent;padding:6px 13px;border-radius:999px;cursor:pointer;
transition:background-color 200ms var(--ease),color 200ms var(--ease)}
.tools button:hover{background:var(--fill-1)}
.tools button.on{background:linear-gradient(180deg,#fff,#e6e9ee);color:#111317;box-shadow:0 2px 6px rgba(0,0,0,.35)}
#objs{margin-left:10px;display:inline-flex;align-items:center;gap:7px}#objs.on{background:var(--fill-2);color:var(--ink);box-shadow:none}
#objs::before{content:"";width:7px;height:7px;border-radius:50%;background:rgba(255,255,255,.3)}
#objs.on::before{background:var(--amber);box-shadow:0 0 0 2px rgba(255,200,87,.22),0 0 9px 1px rgba(255,200,87,.8)}
.legend{position:absolute;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:6px;font-size:.78rem}
.legend span{display:inline-flex;align-items:center;gap:7px;padding:4px 11px 4px 9px;border-radius:999px}
.legend i{width:8px;height:8px;border-radius:50%;display:inline-block;box-shadow:0 0 8px currentColor}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card,.panel{background:var(--glass-sheen),rgba(255,255,255,.035);border-radius:22px;box-shadow:var(--rim),0 10px 30px -18px rgba(0,0,0,.8)}
.card{padding:14px;display:grid;gap:10px;align-content:start;cursor:pointer;transition:background-color 200ms var(--ease),box-shadow 200ms var(--ease)}
.card:hover{background-color:rgba(255,255,255,.06);box-shadow:var(--rim),inset 0 0 0 1px rgba(255,255,255,.1),0 14px 34px -18px rgba(0,0,0,.9)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.name{font-weight:650}.cid{margin-left:auto;color:var(--ink-2);font-size:.8rem;font-variant-numeric:tabular-nums}
.chip{font-size:.75rem;font-weight:650;padding:2px 10px 2px 9px;border-radius:999px;display:inline-flex;align-items:center;gap:6px;
background:color-mix(in srgb,currentColor 14%,transparent);box-shadow:inset 0 0 0 1px color-mix(in srgb,currentColor 35%,transparent)}
.chip::before{content:"";width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 8px currentColor}
.c-moved{color:var(--moved)}.c-added{color:var(--added)}.c-removed{color:var(--removed)}.c-unsure{color:var(--unsure)}
.sent{margin:0}.ev{margin:0;font-size:.8rem;color:var(--ink-2);font-variant-numeric:tabular-nums}
.thumbs{display:grid;grid-template-columns:1fr 1fr;gap:8px}.thumbs figure{margin:0}
.thumbs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:14px;background:rgba(255,255,255,.05);display:block}
.thumbs figcaption{font-size:.75rem;color:var(--ink-2);padding:4px 2px 0}
.prev{width:100%;border-radius:14px;background:#0b0b0c;display:block}
.two{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:14px}@media (max-width:800px){.two{grid-template-columns:minmax(0,1fr)}}
.panel{padding:18px 20px;display:grid;gap:10px;align-content:start}.panel p{margin:0}.panel h2:not(:first-child){margin-top:14px}
.panel img{max-width:100%;image-rendering:pixelated;border-radius:12px}
ul{margin:0;padding:0;list-style:none}li{margin:4px 0}li .chip{margin-right:4px}.muted{color:var(--ink-2)}
code{font-family:var(--mono);font-size:.85em;padding:1px 5px;border-radius:6px;background:var(--fill-1)}
table{border-collapse:collapse;font-size:.78rem;width:100%}td{border-bottom:1px solid rgba(255,255,255,.08);padding:5px 6px;vertical-align:top;overflow-wrap:anywhere}
td+td{font-family:var(--mono);color:var(--ink-2)}
details summary{cursor:pointer;font-weight:600;padding:4px 0}
@media (max-width:700px){.top{position:static;padding:14px 16px}.go,.btn{width:100%}.btn{justify-content:center}.viewer{height:62vh}}
@media (prefers-reduced-motion:reduce){*{transition-duration:0s!important}}
</style></head><body>
<header class="top glass">
<div><p class="mark">whatchanged report</p><h1>__TITLE__</h1><p class="counts">__COUNTS__</p></div>
<nav class="go" aria-label="Other views of this result">
<a id="walk" class="btn primary" href="index.html" hidden><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2.6 3.4 7.3v9.4L12 21.4l8.6-4.7V7.3L12 2.6Zm0 2.1 6.2 3.4L12 11.5 5.8 8.1 12 4.7Zm-6.7 5 5.8 3.2v6.2l-5.8-3.2V9.7Zm7.6 9.4v-6.2l5.8-3.2v6.2l-5.8 3.2Z"/></svg>Walk through the room in 3D</a>
<a id="film" class="btn" href="whatchanged.mp4" hidden><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7.5 5.2v13.6c0 .9 1 1.46 1.76.98l10.7-6.8a1.16 1.16 0 0 0 0-1.96L9.26 4.22c-.77-.48-1.76.07-1.76.98z"/></svg>Watch the video</a>
</nav>
</header>
<main class="wrap">
<section class="viewer"><canvas id="gl"></canvas>
<div class="tools glass"><button data-show="A">Before (A)</button><button data-show="B">After (B)</button><button data-show="AB" class="on">Both</button><button id="objs" class="on">All objects</button><button id="tour">Reset view</button></div>
<div class="legend"><span class="glass c-moved"><i style="background:var(--moved)"></i>moved</span><span class="glass c-added"><i style="background:var(--added)"></i>added</span><span class="glass c-removed"><i style="background:var(--removed)"></i>removed</span><span class="glass c-unsure"><i style="background:var(--unsure)"></i>can't tell</span></div></section>
<section><h2>Changes</h2><p class="muted">Click a card to fly to it. Every number in a sentence comes from the 3D measurements.</p><div class="grid">__CARDS__</div></section>
<section><h2>Where to look next</h2><p class="muted">For each spot the second video could not check, the best place to stand to settle it, judged against the current obstacles. The preview is rendered from the 3D model.</p><div class="grid">__LOOK__</div></section>
<section class="two"><div class="panel"><h2>Robot floor map update</h2><img src="__MAP__" alt="floor map: green newly blocked, orange newly free, hatched stale"><p class="muted">Green: newly blocked (__BLOCKED__ m²). Orange: newly free (__FREED__ m²). Hatched: not seen again, kept from before and marked stale (__STALE__ m²). Exported as Nav2 map_server maps in <code>maps/</code>.</p></div>
<div class="panel"><h2>Can't tell</h2><p class="muted">Space the second video could not check. These are reported, not guessed.</p><ul>__REGIONS__</ul>
<h2>How sure is the 3D?</h2><p class="muted">Static surfaces from A and B line up to __ALIGN__.</p>
<details><summary>Diagnostics</summary><table>__DIAG__</table></details></div></section>
</main>
<script>
// The 3D walkthrough and the video, when they sit next to this report. Over http(s) a HEAD request checks for each;
// from a local file only the video can be checked (the walkthrough needs a web server to load its splats anyway).
for (const [id, href] of [["walk", "index.html"], ["film", "whatchanged.mp4"]]) {
  const show = () => { document.getElementById(id).hidden = false; };
  if (location.protocol.startsWith("http")) fetch(href, { method: "HEAD" }).then((r) => { if (r.ok) show(); }).catch(() => {});
  else if (id === "film") Object.assign(document.createElement("video"), { preload: "metadata", muted: true, onloadedmetadata: show, src: href });
}
</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0b0c);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100); camera.up.set(0,0,1);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
function dec(b64, T){ const s=atob(b64), u=new Uint8Array(s.length); for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i); return new T(u.buffer); }
function cloud(c, tint){ const g=new THREE.BufferGeometry(); const p=dec(c.p, Float32Array); const col=dec(c.c, Uint8Array);
  const f=new Float32Array(col.length); for(let i=0;i<col.length;i++) f[i]=col[i]/255*(1-tint[3])+tint[i%3]*tint[3];
  g.setAttribute('position', new THREE.BufferAttribute(p,3)); g.setAttribute('color', new THREE.BufferAttribute(f,3));
  return new THREE.Points(g, new THREE.PointsMaterial({size:0.012, vertexColors:true})); }
const ptsA = cloud(D.A,[0,0,0,0]), ptsB = cloud(D.B,[0,0,0,0]); scene.add(ptsA, ptsB);
const grid = new THREE.GridHelper(10, 20, 0x34373e, 0x1d1f24); grid.rotation.x = Math.PI/2; scene.add(grid);
function box(c, colour, opacity, dashed){ const geo=new THREE.BoxGeometry(c.size_m[0], c.size_m[1], c.size_m[2]);
  const edges=new THREE.EdgesGeometry(geo);
  const mat = dashed ? new THREE.LineDashedMaterial({color:colour, dashSize:0.04, gapSize:0.03}) : new THREE.LineBasicMaterial({color:colour});
  const line=new THREE.LineSegments(edges, mat); if(dashed) line.computeLineDistances();
  const fill=new THREE.Mesh(geo, new THREE.MeshBasicMaterial({color:colour, transparent:true, opacity, depthWrite:false}));
  const g=new THREE.Group(); g.add(line, fill); g.position.set(...c.centre_m); g.rotation.z = c.yaw_deg*Math.PI/180; return g; }
const targets = {};
for (const c of D.changes){ const col = D.colours[c.type];
  if (c.type==='moved'){ scene.add(box(c.before, col, 0.06, true)); const from=new THREE.Vector3(...c.before.centre_m), to=new THREE.Vector3(...c.centre_m);
    const dir=to.clone().sub(from); const len=dir.length(); scene.add(new THREE.ArrowHelper(dir.normalize(), from, len, col, 0.12, 0.07)); }
  scene.add(box(c, col, c.type==='removed'?0.12:0.22, c.type==='removed'));
  targets[c.id] = new THREE.Vector3(...c.centre_m); }
for (const r of D.regions.slice(0,6)){ const size=r.max.map((v,i)=>v-r.min[i]), centre=r.max.map((v,i)=>(v+r.min[i])/2);
  scene.add(box({size_m:size, centre_m:centre, yaw_deg:0}, D.colours.unsure, 0.05, true)); }
for (const v of D.views){ const cam=new THREE.Vector3(...v.cam), tgt=new THREE.Vector3(...v.target);
  const dir=tgt.clone().sub(cam); const len=dir.length();
  scene.add(new THREE.ArrowHelper(dir.normalize(), cam, Math.min(len,0.6), 0xffc857, 0.15, 0.1));
  const s=new THREE.Mesh(new THREE.SphereGeometry(0.06,16,12), new THREE.MeshBasicMaterial({color:0xffc857})); s.position.copy(cam); scene.add(s);
  targets[v.id]=cam; }
const objGroup = new THREE.Group(); scene.add(objGroup);
for (const [sess, list] of Object.entries(D.objects||{})) for (const o of list){
  const geo=new THREE.BoxGeometry(...o.size_m); const line=new THREE.LineSegments(new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({color: sess==='A'?0x9aa0a6:0xc9ced3, transparent:true, opacity:0.55}));
  line.position.set(...o.centre_m); line.rotation.z=o.yaw_deg*Math.PI/180; line.userData.sess=sess; objGroup.add(line);
  const cv=document.createElement('canvas'); cv.width=256; cv.height=48; const g=cv.getContext('2d'); g.fillStyle='rgba(22,24,29,.8)'; g.beginPath(); g.roundRect(0,0,256,48,24); g.fill();
  g.fillStyle='#f6f7f9'; g.font='500 26px Geist, -apple-system, sans-serif'; g.fillText(o.label,16,33);
  const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(cv), depthTest:false, transparent:true}));
  sp.scale.set(0.5,0.094,1); sp.position.set(o.centre_m[0], o.centre_m[1], o.centre_m[2]+o.size_m[2]/2+0.08); sp.userData.sess=sess; objGroup.add(sp); }
document.getElementById('objs').addEventListener('click', e=>{ objGroup.visible=!objGroup.visible; e.target.classList.toggle('on', objGroup.visible); });
const all = [...D.camsA, ...D.camsB]; const cen = new THREE.Vector3();
const bb = new THREE.Box3().setFromObject(ptsA); bb.getCenter(cen);
function home(){ const sz=bb.getSize(new THREE.Vector3()).length(); camera.position.copy(cen).add(new THREE.Vector3(-0.35*sz, -0.55*sz, 0.6*sz)); controls.target.copy(cen); }
home();
for (const cp of [[D.camsA,0x9aa7b1],[D.camsB,0x5a6773]]){ const g=new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cp[0].flat()),3));
  scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({color:cp[1]}))); }
let fly=null;
function flyTo(p){ const off=camera.position.clone().sub(controls.target).normalize().multiplyScalar(1.6); fly={t0:performance.now(), from:[camera.position.clone(), controls.target.clone()], to:[p.clone().add(off), p.clone()]}; }
document.querySelectorAll('.card').forEach(el=>{ const go=()=>{ if(targets[el.dataset.id]){ flyTo(targets[el.dataset.id]); document.querySelector('.viewer').scrollIntoView({behavior:'smooth'}); } };
  el.addEventListener('click', go); el.addEventListener('keydown', e=>{ if(e.key==='Enter') go(); }); });
document.querySelectorAll('[data-show]').forEach(b=>b.addEventListener('click',()=>{ document.querySelectorAll('[data-show]').forEach(x=>x.classList.toggle('on', x===b));
  ptsA.visible = b.dataset.show.includes('A'); ptsB.visible = b.dataset.show.includes('B');
  objGroup.children.forEach(o=>{ o.visible = b.dataset.show.includes(o.userData.sess); }); }));
document.getElementById('tour').addEventListener('click', home);
function resize(){ const w=canvas.clientWidth, h=canvas.clientHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
(function loop(now){ requestAnimationFrame(loop);
  if(fly){ const t=Math.min(1,(now-fly.t0)/900), e=t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2; camera.position.lerpVectors(fly.from[0], fly.to[0], e); controls.target.lerpVectors(fly.from[1], fly.to[1], e); if(t>=1) fly=null; }
  controls.update(); renderer.render(scene, camera); })(performance.now());
</script></body></html>
"""
