"""One self-contained report.html: 3D viewer, change cards, floor map, diagnostics."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

TYPE_COLOURS = {"moved": "#0072b2", "added": "#009e73", "removed": "#d55e00", "unsure": "#e69f00"}


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
        ev = c["evidence"]
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
        <p class="ev">B saw it free in {ev['free_views']} views · free share {ev['free_share']:.0%} · {ev['noise_margin']:.1f}× past the noise tolerance{extra}{' · ' + times if times else ''}</p>
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
    diag_rows = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(json.dumps(v))}</td></tr>" for k, v in diag.items())

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
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,700&family=Atkinson+Hyperlegible:wght@400;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{--paper:#f3f5f6;--panel:#fff;--ink:#15202a;--muted:#5a6773;--rule:#d6dde2;--soft:#e7ecef;
--moved:#0067a3;--added:#007a5c;--removed:#b9480a;--unsure:#9a6a00;
--display:"Bricolage Grotesque","Avenir Next",system-ui,sans-serif;--body:"Atkinson Hyperlegible","Helvetica Neue",Arial,sans-serif;--mono:"JetBrains Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--paper:#0e1317;--panel:#151c21;--ink:#e3e9ed;--muted:#98a5af;--rule:#28333c;--soft:#1c252c;--moved:#63b0ea;--added:#3fd0a4;--removed:#f28b55;--unsure:#ebb84a;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);line-height:1.5}
.wrap{max-width:1200px;margin:0 auto;padding:24px clamp(16px,3vw,32px) 64px;display:grid;gap:24px}
h1{font-family:var(--display);font-size:clamp(1.8rem,5vw,2.6rem);margin:0;letter-spacing:-.02em;text-wrap:balance}
h2{font-family:var(--display);font-size:1.4rem;margin:0}
.eyebrow{font-family:var(--mono);font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:0}
.counts{font-family:var(--mono);color:var(--muted);margin:0}
.viewer{position:relative;background:#0f1418;border-radius:12px;overflow:hidden;height:min(70vh,640px);min-height:360px}
#gl{width:100%;height:100%;display:block}
.tools{position:absolute;left:12px;top:12px;display:flex;flex-wrap:wrap;gap:6px}
.tools button{font-family:var(--mono);font-size:.75rem;border:1px solid #3a4650;background:#1b242b;color:#e3e9ed;padding:4px 10px;border-radius:6px;cursor:pointer}
.tools button.on{background:#e3e9ed;color:#15202a}
.legend{position:absolute;right:12px;bottom:12px;display:flex;flex-wrap:wrap;gap:6px;font-family:var(--mono);font-size:.72rem;color:#e3e9ed}
.legend span{display:inline-flex;align-items:center;gap:5px;background:#1b242bcc;padding:2px 8px;border-radius:999px}
.legend i{width:9px;height:9px;border-radius:50%;display:inline-block}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:12px;display:grid;gap:8px;cursor:pointer}
.card:focus-visible,.card:hover{outline:2px solid var(--moved);outline-offset:1px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.name{font-weight:700}.cid{margin-left:auto;font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.chip{font-family:var(--mono);font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding:1px 8px;border-radius:999px;border:1px solid currentColor}
.c-moved{color:var(--moved)}.c-added{color:var(--added)}.c-removed{color:var(--removed)}.c-unsure{color:var(--unsure)}
.sent{margin:0}.ev{margin:0;font-family:var(--mono);font-size:.72rem;color:var(--muted)}
.thumbs{display:grid;grid-template-columns:1fr 1fr;gap:6px}.thumbs figure{margin:0}.thumbs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;background:var(--soft)}
.prev{width:100%;border-radius:6px;background:#0f1418}
.thumbs figcaption{font-family:var(--mono);font-size:.68rem;color:var(--muted)}
.two{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px}@media (max-width:800px){.two{grid-template-columns:minmax(0,1fr)}}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;display:grid;gap:10px;align-content:start}
.panel img{max-width:100%;image-rendering:pixelated;border-radius:6px}
ul{margin:0;padding-left:1.1em}.muted{color:var(--muted)}
table{border-collapse:collapse;font-size:.8rem;width:100%}td{border-bottom:1px solid var(--rule);padding:4px 6px;vertical-align:top;font-family:var(--mono);overflow-wrap:anywhere}
details summary{cursor:pointer;font-weight:700}
</style></head><body><div class="wrap">
<header><p class="eyebrow">whatchanged · report</p><h1>__TITLE__</h1><p class="counts">__COUNTS__</p></header>
<section class="viewer"><canvas id="gl"></canvas>
<div class="tools"><button data-show="A">Before (A)</button><button data-show="B">After (B)</button><button data-show="AB" class="on">Both</button><button id="objs" class="on">All objects</button><button id="tour">Reset view</button></div>
<div class="legend"><span><i style="background:#0072b2"></i>moved</span><span><i style="background:#009e73"></i>added</span><span><i style="background:#d55e00"></i>removed</span><span><i style="background:#e69f00"></i>can't tell</span></div></section>
<section><h2>Changes</h2><p class="muted">Click a card to fly to it. Every number in a sentence comes from the 3D measurements.</p><div class="grid">__CARDS__</div></section>
<section><h2>Where to look next</h2><p class="muted">For each spot the second video could not check, the best place to stand to settle it, judged against the current obstacles. The preview is rendered from the 3D model.</p><div class="grid">__LOOK__</div></section>
<section class="two"><div class="panel"><h2>Robot floor map update</h2><img src="__MAP__" alt="floor map: green newly blocked, orange newly free, hatched stale"><p class="muted">Green: newly blocked (__BLOCKED__ m²). Orange: newly free (__FREED__ m²). Hatched: not seen again, kept from before and marked stale (__STALE__ m²). Exported as Nav2 map_server maps in <code>maps/</code>.</p></div>
<div class="panel"><h2>Can't tell</h2><p class="muted">Space the second video could not check. These are reported, not guessed.</p><ul>__REGIONS__</ul>
<h2>How sure is the 3D?</h2><p class="muted">Static surfaces from A and B line up to __ALIGN__.</p>
<details><summary>Diagnostics</summary><table>__DIAG__</table></details></div></section>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0f1418);
const camera = new THREE.PerspectiveCamera(55, 1, 0.05, 100); camera.up.set(0,0,1);
const controls = new OrbitControls(camera, canvas); controls.enableDamping = true;
function dec(b64, T){ const s=atob(b64), u=new Uint8Array(s.length); for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i); return new T(u.buffer); }
function cloud(c, tint){ const g=new THREE.BufferGeometry(); const p=dec(c.p, Float32Array); const col=dec(c.c, Uint8Array);
  const f=new Float32Array(col.length); for(let i=0;i<col.length;i++) f[i]=col[i]/255*(1-tint[3])+tint[i%3]*tint[3];
  g.setAttribute('position', new THREE.BufferAttribute(p,3)); g.setAttribute('color', new THREE.BufferAttribute(f,3));
  return new THREE.Points(g, new THREE.PointsMaterial({size:0.012, vertexColors:true})); }
const ptsA = cloud(D.A,[0,0,0,0]), ptsB = cloud(D.B,[0,0,0,0]); scene.add(ptsA, ptsB);
const grid = new THREE.GridHelper(10, 20, 0x33404a, 0x222c33); grid.rotation.x = Math.PI/2; scene.add(grid);
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
  scene.add(new THREE.ArrowHelper(dir.normalize(), cam, Math.min(len,0.6), 0xe69f00, 0.15, 0.1));
  const s=new THREE.Mesh(new THREE.SphereGeometry(0.06,16,12), new THREE.MeshBasicMaterial({color:0xe69f00})); s.position.copy(cam); scene.add(s);
  targets[v.id]=cam; }
const objGroup = new THREE.Group(); scene.add(objGroup);
for (const [sess, list] of Object.entries(D.objects||{})) for (const o of list){
  const geo=new THREE.BoxGeometry(...o.size_m); const line=new THREE.LineSegments(new THREE.EdgesGeometry(geo),
    new THREE.LineBasicMaterial({color: sess==='A'?0x9aa0a6:0xc9ced3, transparent:true, opacity:0.55}));
  line.position.set(...o.centre_m); line.rotation.z=o.yaw_deg*Math.PI/180; line.userData.sess=sess; objGroup.add(line);
  const cv=document.createElement('canvas'); cv.width=256; cv.height=48; const g=cv.getContext('2d'); g.fillStyle='rgba(15,20,24,.75)'; g.fillRect(0,0,256,48);
  g.fillStyle='#e3e9ed'; g.font='26px monospace'; g.fillText(o.label,10,33);
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
