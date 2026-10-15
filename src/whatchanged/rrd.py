"""Write a Rerun recording (.rrd): both scans, every camera with its frame, change boxes, move arrows.

Open with `rerun out/scene.rrd`, or in a browser at
https://app.rerun.io/version/<rerun version>/index.html?url=<public URL of the .rrd>.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

COLOURS = {"moved": (0, 114, 178), "added": (0, 158, 115), "removed": (213, 94, 0), "unsure": (230, 159, 0)}


def write(path: Path, A, B, changes: list[dict], plans: list[dict], max_points: int = 150_000, boxes: dict | None = None) -> Path:
    import rerun as rr

    rr.init("whatchanged", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rng = np.random.default_rng(0)
    for name, S, key in (("A", A, "before"), ("B", B, "after")):
        p, c = S.points(conf_pct=30, stride=2)
        if len(p) > max_points:
            i = rng.choice(len(p), max_points, replace=False)
            p, c = p[i], c[i]
        rr.log(f"world/{key}", rr.Points3D(p, colors=c, radii=0.005), static=True)
        H, W = S.depth.shape[1:]
        for i in range(len(S.depth)):
            if hasattr(rr, "set_time"):
                rr.set_time(f"frame_{name}", sequence=i)
            else:
                rr.set_time_sequence(f"frame_{name}", i)
            P = S.poses[i]
            rr.log(f"world/camera_{name}", rr.Transform3D(translation=P[:3, 3], mat3x3=P[:3, :3]))
            rr.log(f"world/camera_{name}", rr.Pinhole(image_from_camera=S.K[i], width=W, height=H,
                                                      camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.3))
            buf = io.BytesIO()
            Image.fromarray(S.images[i]).save(buf, format="JPEG", quality=80)
            rr.log(f"world/camera_{name}", rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"))
        rr.log(f"world/path_{name}", rr.LineStrips3D([S.poses[:, :3, 3]], colors=[(150, 160, 170)]), static=True)

    for c in changes:
        col = COLOURS[c["type"]]
        bx = [c] + ([c["before"]] if c["type"] == "moved" else [])
        rr.log(f"world/changes/{c['id']}", rr.Boxes3D(
            centers=[b["centre_m"] for b in bx],
            half_sizes=[np.array(b["size_m"]) / 2 for b in bx],
            rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], degrees=b["yaw_deg"]) for b in bx],
            colors=[col] * len(bx), labels=[f"{c['type']}: {c['label']}"] + (["(was here)"] if len(bx) > 1 else [])),
            static=True)
        if c["type"] == "moved":
            o = np.array(c["before"]["centre_m"])
            rr.log(f"world/changes/{c['id']}/arrow", rr.Arrows3D(origins=[o], vectors=[np.array(c["centre_m"]) - o], colors=[col]), static=True)
    for name, key in (("A", "before"), ("B", "after")):
        for_session = (boxes or {}).get(name) or []
        if for_session:
            rr.log(f"world/objects_{key}", rr.Boxes3D(
                centers=[b.centre for b in for_session], half_sizes=[b.size / 2 for b in for_session],
                rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], radians=b.yaw) for b in for_session],
                colors=[(160, 166, 172)] * len(for_session), labels=[b.label for b in for_session]), static=True)
    for p in plans:
        cam = np.array(p["stand_at_m"] + [p["camera_height_m"]])
        rr.log(f"world/look_next/{p['id']}", rr.Arrows3D(origins=[cam], vectors=[(np.array(p["region_centre_m"]) - cam) * 0.5],
                                                          colors=[COLOURS["unsure"]], labels=["look here"]), static=True)
    lines = [f"- **{c['type']}** {c['label']}: {c['sentence']}" for c in changes] or ["No changes found."]
    rr.log("summary", rr.TextDocument("\n".join(lines), media_type=rr.MediaType.MARKDOWN), static=True)
    return path
