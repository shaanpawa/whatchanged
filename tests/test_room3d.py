"""The photoreal-room helpers that need no video, COLMAP or GPU: pairs, file formats, levelling."""

import gzip
import struct

import numpy as np

from whatchanged.room3d import (cam_to_world, match_pairs, read_images, read_splat_ply, viewer_frame,
                                write_images, write_spz)


def test_pairs_link_neighbours_loops_and_videos():
    a = [f"A/A_{i:04d}.jpg" for i in range(40)]
    b = [f"B/B_{i:04d}.jpg" for i in range(40)]
    pairs = set(match_pairs([a, b], window=5, stride=4))
    assert (a[0], a[5]) in pairs and (a[0], a[6]) not in pairs   # time neighbours only
    assert (a[0], a[36]) in pairs                                 # the walk's end meets its start
    assert (a[0], b[0]) in pairs and (a[4], b[8]) in pairs        # the videos are tied together
    assert all(x < y for x, y in pairs)


def test_images_bin_round_trip(tmp_path):
    imgs = {1: {"name": "room/room_0000.jpg", "camera_id": 1, "q": (1.0, 0.0, 0.0, 0.0), "t": (0.5, -1.0, 2.0)},
            7: {"name": "room/room_0001.jpg", "camera_id": 1, "q": (0.5, 0.5, 0.5, 0.5), "t": (0.0, 0.0, 0.0)}}
    write_images(tmp_path / "images.bin", imgs)
    assert read_images(tmp_path / "images.bin") == imgs
    assert np.allclose(cam_to_world(imgs[1])[:3, 3], [-0.5, 1.0, -2.0])


def _splats(n=50, rest=45, seed=0):
    rng = np.random.default_rng(seed)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(rest)],
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    g = np.zeros(n, dtype=[(k, "<f4") for k in names])
    for k in names:
        g[k] = rng.normal(size=n)
    return g


def test_spz_header_and_size(tmp_path):
    g = _splats()
    write_spz(tmp_path / "a.spz", g, sh_degree=3)
    raw = gzip.decompress((tmp_path / "a.spz").read_bytes())
    magic, version, n, deg, frac, _, _ = struct.unpack("<IIIBBBB", raw[:16])
    assert (magic, version, n, deg) == (0x5053474E, 2, 50, 3)
    assert len(raw) == 16 + n * (9 + 1 + 3 + 3 + 3 + 45)
    pos = np.frombuffer(raw[16:16 + 9 * n], dtype=np.uint8).reshape(n, 3, 3).astype(np.int32)
    fixed = pos[..., 0] | pos[..., 1] << 8 | pos[..., 2] << 16
    fixed = np.where(fixed >= 1 << 23, fixed - (1 << 24), fixed)
    assert np.allclose(fixed / (1 << frac), np.stack([g["x"], g["y"], g["z"]], 1), atol=2 ** -frac)
    write_spz(tmp_path / "b.spz", g, sh_degree=0)
    assert len(gzip.decompress((tmp_path / "b.spz").read_bytes())) == 16 + n * 19


def test_ply_reader(tmp_path):
    g = _splats(rest=0)
    head = "ply\nformat binary_little_endian 1.0\nelement vertex 50\n" + "".join(
        f"property float {k}\n" for k in g.dtype.names) + "end_header\n"
    (tmp_path / "s.ply").write_bytes(head.encode() + g.tobytes())
    back = read_splat_ply(tmp_path / "s.ply")
    assert np.array_equal(back["opacity"], g["opacity"])


def test_viewer_frame_levels_and_scales():
    # COLMAP-like world: tilted, scaled by 0.3, floor 1.5 m below cameras that walk a circle.
    rng = np.random.default_rng(1)
    ang = np.radians(25)
    tilt = np.array([[1, 0, 0], [0, np.cos(ang), -np.sin(ang)], [0, np.sin(ang), np.cos(ang)]])
    s = 0.3
    floor = np.c_[rng.uniform(-2, 2, 3000), np.zeros(3000), rng.uniform(-2, 2, 3000)]
    wall = np.c_[np.full(800, 2.0), rng.uniform(0, 2.5, 800), rng.uniform(-2, 2, 800)]
    pts = (np.r_[floor, wall] @ tilt.T) * s
    cams = []
    for a in np.linspace(0, 2 * np.pi, 30, endpoint=False):
        c2w = np.eye(4)
        fwd = np.array([np.cos(a), 0, np.sin(a)])
        down = np.array([0, -1.0, 0])
        c2w[:3, :3] = np.stack([np.cross(down, fwd), down, fwd], 1)   # COLMAP camera: x right, y down, z forward
        c2w[:3, 3] = [0.5 * np.cos(a), 1.5, 0.5 * np.sin(a)]
        c2w[:3, :3] = tilt @ c2w[:3, :3]
        c2w[:3, 3] = tilt @ c2w[:3, 3] * s
        cams.append(c2w)
    f = viewer_frame(np.stack(cams), pts, camera_height=1.5)
    M = f["matrix"]
    p = pts @ M[:3, :3].T + M[:3, 3]
    assert abs(np.median(p[:3000, 1])) < 0.02            # floor at y = 0
    assert np.ptp(p[:3000, 1]) < 0.05                    # and level
    assert abs(f["scale"] * s - 1.0) < 0.03              # back to metres
