"""Measure the bench layout in the UR base frame from the RealSense RGB-D capture.

Finds the table plane, the two orange fixtures, and the black cube at the first and last frame
of each episode (both cameras fused). Writes <output> json, used to lay out the IsaacLab scene
and to sanity-check tracked object trajectories.

Example:
    python scripts/ur_cube/measure_scene_from_rgbd.py --output outputs/ur_cube/scene_layout.json
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro

from rgbd_utils import backproject, load_frame

WORKSPACE = {"x": (-0.4, 0.6), "y": (-1.1, -0.4)}  # around the fixtures, base frame (m)


def hsv(rgb: np.ndarray):
    c = rgb.astype(np.float32) / 255.0
    mx, mn = c.max(1), c.min(1)
    d = np.maximum(mx - mn, 1e-6)
    r, g, b = c.T
    hue = np.select([mx == r, mx == g], [((g - b) / d) % 6, (b - r) / d + 2], (r - g) / d + 4) * 60
    return hue, (mx - mn) / np.maximum(mx, 1e-6), mx


def fused_cloud(capture_dir: Path, episode: int, step: int):
    meta = load_frame(capture_dir, episode, step)
    pts, cols = zip(*[backproject(c, stride=1)[:2] for c in meta["cameras"].values()])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    keep = (
        (pts[:, 0] > WORKSPACE["x"][0]) & (pts[:, 0] < WORKSPACE["x"][1])
        & (pts[:, 1] > WORKSPACE["y"][0]) & (pts[:, 1] < WORKSPACE["y"][1])
    )
    return pts[keep], cols[keep]


def clusters(pts: np.ndarray, eps: float, min_points: int):
    labels = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).cluster_dbscan(eps, min_points))
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    return [pts[labels == i] for i in ids[np.argsort(-counts)]]


def main(
    output: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episodes: tuple[int, ...] = (0, 1),
) -> None:
    pts, cols = fused_cloud(capture_dir, episodes[0], 0)
    plane, inliers = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).segment_plane(0.004, 3, 2000)
    n = np.array(plane[:3]) / np.linalg.norm(plane[:3])
    table_z = float(np.median(pts[inliers][:, 2]))
    print(f"table plane normal {n.round(4)}, table z {table_z:.4f} m")

    # Fixtures: orange, a little above the table. The tape splits each one, so cluster loosely.
    hue, sat, val = hsv(cols)
    height = pts[:, 2] - table_z
    orange = (hue > 8) & (hue < 35) & (sat > 0.45) & (val > 0.3) & (height > 0.003)
    fixtures = clusters(pts[orange], eps=0.03, min_points=50)[:2]
    fixture_info = []
    for f in fixtures:
        c = np.median(f, axis=0)
        fixture_info.append({
            "center_xy": c[:2].tolist(),
            "top_z": float(np.percentile(f[:, 2], 95)),
            "extent_xy": (np.percentile(f[:, :2], 98, axis=0) - np.percentile(f[:, :2], 2, axis=0)).tolist(),
        })
    fixture_info.sort(key=lambda f: f["center_xy"][0])
    for i, f in enumerate(fixture_info):
        print(f"fixture {i}: center {np.round(f['center_xy'], 4)}, top z {f['top_z']:.4f}, extent {np.round(f['extent_xy'], 3)}")

    def find_cube(ep, step):
        p, c = fused_cloud(capture_dir, ep, step)
        _, _, v = hsv(c)
        h = p[:, 2] - table_z
        dark = (v < 0.22) & (h > 0.015) & (h < 0.12)
        cl = clusters(p[dark], eps=0.012, min_points=40)
        if not cl:
            return None
        cube = cl[0]
        return {"center": np.median(cube, axis=0).tolist(), "top_z": float(np.percentile(cube[:, 2], 95)), "num_points": len(cube)}

    cube_info = {}
    for ep in episodes:
        n_steps = len(list((capture_dir / "rgbd" / str(ep)).iterdir()))
        start, end = find_cube(ep, 0), find_cube(ep, n_steps - 1)
        cube_info[str(ep)] = {"start": start, "end": end, "num_steps": n_steps}
        for name, cinfo in (("start", start), ("end", end)):
            if cinfo is None:
                print(f"ep{ep} {name}: cube not found")
                continue
            near = int(np.argmin([np.linalg.norm(np.array(cinfo["center"][:2]) - f["center_xy"]) for f in fixture_info]))
            d = np.linalg.norm(np.array(cinfo["center"][:2]) - fixture_info[near]["center_xy"])
            cinfo["nearest_fixture"], cinfo["dist_to_fixture_m"] = near, float(d)
            print(f"ep{ep} {name}: cube center {np.round(cinfo['center'], 4)} (top z {cinfo['top_z']:.4f}), "
                  f"on fixture {near} ({d * 1000:.0f} mm from its center)")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"table_z": table_z, "table_normal": n.tolist(), "fixtures": fixture_info, "cube": cube_info}, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
