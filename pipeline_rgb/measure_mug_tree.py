"""Measured-shape model of the mug tree (the tape task's target), from hand-picked image points in
the two calibrated views of the RGB-only capture (episode 1, frame 0).

  post      vertical cylinder: axis from the triangulated top and bottom, radius from its image width
  base      disc around the post, radius/thickness fitted to the base ellipse's extent in both views
  branches  straight dowels: tip triangulated from both views (correspondences checked on the
            epipolar lines, < 1 px except B at 2 px), root on the post at the height where the root
            pixel's ray meets the post axis (mean of the two views)

Writes outputs/ur_tape/scene.json (base frame, metres) and an overlay of the projected model on both
views for checking.

    python scripts/ur_tape/measure_mug_tree.py
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams, triangulate  # noqa: E402

VIDEOS = Path(os.environ.get("UR_RGB_VIDEOS", Path.home() / "Downloads/ur_human_capture/videos"))

# (front px, right px), episode 1 frame 0
POST_TOP = ((778, 267), (630, 335))
POST_BOTTOM = ((760, 470), (632, 585))
# base ellipse extent: x min, x max, y min, y max (px)
BASE_EXTENT = ((711.7, 818.3, 438, 507), (562.5, 699, 550, 646))
POST_WIDTH_PX = (15, 15)       # mid-height, front / right
BASE_EDGE_PX = 8               # height of the base's rim at its front edge, front view
BRANCH_WIDTH_PX = 6
BRANCHES = {  # name: (tip front, tip right, root height on the axis above the base frame origin)
    "A": ((727.3, 254.3), (663.3, 310.7), 0.294),
    "B": ((840.3, 270.7), (596.7, 368.3), 0.283),   # the tape hangs here
    "C": ((792.7, 305.7), (697.3, 434.0), 0.188),
    "D": ((764.3, 354.3), (567.7, 405.0), 0.193),
    "E": ((725.0, 377.0), (657.3, 458.3), 0.118),
}


def cylinder_outline(c0, c1, r, n=64):
    """Points on the two rims of a cylinder from c0 to c1 (for projection extents / overlays)."""
    a = (c1 - c0) / np.linalg.norm(c1 - c0)
    u = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    return c0 + ring, c1 + ring


def main(output: Path = Path("outputs/ur_tape/scene.json")) -> None:
    cams = load_cams()
    table_z = json.loads(Path("outputs/ur_cube/scene_layout.json").read_text())["table_z"]
    top, e_top = triangulate(cams, POST_TOP)
    bot, e_bot = triangulate(cams, POST_BOTTOM)
    axis_xy = (top[:2] + bot[:2]) / 2
    depth = [cams[i].project(np.r_[axis_xy, 0.15])[1] for i in range(2)]
    r_post = float(np.mean([w * d / cams[i].K[0, 0] / 2 for i, (w, d) in enumerate(zip(POST_WIDTH_PX, depth))]))
    r_branch = float(BRANCH_WIDTH_PX * np.mean(depth) / cams[0].K[0, 0] / 2)

    t_base = BASE_EDGE_PX * depth[0] / cams[0].K[0, 0]

    def base_res(x):
        r = x[0]
        t = t_base
        lo, hi = cylinder_outline(np.r_[axis_xy, table_z], np.r_[axis_xy, table_z + t], r)
        res = []
        for cam, ext in zip(cams, BASE_EXTENT):
            uv = cam.project(np.vstack([lo, hi]))[0]
            res += [uv[:, 0].min() - ext[0], uv[:, 0].max() - ext[1], uv[:, 1].min() - ext[2], uv[:, 1].max() - ext[3]]
        return res
    fit = least_squares(base_res, [0.08], bounds=([0.03], [0.15]))
    r_base = fit.x[0]

    branches = {}
    for k, (u0, u1, h) in BRANCHES.items():
        tip, err = triangulate(cams, [u0, u1])
        d = tip[:2] - axis_xy
        d /= np.linalg.norm(d)
        root = np.r_[axis_xy + d * r_post * 0.5, h]  # dowel starts inside the post
        L = np.linalg.norm(tip - root)
        elev = np.degrees(np.arcsin((tip[2] - root[2]) / L))
        branches[k] = dict(root=root.round(4).tolist(), tip=tip.round(4).tolist(), radius=round(r_branch, 4),
                           length=round(float(L), 4), elevation_deg=round(float(elev), 1),
                           azimuth_deg=round(float(np.degrees(np.arctan2(d[1], d[0]))), 1),
                           tip_reproj_px=round(float(err), 2))
        print(k, branches[k])
    scene = dict(
        table_z=table_z,
        mug_tree=dict(
            post=dict(xy=axis_xy.round(4).tolist(), z0=table_z, z1=float(top[2]), radius=round(r_post, 4),
                      reproj_px=[round(e_top, 2), round(e_bot, 2)]),
            base=dict(xy=axis_xy.round(4).tolist(), radius=round(float(r_base), 4), thickness=round(float(t_base), 4),
                      extent_residual_px=round(float(np.abs(fit.fun).max()), 1)),
            branches=branches, target_branch="B"),
        source="hand-picked points in the front and right RealSense views, RGB-only capture ep 1 frame 0",
    )
    print("post", scene["mug_tree"]["post"], "\nbase", scene["mug_tree"]["base"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, indent=1))

    tiles = []
    for i, cam in enumerate(cams):
        cap = cv2.VideoCapture(str(VIDEOS / "1" / f"{i}.mp4"))
        im = cap.read()[1]
        def draw(c0, c1, r, col):
            lo, hi = cylinder_outline(np.asarray(c0, float), np.asarray(c1, float), r, 48)
            for ring in (lo, hi):
                cv2.polylines(im, [cam.project(ring)[0].astype(np.int32)], True, col, 1)
            for j in (0, 24):
                p = cam.project(np.stack([lo[j], hi[j]]))[0].astype(int)
                cv2.line(im, tuple(p[0]), tuple(p[1]), col, 1)
        draw([*axis_xy, table_z], [*axis_xy, table_z + t_base], r_base, (255, 0, 255))
        draw([*axis_xy, table_z], [*axis_xy, top[2]], r_post, (0, 255, 255))
        for k, b in branches.items():
            draw(b["root"], b["tip"], b["radius"], (0, 0, 255) if k == "B" else (0, 255, 0))
        c = cam.project(np.r_[axis_xy, 0.15])[0].astype(int)
        crop = im[max(c[1] - 230, 0):c[1] + 170, c[0] - 150:c[0] + 150]
        tiles.append(cv2.resize(crop, (450, 600)))
    cv2.imwrite(str(output.parent / "mug_tree_overlay.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    tyro.cli(main)
