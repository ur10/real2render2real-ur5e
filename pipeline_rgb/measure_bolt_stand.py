"""Measured-shape model of the bolt stand (the nut task's target), RGB-only capture episode 4.

  axis      triangulated rod tip and spacer-stack top (front + right views)
  spacers   the green spacer stack: a vertical cylinder fitted to its green silhouette in both views
  rod       threaded rod from the spacers to the triangulated tip; radius from its image width
  base      grey nut under the spacers; radius from its image width, height from the right view
  block     wooden block under it, down to the bench; side from its image width, one face turned to
            the right camera as in the images
  target    the nut's resting pose on the rod (nut_dims.json, last frame)

    python scripts/ur_tape/measure_bolt_stand.py --output outputs/ur_nut/scene.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams, triangulate  # noqa: E402
from fit_tape_dims import read_frame  # noqa: E402
from tape_model import TapeDims, green_mask, render_labels, sample_surface  # noqa: E402

ROD_TIP = ((736.75, 387), (565, 425))        # (front px, right px), episode 4 frame 0
SPACER_TOP = ((736, 428.75), (570.8, 470))
ROD_WIDTH_PX = (13, 11)
BASE_WIDTH_PX = (33, 36)
BASE_HEIGHT_PX_RIGHT = 16.7
BLOCK_WIDTH_PX_RIGHT = 85


def main(output: Path = Path("outputs/ur_nut/scene.json"), nut_dims: Path = Path("outputs/ur_nut/nut_dims.json")) -> None:
    cams = load_cams()
    table_z = json.loads(Path("outputs/ur_cube/scene_layout.json").read_text())["table_z"]
    tip, e_tip = triangulate(cams, ROD_TIP)
    top, e_top = triangulate(cams, SPACER_TOP)
    xy = (tip[:2] + top[:2]) / 2
    depth = [cam.project(np.r_[xy, 0.08])[1] for cam in cams]
    m_per_px = [d / cam.K[0, 0] for d, cam in zip(depth, cams)]
    r_rod = float(np.mean([w * s / 2 for w, s in zip(ROD_WIDTH_PX, m_per_px)]))
    r_base = float(np.mean([w * s / 2 for w, s in zip(BASE_WIDTH_PX, m_per_px)]))
    h_base = BASE_HEIGHT_PX_RIGHT * m_per_px[1]
    block = BLOCK_WIDTH_PX_RIGHT * m_per_px[1]

    # spacer stack: vertical cylinder on the axis, fitted to the green silhouette in frame 0
    ims = [read_frame(4, i, 0) for i in range(2)]
    masks = [green_mask(im) for im in ims]
    rois = []
    for cam in cams:
        uv = cam.project(np.r_[xy, 0.08])[0].astype(int)
        rois.append((uv[0] - 60, uv[1] - 90, uv[0] + 60, uv[1] + 90))

    def cost(x):
        r, z0, z1 = x
        if not (0.005 < r < 0.05 and z1 > z0 + 0.01):
            return 10.0
        d = TapeDims(R=r, r_core=r_rod, r_in=r_rod, w=z1 - z0, core=False)
        pts, lab = sample_surface(d, step=0.0007)
        c = 0.0
        for cam, roi, m in zip(cams, rois, masks):
            x0, y0, x1, y1 = roi
            img = m[y0:y1, x0:x1]
            ren = render_labels(cam, np.eye(3), np.r_[xy, (z0 + z1) / 2], pts, lab, roi) > 0
            c += 1 - (ren & img).sum() / max((ren | img).sum(), 1)
        return c / 2
    r = minimize(cost, [0.019, 0.05, 0.11], method="Powell", options=dict(xtol=1e-5, ftol=1e-5))
    r_sp, z0_sp, z1_sp = r.x
    block_top = z0_sp - h_base
    # turn one block face to the right camera
    cdir = cams[1].T_cam2base[:2, 3] - xy
    yaw = float(np.arctan2(cdir[1], cdir[0]))
    nd = json.loads(nut_dims.read_text())
    tgt = nd["poses"][-1]
    scene = dict(
        table_z=table_z,
        bolt_stand=dict(
            xy=xy.round(4).tolist(),
            block=dict(size=[round(block, 4), round(block, 4), round(block_top - table_z, 4)], yaw=round(yaw, 3),
                       z0=table_z, z1=round(float(block_top), 4)),
            base=dict(radius=round(r_base, 4), z0=round(float(block_top), 4), z1=round(float(z0_sp), 4)),
            spacers=dict(radius=round(float(r_sp), 4), z0=round(float(z0_sp), 4), z1=round(float(z1_sp), 4),
                         fit_iou=round(1 - float(r.fun), 3)),
            rod=dict(radius=round(r_rod, 4), z0=round(float(z1_sp), 4), z1=round(float(tip[2]), 4)),
            reproj_px=[round(e_tip, 2), round(e_top, 2)]),
        target=dict(centre=tgt["centre"], axis=tgt["axis"]),
        source="hand-picked points + green silhouette fit, front and right RealSense views, RGB-only capture ep 4",
    )
    print(json.dumps(scene["bolt_stand"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scene, indent=1))

    # overlay check
    from measure_mug_tree import cylinder_outline
    tiles = []
    for i, (cam, im) in enumerate(zip(cams, ims)):
        im = im.copy()
        def cyl(z0, z1, rr, col):
            lo, hi = cylinder_outline(np.r_[xy, z0], np.r_[xy, z1], rr, 48)
            for ring in (lo, hi):
                cv2.polylines(im, [cam.project(ring)[0].astype(np.int32)], True, col, 1)
            for j in (12, 36):
                p = cam.project(np.stack([lo[j], hi[j]]))[0].astype(int)
                cv2.line(im, tuple(p[0]), tuple(p[1]), col, 1)
        cyl(z0_sp, z1_sp, r_sp, (255, 0, 255)); cyl(z1_sp, tip[2], r_rod, (0, 255, 255)); cyl(block_top, z0_sp, r_base, (0, 128, 255))
        hs = block / 2
        cs = np.array([[sx, sy] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]) * hs
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        cxy = xy + cs @ Rz.T
        for z in (table_z, block_top):
            cv2.polylines(im, [cam.project(np.c_[cxy, np.full(4, z)])[0].astype(np.int32)], True, (0, 255, 0), 1)
        uv = cam.project(np.r_[xy, 0.08])[0].astype(int)
        tiles.append(cv2.resize(im[uv[1] - 110:uv[1] + 90, uv[0] - 80:uv[0] + 80], (320, 400)))
    cv2.imwrite(str(output.parent / "stand_overlay.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    tyro.cli(main)
