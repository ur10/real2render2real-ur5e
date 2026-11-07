"""Measure the tape roll (outer radius, core radius, width) from its blue silhouette in both
calibrated views, on frames where it is at rest in two very different orientations: flat on the
bench and hanging on the mug tree. Dimensions are shared; each frame has its own 5-DoF pose.

    python scripts/ur_tape/fit_tape_dims.py --output outputs/ur_tape/tape_dims.json
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from tape_model import (BLUE, TapeDims, axis_from_angles, blue_mask, frame_from_axis, render_labels,  # noqa: E402
                        roi_around, sample_surface, skin_mask)

VIDEOS = Path(os.environ.get("UR_RGB_VIDEOS", Path.home() / "Downloads/ur_human_capture/videos"))


def read_frame(ep: int, cam: int, f: int):
    cap = cv2.VideoCapture(str(VIDEOS / str(ep) / f"{cam}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, im = cap.read()
    assert ok, (ep, cam, f)
    return im


def iou_cost(cams, obs, dims, pose):
    pts, lab = sample_surface(dims, step=0.0009)
    c, th, ph = pose[:3], pose[3], pose[4]
    R = frame_from_axis(axis_from_angles(th, ph))
    cost = 0.0
    for cam, (roi, img_blue, unknown) in zip(cams, obs):
        m = render_labels(cam, R, c, pts, lab, roi) == BLUE
        known = ~unknown
        inter = (m & img_blue & known).sum()
        union = ((m | img_blue) & known).sum()
        cost += 1 - inter / max(union, 1)
    return cost


def main(output: Path = Path("outputs/ur_tape/tape_dims.json"),
         frames: tuple[str, ...] = ("1:0", "1:470")) -> None:
    cams = load_cams()
    table_z = json.loads(Path("outputs/ur_cube/scene_layout.json").read_text())["table_z"]
    # rough starts (blob triangulation; hanging axis along the upper -y branch)
    init = {"1:0": ([-0.11, -0.821, table_z + 0.012], [0, 0, 1]),
            "1:470": ([0.158, -0.716, 0.267], [-0.3, -0.85, 0.42]),
            "2:440": ([0.20, -0.95, table_z + 0.012], [0, 0, 1])}
    obs, poses = [], []
    for key in frames:
        ep, f = map(int, key.split(":"))
        c0, a0 = init[key]
        o = []
        for i, cam in enumerate(cams):
            im = read_frame(ep, i, f)
            roi = roi_around(cam, np.array(c0), 0.09)
            x0, y0, x1, y1 = roi
            o.append((roi, blue_mask(im)[y0:y1, x0:x1], skin_mask(im)[y0:y1, x0:x1]))
        obs.append(o)
        from tape_model import angles_from_axis
        poses.append(np.r_[c0, angles_from_axis(np.array(a0, float))])
    # blob-centroid start for frames without a good guess: refine poses first with default dims
    d0 = TapeDims()
    for k in range(len(poses)):
        r = minimize(lambda p: iou_cost(cams, obs[k], d0, p), poses[k], method="Powell",
                     options=dict(xtol=1e-4, ftol=1e-4, maxfev=1500))
        poses[k] = r.x
        print(f"{frames[k]} pose-only cost {r.fun:.3f}", flush=True)

    def total(x):
        R, rc, w = x[:3]
        if not (0.02 < rc < R < 0.09 and 0.005 < w < 0.06):
            return 10.0
        d = TapeDims(R=R, r_core=rc, r_in=rc - 0.003, w=w)
        return sum(iou_cost(cams, obs[k], d, x[3 + 5 * k: 8 + 5 * k]) for k in range(len(poses)))

    x0 = np.r_[d0.R, d0.r_core, d0.w, np.concatenate(poses)]
    r = minimize(total, x0, method="Powell", options=dict(xtol=1e-5, ftol=1e-5, maxfev=8000))
    R, rc, w = r.x[:3]
    dims = TapeDims(R=float(R), r_core=float(rc), r_in=float(rc - 0.003), w=float(w))
    per = [iou_cost(cams, obs[k], dims, r.x[3 + 5 * k: 8 + 5 * k]) / 2 for k in range(len(poses))]
    print("dims", dims, "mean 1-IoU per view", np.round(per, 3))
    out = dict(R=dims.R, r_core=dims.r_core, r_in=dims.r_in, w=dims.w,
               fit_frames=list(frames), mean_iou_per_frame=[round(1 - p, 3) for p in per],
               poses=[dict(frame=frames[k], centre=r.x[3 + 5 * k: 6 + 5 * k].tolist(),
                           axis=axis_from_angles(*r.x[6 + 5 * k: 8 + 5 * k]).tolist()) for k in range(len(poses))])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1))

    # overlays for checking: model blue outline drawn on each view
    for k, key in enumerate(frames):
        ep, f = map(int, key.split(":"))
        tiles = []
        for i, cam in enumerate(cams):
            im = read_frame(ep, i, f)
            roi = obs[k][i][0]
            p = r.x[3 + 5 * k: 8 + 5 * k]
            pts, lab = sample_surface(dims)
            m = render_labels(cam, frame_from_axis(axis_from_angles(p[3], p[4])), p[:3], pts, lab, roi)
            x0, y0, x1, y1 = roi
            crop = im[y0:y1, x0:x1].copy()
            for lbl, col in ((BLUE, (0, 255, 255)), (2, (255, 0, 255))):
                cnts, _ = cv2.findContours((m == lbl).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(crop, cnts, -1, col, 1)
            tiles.append(cv2.resize(crop, (300, int(300 * crop.shape[0] / crop.shape[1]))))
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        cv2.imwrite(str(output.parent / f"dims_overlay_{key.replace(':', '_')}.jpg"), np.hstack(tiles))


if __name__ == "__main__":
    tyro.cli(main)
