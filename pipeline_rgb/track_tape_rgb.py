"""Track the tape roll's pose through a human demo from the two calibrated RGB views (no depth).

Per frame, the measured roll model (fit_tape_dims.py) is fitted to the blue silhouette in both
views at once (tape_model.py: z-buffered render, 1 - IoU, hand pixels ignored), starting from the
previous frame's pose. Two views ~90 deg apart pin down the centre and the axis direction; the spin
about the axis is unobservable for a roll and irrelevant to it, so the output frame is carried along
with minimal twist. The fitted track is smoothed (centre and axis) and resampled to the video rate.

    python scripts/ur_tape/track_tape_rgb.py --episode 1 --output outputs/ur_tape/traj_ep1_hang.json

Output (same format as the cube track): T_base_obj (T, wxyz_xyz) in the robot base frame for the
pick -> place segment, plus the per-frame fit quality and the full-episode track.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from tape_model import (BLUE, COLOUR_MASKS, TapeDims, angles_from_axis, axis_from_angles, frame_from_axis,  # noqa: E402
                        render_labels, roi_around, sample_surface, skin_mask)

VIDEOS = Path(os.environ.get("UR_RGB_VIDEOS", Path.home() / "Downloads/ur_human_capture/videos"))


def read_all(ep: int, cam: int):
    cap = cv2.VideoCapture(str(VIDEOS / str(ep) / f"{cam}.mp4"))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    return frames


class Fitter:
    """Silhouette fit of a ring-shaped object in the calibrated views. `static` (per view, full-image
    bool masks) marks same-coloured scenery to ignore, e.g. the green spacers on the bolt stand. A
    view where neither the model nor the image shows the object (out of frame) carries no information
    and is skipped; with a single informative view only x, y are fitted (height and axis kept), which
    is right for an object resting on the bench."""

    def __init__(self, cams, dims: TapeDims, colour: str = "blue", static=None, roi_r: float = 0.11):
        self.cams, self.dims, self.roi_r = cams, dims, roi_r
        self.mask = COLOUR_MASKS[colour]
        self.static = static
        self.pts, self.lab = sample_surface(dims, step=0.0009 if dims.R > 0.035 else 0.0006)

    def observe(self, ims, centre):
        obs = []
        for i, (cam, im) in enumerate(zip(self.cams, ims)):
            roi = roi_around(cam, centre, self.roi_r)
            x0, y0, x1, y1 = roi
            unk = skin_mask(im)[y0:y1, x0:x1]
            if self.static is not None:
                unk = unk | self.static[i][y0:y1, x0:x1]
            obs.append((roi, self.mask(im)[y0:y1, x0:x1], unk))
        return obs

    def view_costs(self, obs, p):
        R = frame_from_axis(axis_from_angles(p[3], p[4]))
        out = []
        for cam, (roi, img, unk) in zip(self.cams, obs):
            if roi[2] <= roi[0] or roi[3] <= roi[1]:
                out.append(None)
                continue
            m = render_labels(cam, R, p[:3], self.pts, self.lab, roi) == BLUE
            k = ~unk
            union = ((m | img) & k).sum()
            out.append(None if union == 0 else (1 - (m & img & k).sum() / union, int(img[k].sum())))
        return out

    def cost(self, obs, p):
        v = [c for c in self.view_costs(obs, p) if c is not None]
        if not v:
            return 1.0, 0
        return sum(c for c, _ in v) / len(v), sum(n for _, n in v)

    def fit(self, ims, p0, seeds=(), keep_below=None):
        obs = self.observe(ims, p0[:3])
        if keep_below is not None:  # nothing moved: the previous pose still fits as well as before
            c0, m0 = self.cost(obs, p0)
            if c0 <= keep_below:
                return p0, float(c0), int(m0)
        n_views = sum(c is not None for c in self.view_costs(obs, p0))
        if n_views < 2:  # one informative view: slide on the bench only
            r = minimize(lambda xy: self.cost(obs, np.r_[xy, p0[2:]])[0], p0[:2], method="Powell",
                         options=dict(xtol=2e-4, ftol=1e-4, maxfev=400))
            q = np.r_[r.x, p0[2:]]
            return q, float(r.fun), int(self.cost(obs, q)[1])
        best = None
        for s in (p0, *seeds):
            r = minimize(lambda p: self.cost(obs, p)[0], s, method="Powell",
                         options=dict(xtol=2e-4, ftol=1e-4, maxfev=900))
            if best is None or r.fun < best.fun:
                best = r
        return best.x, float(best.fun), int(self.cost(obs, best.x)[1])


def static_masks(cams, ims0, dims, colour, pose, pad: int = 15):
    """Same-coloured pixels in the first frame that are not the object (pose = its first pose)."""
    pts, lab = sample_surface(dims)
    out = []
    for cam, im in zip(cams, ims0):
        m = render_labels(cam, frame_from_axis(axis_from_angles(pose[3], pose[4])), pose[:3], pts, lab,
                          (0, 0, cam.W, cam.H)) > 0
        m = cv2.dilate(m.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
        out.append(COLOUR_MASKS[colour](im) & ~m)
    return out


WORKSPACE = ((-0.45, 0.55), (-1.15, -0.25), (-0.04, 0.45))  # base frame box over the bench


def redetect(cams, ims, colour, static, predicted, max_err_px: float = 6.0, min_area: int = 25, max_jump: float = 1.0):
    """Object position from colour blobs in both views (static scenery and skin removed): the
    triangulated blob pair closest to the predicted position, if any pair is consistent."""
    from cams import triangulate
    blobs = []
    for i, im in enumerate(ims):
        m = COLOUR_MASKS[colour](im) & ~skin_mask(im)
        if static is not None:
            m &= ~static[i]
        n, _, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        blobs.append([cen[k] for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= min_area])
    best = None
    for u0 in blobs[0]:
        for u1 in blobs[1]:
            X, err = triangulate(cams, [u0, u1])
            inside = all(lo < x < hi for x, (lo, hi) in zip(X, WORKSPACE))
            d = np.linalg.norm(X - predicted)
            if err < max_err_px and inside and d < max_jump:
                if best is None or d < best[0]:
                    best = (d, X)
    return None if best is None else best[1]


def flip_consistent(p, prev):
    """Same axis, sign closest to the previous frame's (the roll is symmetric under flips)."""
    a, b = axis_from_angles(p[3], p[4]), axis_from_angles(prev[3], prev[4])
    if a @ b < 0:
        th, ph = np.pi - p[3], p[4] + np.pi
        return np.r_[p[:3], th, ph]
    return p


def minimal_twist_frames(axes):
    """Rotations with z = axes[t], x carried along by the smallest rotation between steps."""
    Rs = [frame_from_axis(axes[0])]
    for a in axes[1:]:
        z0 = Rs[-1][:, 2]
        z1 = a / np.linalg.norm(a)
        v = np.cross(z0, z1)
        s, c = np.linalg.norm(v), z0 @ z1
        if s < 1e-9:
            Rs.append(np.stack([Rs[-1][:, 0], Rs[-1][:, 1], z1], 1))
            continue
        rot = Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
        Rs.append(rot @ Rs[-1])
    return np.stack(Rs)


def segment_and_smooth(centres, axes, fps, move_thresh=0.004, pad_s=0.25, smooth_s=0.23):
    """Pick -> place segment (just before the object starts moving to just after it settles),
    Savitzky-Golay smoothed centre and axis, minimal-twist frames. Returns f0, f1, T (wxyz_xyz), window."""
    n = len(centres)
    speed = np.linalg.norm(np.gradient(savgol_filter(centres, 9, 2, axis=0), axis=0), axis=1) * fps
    moving = np.where(speed > move_thresh * fps / 3)[0]
    pad = int(pad_s * fps)
    f0, f1 = max(moving[0] - pad, 0), min(moving[-1] + pad, n - 1)
    win = max(int(smooth_s * fps) | 1, 5)
    sc = savgol_filter(centres, win, 2, axis=0)
    sa = savgol_filter(axes, win, 2, axis=0)
    sa /= np.linalg.norm(sa, axis=1, keepdims=True)
    seg = slice(f0, f1 + 1)
    Rs = minimal_twist_frames(sa[seg])
    q = Rotation.from_matrix(Rs).as_quat()[:, [3, 0, 1, 2]]
    return int(f0), int(f1), np.concatenate([q, sc[seg]], 1), win


def main(episode: int = 1, output: Path = Path("outputs/ur_tape/traj_ep1_hang.json"),
         dims_file: Path = Path("outputs/ur_tape/tape_dims.json"), fps: float = 30.0,
         move_thresh: float = 0.004, pad_s: float = 0.25, smooth_s: float = 0.23, colour: str = "blue",
         ignore_static: bool = False, roi_r: float = 0.11, obj_name: str = "tape_roll", lost_cost: float = 0.45) -> None:
    """ignore_static: treat same-coloured scenery in the first frame (not the object) as unknown."""
    cams = load_cams()
    dj = json.loads(dims_file.read_text())
    dims = TapeDims.from_json(dj)
    vids = [read_all(episode, c) for c in range(2)]
    n = min(len(v) for v in vids)
    print(f"episode {episode}: {n} frames", flush=True)

    # start: the dims fit's flat pose for this episode's first frame, else search from the bench centre
    start = next((p for p in dj["poses"] if p["frame"] == f"{episode}:0"), None)
    p = np.r_[start["centre"], angles_from_axis(np.array(start["axis"]))] if start else None
    assert p is not None, "add this episode's first frame to the dims fit's frames"
    static = static_masks(cams, [vids[0][0], vids[1][0]], dims, colour, p) if ignore_static else None
    fitter = Fitter(cams, dims, colour=colour, static=static, roi_r=roi_r)

    track, quality, npx = [], [], []
    still, last_good = 0, None
    for f in range(n):
        ims = [vids[0][f], vids[1][f]]
        seeds = []
        if f > 1:  # constant-velocity prediction as a second start
            v = track[-1] - track[-2]
            seeds.append(track[-1] + v)
        q, c, m = fitter.fit(ims, p, seeds, keep_below=quality[-1] + 0.01 if quality and still < 15 else None)
        if c > lost_cost:  # lost (fast motion, occlusion): restart from the triangulated colour blobs
            pred = last_good[:3] if last_good is not None else p[:3]
            X = redetect(cams, ims, colour, static, pred)
            if X is not None:
                q2, c2, m2 = fitter.fit(ims, np.r_[X, p[3:]])
                if c2 < c:
                    q, c, m = q2, c2, m2
                    print(f"f{f:4d} re-detected at {X.round(3)}, 1-IoU {c:.3f}", flush=True)
        still = still + 1 if q is p else 0
        q = flip_consistent(q, p)
        if c <= lost_cost:
            last_good = q
        track.append(q)
        quality.append(c)
        npx.append(m)
        p = q
        if f % 20 == 0:
            print(f"f{f:4d} centre {q[:3].round(3)} axis {axis_from_angles(q[3], q[4]).round(2)} 1-IoU {c:.3f} blue px {m}",
                  flush=True)
    track = np.array(track)
    centres = track[:, :3]
    axes = np.array([axis_from_angles(t[3], t[4]) for t in track])

    f0, f1, T, win = segment_and_smooth(centres, axes, fps, move_thresh, pad_s, smooth_s)
    out = dict(
        T_base_obj=T.tolist(),
        source=f"two-view RGB silhouette track (front + right RealSense, no depth), episode {episode}, "
               f"frames {f0}-{f1} @ {fps} Hz, Savitzky-Golay {win} frames",
        object=obj_name, dims=dims.__dict__, frames=[int(f0), int(f1)], fps=fps,
        fit_1_minus_iou=np.round(quality, 4).tolist(), blue_pixels=npx,
        full_track=dict(centre=np.round(centres, 5).tolist(), axis=np.round(axes, 4).tolist()),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out))
    print(f"segment frames {f0}-{f1} ({len(T)} steps); start {T[0, 4:].round(3)} end {T[-1, 4:].round(3)}; "
          f"median 1-IoU {np.median(quality[f0:f1 + 1]):.3f}, max {np.max(quality[f0:f1 + 1]):.3f}")


if __name__ == "__main__":
    tyro.cli(main)
