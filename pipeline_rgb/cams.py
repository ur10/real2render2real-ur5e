"""The two calibrated RealSense views of the bench (front = video 0, right = video 1).

The RGB-only capture in ~/Downloads/ur_human_capture has no camera metadata; the cameras are the ones
calibrated in the RGB-D capture (~/ur_human_capture), which have not moved: static points match to
~1 px and a static point seen in both views reprojects within 0.3 px. Intrinsics and T_cam2base come
from that capture's meta.json. Distortion is ignored, as in the cube pipeline and the Isaac renders.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# the two cameras' intrinsics + T_cam2base (copied from the RGB-D capture's meta.json); falls back to that file
CALIB_META = REPO / "outputs/ur_tape/realsense_calib.json"
if not CALIB_META.exists():
    CALIB_META = Path.home() / "ur_human_capture/rgbd/1/000127/meta.json"
NAMES = ("realsense_front", "realsense_right")


class Cam:
    def __init__(self, meta: dict):
        it = meta["intrinsics"]
        self.K = np.array([[it["fx"], 0, it["ppx"]], [0, it["fy"], it["ppy"]], [0, 0, 1]])
        self.T_cam2base = np.array(meta["T_cam2base"])
        self.T_base2cam = np.linalg.inv(self.T_cam2base)
        self.P = self.K @ self.T_base2cam[:3]
        self.W, self.H = it["width"], it["height"]

    def project(self, X):
        """(..., 3) base-frame points -> (..., 2) pixels, (...,) depth."""
        Xc = X @ self.T_base2cam[:3, :3].T + self.T_base2cam[:3, 3]
        uv = Xc @ self.K.T
        return uv[..., :2] / uv[..., 2:3], Xc[..., 2]

    def ray(self, uv):
        """Pixel -> (origin, unit direction) in the base frame."""
        d = self.T_cam2base[:3, :3] @ np.linalg.solve(self.K, np.array([uv[0], uv[1], 1.0]))
        return self.T_cam2base[:3, 3], d / np.linalg.norm(d)


def load_cams(meta_file: Path = CALIB_META) -> list:
    m = json.loads(Path(meta_file).read_text())["cameras"]
    return [Cam(m[n]) for n in NAMES]


def triangulate(cams, uvs):
    """Linear triangulation from >= 2 views. Returns X (3,), max reprojection error (px)."""
    A = []
    for c, (u, v) in zip(cams, uvs):
        A += [u * c.P[2] - c.P[0], v * c.P[2] - c.P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]
    X = X[:3] / X[3]
    err = max(np.linalg.norm(c.project(X)[0] - np.array(uv)) for c, uv in zip(cams, uvs))
    return X, err
