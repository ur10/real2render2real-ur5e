"""Measured-shape model of the blue tape roll and a two-view silhouette fit.

The roll is an annulus around its axis (object z): a band of blue tape (radius r_core..R) wound on
a white plastic core (r_in..r_core), width w. It looks the same under any rotation about its axis,
so a pose has 5 DoF: centre (3) + axis direction (2).

Rendering is z-buffered point splatting: the solid's surfaces are sampled densely (< 0.5 px apart
at the bench's ~1 m), projected into each calibrated view and the nearest point per pixel wins, so
the core hides the blue band behind it as in the real images. Only the blue label is compared with
the image (a colour mask); the white core is hard to separate from the bench, but it still occludes.
"""

from dataclasses import dataclass

import cv2
import numpy as np

BLUE, CORE = 1, 2


@dataclass
class TapeDims:
    R: float = 0.050       # outer radius of the blue band
    r_core: float = 0.040  # blue band inner radius = core outer radius
    r_in: float = 0.037    # core inner radius
    w: float = 0.024       # width (along the axis)
    core: bool = True      # False: one colour throughout (a plain ring such as the green nut)

    def as_list(self):
        return [self.R, self.r_core, self.r_in, self.w]

    @staticmethod
    def from_json(d: dict) -> "TapeDims":
        return TapeDims(**{k: d[k] for k in ("R", "r_core", "r_in", "w", "core") if k in d})


def axis_from_angles(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def angles_from_axis(a):
    a = a / np.linalg.norm(a)
    if a[2] < 0:  # the roll is symmetric under flipping its axis
        a = -a
    return np.arccos(np.clip(a[2], -1, 1)), np.arctan2(a[1], a[0])


def frame_from_axis(a, ref_x=np.array([1.0, 0.0, 0.0])):
    """Rotation whose z column is `a` and whose x column is ref_x projected off the axis."""
    z = a / np.linalg.norm(a)
    x = ref_x - z * (ref_x @ z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 1.0, 0.0]) - z * z[1]
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def sample_surface(d: TapeDims, step: float = 0.0007):
    """Points (N, 3) in the object frame and labels (N,): outer wall, end faces, core, inner wall."""
    pts, lab = [], []

    def ring_wall(r, label):
        n = max(int(2 * np.pi * r / step), 8)
        m = max(int(d.w / step), 2)
        t, z = np.meshgrid(np.linspace(0, 2 * np.pi, n, endpoint=False), np.linspace(-d.w / 2, d.w / 2, m))
        pts.append(np.stack([r * np.cos(t), r * np.sin(t), z], -1).reshape(-1, 3))
        lab.append(np.full(t.size, label))

    def face(r0, r1, label):
        for zf in (-d.w / 2, d.w / 2):
            for r in np.arange(r0, r1 + 1e-9, step):
                n = max(int(2 * np.pi * r / step), 8)
                t = np.linspace(0, 2 * np.pi, n, endpoint=False)
                pts.append(np.stack([r * np.cos(t), r * np.sin(t), np.full(n, zf)], -1))
                lab.append(np.full(n, label))

    ring_wall(d.R, BLUE)
    if d.core:
        ring_wall(d.r_in, CORE)
        face(d.r_core, d.R, BLUE)
        face(d.r_in, d.r_core, CORE)
    else:  # one colour: BLUE stands for "the object's colour" throughout
        if d.r_in > step:
            ring_wall(d.r_in, BLUE)
        face(d.r_in, d.R, BLUE)
    return np.concatenate(pts), np.concatenate(lab)


def render_labels(cam, Rwo, t, pts, lab, roi):
    """Label image over roi=(x0, y0, x1, y1): 0 background, BLUE, CORE (z-buffered)."""
    x0, y0, x1, y1 = roi
    X = pts @ Rwo.T + t
    uv, z = cam.project(X)
    u = np.round(uv[:, 0]).astype(int) - x0
    v = np.round(uv[:, 1]).astype(int) - y0
    W, H = x1 - x0, y1 - y0
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    u, v, z, lb = u[ok], v[ok], z[ok], lab[ok]
    out = np.zeros((H, W), np.uint8)
    if len(z) == 0:
        return out
    order = np.argsort(-z)  # far first, near overwrites
    out[v[order], u[order]] = lb[order]
    # close pin-holes between splats without growing the outline
    blue = (out == BLUE).astype(np.uint8)
    any_ = (out > 0).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    any_c = cv2.morphologyEx(any_, cv2.MORPH_CLOSE, k)
    blue_c = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, k) & any_c
    out = np.where(blue_c > 0, BLUE, np.where(any_c > 0, np.maximum(out, CORE * (out == 0)), out)).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- image masks
def blue_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (100, 120, 50), (128, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0


def green_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (45, 90, 45), (88, 255, 255))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0


COLOUR_MASKS = {"blue": blue_mask, "green": green_mask}


def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (40, 138, 80), (255, 175, 125))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0


def roi_around(cam, centre, radius_m, pad=25):
    """Pixel box around a sphere of radius_m at centre, clipped to the image."""
    uv, z = cam.project(centre)
    r = cam.K[0, 0] * radius_m / max(z, 0.1) + pad
    return (int(max(uv[0] - r, 0)), int(max(uv[1] - r, 0)), int(min(uv[0] + r, cam.W)), int(min(uv[1] + r, cam.H)))
