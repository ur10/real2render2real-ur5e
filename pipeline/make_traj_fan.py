"""Trajectory-interpolation figure: the tracked demo and the generator's warped variants, drawn on the
real front RealSense frame (same distribution as UR5eCubePnP: start jitter on the pick fixture,
leading 60% of the path warped by trajgen, place pose kept).

Example:
    python scripts/ur_cube/make_traj_fan.py --traj outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json \
        --output outputs/ur_cube/media/traj_fan.jpg
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from trajgen.traj_interp import traj_interp_batch

from rgbd_utils import load_frame, project

FONTS = Path(__file__).resolve().parents[2] / "outputs/ur_cube/fonts"
TEAL, AMBER = (95, 181, 192), (230, 160, 70)


def main(
    traj: Path,
    output: Path,
    n: int = 64,
    xy_noise: float = 0.03,
    yaw_noise_deg: float = 30.0,
    capture_dir: Path = Path.home() / "ur_human_capture",
    step: int = 52,
    seed: int = 0,
) -> None:
    demo = np.asarray(json.loads(traj.read_text())["T_base_obj"])  # (T, 7) wxyz_xyz
    rng = np.random.default_rng(seed)
    starts = np.repeat(demo[:1], n, axis=0)
    yaw = Rotation.from_euler("z", rng.uniform(-1, 1, n) * np.radians(yaw_noise_deg))
    q0 = Rotation.from_quat(demo[0, [1, 2, 3, 0]])
    q = (yaw * q0).as_quat()
    starts[:, :4] = q[:, [3, 0, 1, 2]]
    starts[:, 4:6] += rng.uniform(-1, 1, (n, 2)) * xy_noise
    trajs = np.asarray(traj_interp_batch(traj=demo, new_starts=starts, proportion=0.6))  # (n, T, 7)

    cam = load_frame(capture_dir, 0, step)["cameras"]["realsense_front"]
    # Crop to the paths (16:9, with margin) so the actual spread is readable.
    allp = project(cam, trajs[:, :, 4:].reshape(-1, 3))[0]
    lo, hi = allp.min(0) - 70, allp.max(0) + 70
    h = max(hi[1] - lo[1], (hi[0] - lo[0]) * 9 / 16)
    w = h * 16 / 9
    cx, cy = (lo + hi) / 2
    x0, y0 = int(np.clip(cx - w / 2, 0, 1280 - w)), int(np.clip(cy - h / 2, 0, 720 - h))
    w, h = int(w), int(h)
    out_w = 1600
    s = out_w / w
    img = cv2.resize(cam["color"][y0 : y0 + h, x0 : x0 + w], (out_w, int(h * s)), interpolation=cv2.INTER_CUBIC)
    img = (img * 0.62 + 20).astype(np.uint8)  # dim the photo so the paths read
    layer = img.copy()

    def px(p):
        return ((project(cam, p)[0] - [x0, y0]) * s).astype(np.int32)

    for t in trajs:
        cv2.polylines(layer, [px(t[:, 4:])], False, AMBER, 2, cv2.LINE_AA)
    img = cv2.addWeighted(layer, 0.55, img, 0.45, 0)
    for t in trajs:  # start markers
        cv2.circle(img, tuple(px(t[:1, 4:])[0]), 4, AMBER, -1, cv2.LINE_AA)
    cv2.polylines(img, [px(demo[:, 4:])], False, TEAL, 5, cv2.LINE_AA)
    cv2.circle(img, tuple(px(demo[:1, 4:])[0]), 8, TEAL, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(px(demo[-1:, 4:])[0]), 8, TEAL, -1, cv2.LINE_AA)

    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    mono = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), 30)
    mono.set_variation_by_name("Bold")
    d.text((28, 22), "1 tracked demo", font=mono, fill=TEAL)
    d.text((28, 58), f"{n} interpolated trajectories", font=mono, fill=AMBER)
    small = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), 23)
    small.set_variation_by_name("SemiBold")
    d.text((28, 94), f"start jitter ±{xy_noise * 100:.0f} cm, ±{yaw_noise_deg:.0f}° · place pose kept", font=small, fill=(225, 225, 225))
    output.parent.mkdir(parents=True, exist_ok=True)
    pil.save(output, quality=90)
    print(f"wrote {output} ({pil.size[0]}x{pil.size[1]})")


if __name__ == "__main__":
    tyro.cli(main)
