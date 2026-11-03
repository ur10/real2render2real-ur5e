"""Build an RSRD input clip from the RealSense RGB-D frames: color video + aligned metric depth.

RSRD's metric-depth loss pins the object's distance, which monodepth + DINO can't for a small,
far object. Depth exists only at the capture rate (10 Hz), so the clip is built from the RGB-D
steps themselves. Color and depth get the same undistortion and optional zoom crop (depth with
nearest-neighbour sampling). Depth is written in scan (COLMAP) units, which RSRD treats as metres.

Example:
    python scripts/ur_cube/prep_rgbd_clip.py --episode 0 --camera front --steps 52 85 \
        --crop 550 248 640 360 --output demonstrations/cube_pnp_ep0_front_rgbd_s52_85.mp4
writes <output>, <stem>_depth.npy (T, H, W float32) and <stem>_intr.json.
"""

import json
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import tyro

from rgbd_utils import intrinsics_matrix, load_frame


def main(
    episode: int,
    camera: Literal["front", "right"],
    steps: tuple[int, int],
    output: Path,
    crop: Optional[tuple[int, int, int, int]] = None,
    capture_dir: Path = Path.home() / "ur_human_capture",
    scan_to_base: Path = Path("outputs/table_new/scan_to_base.json"),
    capture_hz: float = 10.0,
) -> None:
    m_per_unit = json.loads(scan_to_base.read_text())["scale_m_per_colmap_unit"]
    first = load_frame(capture_dir, episode, steps[0])["cameras"][f"realsense_{camera}"]
    K, dist = intrinsics_matrix(first), np.array(first["intrinsics"]["coeffs"])
    h, w = first["depth_m"].shape
    m1, m2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if crop is not None:
        x0, y0, cw, ch = crop
        s = w / cw
        assert abs(h / ch - s) < 1e-6, "crop must keep the frame aspect ratio"
        fx, fy, cx, cy = fx * s, fy * s, (cx - x0) * s, (cy - y0) * s

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), capture_hz, (w, h))
    depths = []
    for step in range(steps[0], steps[1] + 1):
        cam = load_frame(capture_dir, episode, step)["cameras"][f"realsense_{camera}"]
        color = cv2.remap(cam["color"], m1, m2, cv2.INTER_LINEAR)
        depth = cv2.remap(cam["depth_m"], m1, m2, cv2.INTER_NEAREST)
        if crop is not None:
            color = cv2.resize(color[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
            depth = cv2.resize(depth[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
        depth = np.where(depth > 0.1, depth / m_per_unit, np.nan).astype(np.float32)
        writer.write(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        depths.append(depth)
    writer.release()
    depths = np.stack(depths)
    np.save(output.with_name(output.stem + "_depth.npy"), depths)

    intr = {
        "name": f"realsense_{camera}", "fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": w, "height": h,
        "T_cam2base": first["T_cam2base"], "fps": capture_hz, "num_frames": len(depths),
        "episode": episode, "steps": list(steps), "crop": list(crop) if crop is not None else None,
        "depth_units": f"scan units (m / {m_per_unit:.5f})",
    }
    output.with_name(output.stem + "_intr.json").write_text(json.dumps(intr, indent=2))
    valid = np.isfinite(depths)
    print(f"wrote {len(depths)} frames to {output}; depth valid {valid.mean():.2f}, "
          f"median {np.nanmedian(depths) * m_per_unit:.3f} m")


if __name__ == "__main__":
    tyro.cli(main)
