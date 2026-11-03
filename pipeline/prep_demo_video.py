"""Undistort a RealSense demo video so RSRD can treat it as a pinhole camera.

The tracker ignores lens distortion, so we undistort each frame while keeping
the original camera matrix K. The intrinsics to pass to run_tracker.py are
written next to the output video as <name>_intr.json.

Example:
    python scripts/ur_cube/prep_demo_video.py \
        --capture-dir ~/ur_human_capture --episode 0 --camera front \
        --output demonstrations/cube_pnp_ep0_front.mp4
"""

import json
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import tyro

# videos/<ep>/0.mp4 is the front RealSense, 1.mp4 the right one (checked against rgbd/ frames).
VIDEO_INDEX = {"front": 0, "right": 1}


def load_camera(capture_dir: Path, episode: int, camera: str) -> dict:
    meta = json.loads((capture_dir / "rgbd" / str(episode) / "000000" / "meta.json").read_text())
    return meta["cameras"][f"realsense_{camera}"]


def main(
    capture_dir: Path,
    episode: int,
    camera: Literal["front", "right"],
    output: Path,
    frames: Optional[tuple[int, int]] = None,
    crop: Optional[tuple[int, int, int, int]] = None,
) -> None:
    """
    Args:
        frames: inclusive (first, last) video frame range to keep.
        crop: (x0, y0, w, h) digital zoom, rescaled back to the full frame size. RSRD rasterizes
            at 490 px, so a small/far object needs zooming to cover more than one DINO patch.
            Intrinsics are adjusted to match; the camera pose (T_cam2base) is unchanged.
    """
    cam = load_camera(capture_dir, episode, camera)
    intr = cam["intrinsics"]
    # RealSense "inverse_brown_conrady" color distortion uses OpenCV's (k1, k2, p1, p2, k3) convention.
    assert intr["model"] == "inverse_brown_conrady", intr["model"]
    K = np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
    dist = np.array(intr["coeffs"])
    w, h = intr["width"], intr["height"]
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (w, h), cv2.CV_32FC1)

    src = capture_dir / "videos" / str(episode) / f"{VIDEO_INDEX[camera]}.mp4"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    if crop is not None:
        x0, y0, cw, ch = crop
        s = w / cw
        assert abs(h / ch - s) < 1e-6, "crop must keep the frame aspect ratio"
        fx, fy, cx, cy = fx * s, fy * s, (cx - x0) * s, (cy - y0) * s
    n, idx = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if frames is not None and not (frames[0] <= idx <= frames[1]):
            continue
        assert frame.shape[:2] == (h, w), frame.shape
        frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        if crop is not None:
            frame = cv2.resize(frame[y0 : y0 + ch, x0 : x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)
        writer.write(frame)
        n += 1
    writer.release()

    intr_out = {
        "name": f"realsense_{camera}",
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": w,
        "height": h,
        "frames": list(frames) if frames is not None else None,
        "crop": list(crop) if crop is not None else None,
        "T_cam2base": cam["T_cam2base"],
        "source_video": str(src),
        "fps": fps,
        "num_frames": n,
    }
    output.with_name(output.stem + "_intr.json").write_text(json.dumps(intr_out, indent=2))
    print(f"Wrote {n} undistorted frames at {fps} fps to {output}")


if __name__ == "__main__":
    tyro.cli(main)
