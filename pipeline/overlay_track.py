"""Draw the tracked cube (base-frame poses) as a 3D box over both RealSense views.

The front camera was used for tracking; the right camera was not, so a box that stays on the cube
in the right view checks the whole chain (track -> scale -> T_cam2base) independently.

Example:
    python scripts/ur_cube/overlay_track.py --traj outputs/ur_cube/traj_ep0_f1_to_f0.json \
        --video-start-frame 156 --output /tmp/track_overlay.jpg
"""

import json
from pathlib import Path

import cv2
import numpy as np
import tyro

from rgbd_utils import load_frame, project


def box_edges(T: np.ndarray, size) -> list:
    hx, hy, hz = np.asarray(size) / 2
    c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    pts = c @ T[:3, :3].T + T[:3, 3]
    idx = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return pts, idx


def main(
    traj: Path,
    output: Path,
    video_start_frame: int,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    video_fps: float = 30.0,
    rgbd_hz: float = 10.0,
    num_tiles: int = 6,
) -> None:
    meta = json.loads(traj.read_text())
    poses = np.load(traj.with_suffix(".poses_all.npy"))  # (T, 4, 4) at video rate, full track
    size = meta["cube_size_m"]
    picks = np.linspace(0, len(poses) - 1, num_tiles).round().astype(int)
    rows = []
    for k in picks:
        step = int(round((video_start_frame + k) / video_fps * rgbd_hz))
        frame = load_frame(capture_dir, episode, step)
        tiles = []
        for name in ("front", "right"):
            cam = frame["cameras"][f"realsense_{name}"]
            img = cam["color"].copy()
            pts, idx = box_edges(poses[k], size)
            uv, _ = project(cam, pts)
            for a, b in idx:
                cv2.line(img, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)), (255, 0, 255), 2, cv2.LINE_AA)
            ctr = uv.mean(0).astype(int)
            x0, y0 = np.clip(ctr - [200, 130], 0, [1280 - 400, 720 - 260])
            crop = img[y0 : y0 + 260, x0 : x0 + 400]
            cv2.putText(crop, f"{name} t={k} step={step}", (5, 20), 0, 0.55, (0, 255, 0), 2)
            tiles.append(crop)
        rows.append(np.hstack(tiles))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
