"""Export the calibrated RealSense cameras for the IsaacLab camera manager.

Writes:
  <poses_h5>   dataset `poses` (N, 7): xyz + wxyz of each camera in the robot base frame, ROS/OpenCV
               optical convention (x right, y down, z forward), the format CameraManager loads.
  <intr_json>  per-camera intrinsics plus the averaged 3x3 K used by the shared MultiTiledCameraCfg.

Example:
    python scripts/ur_cube/export_realsense_cameras.py \
        --poses-h5 data/camera_poses/ur_realsense_front_right.h5 \
        --intr-json data/camera_poses/ur_realsense_intrinsics.json
"""

import json
from pathlib import Path

import h5py
import numpy as np
import tyro
from scipy.spatial.transform import Rotation

CAMERAS = ("front", "right")


def main(
    poses_h5: Path,
    intr_json: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
) -> None:
    meta = json.loads((capture_dir / "rgbd" / str(episode) / "000000" / "meta.json").read_text())
    poses, intr = [], {}
    for name in CAMERAS:
        cam = meta["cameras"][f"realsense_{name}"]
        T = np.asarray(cam["T_cam2base"])
        x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
        poses.append([*T[:3, 3], w, x, y, z])
        i = cam["intrinsics"]
        intr[name] = {k: i[k] for k in ("fx", "fy", "ppx", "ppy", "width", "height", "coeffs")}

    mean = {k: float(np.mean([intr[c][k] for c in CAMERAS])) for k in ("fx", "fy", "ppx", "ppy")}
    intr["shared_K"] = [mean["fx"], 0, mean["ppx"], 0, mean["fy"], mean["ppy"], 0, 0, 1]
    intr["width"], intr["height"] = intr["front"]["width"], intr["front"]["height"]
    intr["camera_order"] = list(CAMERAS)

    poses_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(poses_h5, "w") as f:
        f.create_dataset("poses", data=np.array(poses), maxshape=(None, 7), chunks=True)
    intr_json.write_text(json.dumps(intr, indent=2))
    for name, p in zip(CAMERAS, poses):
        print(f"{name}: xyz {np.round(p[:3], 4)} wxyz {np.round(p[3:], 4)}")
    print(f"shared K: fx {mean['fx']:.1f} fy {mean['fy']:.1f} cx {mean['ppx']:.1f} cy {mean['ppy']:.1f}")
    print(f"wrote {poses_h5} and {intr_json}")


if __name__ == "__main__":
    tyro.cli(main)
