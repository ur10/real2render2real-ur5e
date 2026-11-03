"""Write RSRD's first-frame object pose (track_init.json) from the calibration instead of DINO seeds.

RSRD's seed search localizes the object by DINO feature matching, which drifts away for a small,
featureless cube ~1 m from the camera. But the phone scan was taken with the cube where the demo
starts, so its pose in the demo camera follows from the calibrations alone:
    T_camGL_model = GL<-CV @ inv(T_base_cam) @ T_base_model   (scale removed: model units)
    T_objreg_objinit = inv(T_world_objinit) @ T_camGL_model @ T_world_objinit
where T_world_objinit = [I | gaussian centroid] is RSRD's object frame. RSRD loads this file and
skips its seed search.

Example:
    python scripts/ur_cube/calib_track_init.py --state outputs/table_new/state_rigid_20261102_190817.pt \
        --intr demonstrations/cube_pnp_ep0_front_rgbd_s52_85_intr.json --track-dir outputs/table_new/track
"""

import json
from pathlib import Path

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def main(
    state: Path,
    intr: Path,
    track_dir: Path,
    scan_to_base: Path = Path("outputs/table_new/scan_to_base.json"),
) -> None:
    cal = json.loads(scan_to_base.read_text())
    T_base_model = np.array(cal["T_base_model"])
    s = np.cbrt(np.linalg.det(T_base_model[:3, :3]))  # metres per model unit
    cam = json.loads(intr.read_text())
    T_base_cam = np.array(cam["T_cam2base"])

    T_cv_model = np.linalg.inv(T_base_cam) @ T_base_model  # model units -> camera metres (scaled)
    T_gl_model = GL_TO_CV @ T_cv_model
    T_gl_model[:3, :3] /= s  # back to a rigid transform in model units
    T_gl_model[:3, 3] /= s

    centroid = torch.load(state, map_location="cpu")["means"].detach().numpy().mean(0)
    T_wo = np.eye(4)
    T_wo[:3, 3] = centroid
    T_ro = np.linalg.inv(T_wo) @ T_gl_model @ T_wo

    x, y, z, w = Rotation.from_matrix(T_ro[:3, :3]).as_quat()
    pose = [w, x, y, z, *T_ro[:3, 3]]
    track_dir.mkdir(parents=True, exist_ok=True)
    (track_dir / "track_init.json").write_text(json.dumps({"best_pose": [pose]}))

    # Sanity: project the posed centroid with the clip's intrinsics.
    p = (T_wo @ T_ro)[:3, 3]  # object centroid in the camera (GL), model units
    u = cam["fx"] * p[0] / -p[2] + cam["cx"]
    v = cam["fy"] * -p[1] / -p[2] + cam["cy"]
    print(f"centroid in camera: depth {-p[2] * s:.3f} m, pixel ({u:.1f}, {v:.1f}) of {cam['width']}x{cam['height']}")
    print(f"wrote {track_dir / 'track_init.json'}: {np.round(pose, 4).tolist()}")


if __name__ == "__main__":
    tyro.cli(main)
