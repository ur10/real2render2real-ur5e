"""Convert an RSRD rigid-object track into a metric cube trajectory in the UR base frame.

RSRD renders from a camera at the origin (nerfstudio/OpenGL axes) in DiG model units, and the
object pose at keyframe t is (rigid case, see rsrd/util/warp_kernels.py):
    T_camGL_obj(t) = T_world_objinit @ T_objreg_objinit @ part_deltas[t]
with the object frame at the Gaussians' centroid, axes as scanned. We scale translations by the
scan's metres-per-model-unit (scan_to_base.json; RSRD itself assumes a metric scan), switch the
camera to OpenCV axes, and apply the demo camera's calibrated T_cam2base.

The output uses the simulator's format (see make_synthetic_traj.py). The object frame is re-based
to the cube's own axes (z up, x/y along its faces), so the grasp and the placeholder cuboid line up.

Example:
    python scripts/ur_cube/track_to_base_traj.py \
        --track outputs/table_new/track/keyframes.txt \
        --state outputs/table_new/state_rigid_20261102_190817.pt \
        --intr demonstrations/cube_pnp_ep0_front_f1_to_f0_intr.json \
        --output outputs/ur_cube/traj_ep0_f1_to_f0.json
"""

import json
from pathlib import Path

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation, Slerp

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def wxyz_xyz_to_mat(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([v[1], v[2], v[3], v[0]]).as_matrix()
    T[:3, 3] = v[4:]
    return T


def mat_to_wxyz_xyz(T: np.ndarray) -> list:
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [w, x, y, z, *T[:3, 3]]


def main(
    track: Path,
    state: Path,
    intr: Path,
    output: Path,
    scan_to_base: Path = Path("outputs/table_new/scan_to_base.json"),
    sim_steps_per_second: float = 30.0,
    static_speed: float = 0.02,  # m/s; below this the cube counts as resting
    keep_static_s: float = 0.15,  # keep a little rest before lift-off and after set-down
    grasp_height_above_center: float = 0.01,
    gripper_close: float = 0.42,
) -> None:
    data = json.loads(track.read_text())
    deltas = np.asarray(data["part_deltas"])[:, 0]  # (T, 7) wxyz_xyz, single rigid group
    T_wo = wxyz_xyz_to_mat(np.asarray(data["T_world_objinit"]).reshape(-1, 7)[0])
    T_ro = wxyz_xyz_to_mat(np.asarray(data["T_objreg_objinit"]).reshape(-1, 7)[0])
    s = json.loads(scan_to_base.read_text())["scale_m_per_model_unit"]
    cam = json.loads(intr.read_text())
    T_base_cam = np.asarray(cam["T_cam2base"])
    fps = float(cam["fps"])

    poses = []
    for d in deltas:
        T = T_wo @ T_ro @ wxyz_xyz_to_mat(d)
        T[:3, 3] *= s
        poses.append(T_base_cam @ GL_TO_CV @ T)
    poses = np.stack(poses)

    # Re-base the object frame onto the cube's axes: z = most-upward object axis at t0,
    # x/y = face directions from the Gaussians' footprint (min-area box in the object frame).
    g = torch.load(state, map_location="cpu")["means"].detach().numpy().astype(np.float64)
    g_obj = (g - T_wo[:3, 3]) * s  # object frame (identity rotation), metres
    R0 = poses[0][:3, :3]
    up_axis = int(np.argmax(np.abs(R0.T @ [0, 0, 1])))
    sign = np.sign((R0.T @ [0, 0, 1])[up_axis])
    others = [i for i in range(3) if i != up_axis]
    best = None
    for ang in np.radians(np.arange(0, 90, 1.0)):
        c, sn = np.cos(ang), np.sin(ang)
        a = c * g_obj[:, others[0]] + sn * g_obj[:, others[1]]
        b = -sn * g_obj[:, others[0]] + c * g_obj[:, others[1]]
        area = np.ptp(np.percentile(a, [3, 97])) * np.ptp(np.percentile(b, [3, 97]))
        if best is None or area < best[0]:
            best = (area, ang)
    ang = best[1]
    ez = np.zeros(3); ez[up_axis] = sign
    ex = np.zeros(3); ex[others[0]], ex[others[1]] = np.cos(ang), np.sin(ang)
    ex -= ex.dot(ez) * ez; ex /= np.linalg.norm(ex)
    T_obj_cube = np.eye(4)
    T_obj_cube[:3, :3] = np.stack([ex, np.cross(ez, ex), ez], axis=1)
    T_obj_cube[:3, 3] = np.median(g_obj, axis=0)  # centroid -> robust centre
    poses = poses @ T_obj_cube
    size = [float(np.ptp(np.percentile(g_obj @ T_obj_cube[:3, i], [3, 97]))) for i in range(3)]

    # Trim to the moving part (+ a short rest at both ends).
    pos = poses[:, :3, 3]
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * fps
    moving = np.where(speed > static_speed)[0]
    pad = int(round(keep_static_s * fps))
    i0, i1 = max(moving[0] - pad, 0), min(moving[-1] + pad, len(poses) - 1)
    seg = poses[i0 : i1 + 1]

    # Resample from video rate to the simulator step rate.
    t_src = np.arange(len(seg)) / fps
    t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / sim_steps_per_second)
    rots = Slerp(t_src, Rotation.from_matrix(seg[:, :3, :3]))(t_dst).as_matrix()
    trans = np.stack([np.interp(t_dst, t_src, seg[:, k, 3]) for k in range(3)], axis=1)
    out_poses = np.tile(np.eye(4), (len(t_dst), 1, 1))
    out_poses[:, :3, :3], out_poses[:, :3, 3] = rots, trans

    # Top-down grasp: TCP z down, fingers closing along the cube's x faces.
    T_obj_tcp = np.eye(4)
    T_obj_tcp[:3, :3] = Rotation.from_euler("x", np.pi).as_matrix()
    T_obj_tcp[:3, 3] = [0, 0, grasp_height_above_center]

    out = {
        "T_base_obj": [mat_to_wxyz_xyz(T) for T in out_poses],
        "grasp_T_obj_tcp": mat_to_wxyz_xyz(T_obj_tcp),
        "gripper_close": gripper_close,
        "source": f"RSRD track {track} frames {i0}-{i1} of {len(poses)} @ {fps} fps, resampled to {sim_steps_per_second} Hz",
        "cube_size_m": size,
        "demo_frames": [int(i0), int(i1)],
        # RSRD object frame (scan axes, at the gaussian centroid, metres) -> cube frame used above;
        # mesh_to_object_frame.py expresses the SuGaR mesh in this same frame.
        "T_objinit_cube": T_obj_cube.tolist(),
        "objinit_centroid_model": T_wo[:3, 3].tolist(),
        "m_per_model_unit": s,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out))
    print(f"track: {len(poses)} keyframes; moving {moving[0]}-{moving[-1]}, kept {i0}-{i1} -> {len(out_poses)} sim steps")
    print(f"cube extent from gaussians (x, y, z): {np.round(size, 4)} m")
    print(f"start {pos[0].round(4)}  end {pos[-1].round(4)}  max height {pos[:, 2].max():.3f} m, peak speed {speed.max():.2f} m/s")
    np.save(output.with_suffix(".poses_all.npy"), poses)  # full-length, video-rate, for overlays
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
