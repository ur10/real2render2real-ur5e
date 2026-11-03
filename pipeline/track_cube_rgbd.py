"""Track the cube through a demo with both calibrated RealSense cameras (replaces RSRD here).

RSRD's DINO-feature tracking slides off this small black cube onto the nearby fixture. With two
calibrated RGB-D views we can track it geometrically instead, per capture step (10 Hz):
  1. fuse both cameras' points in the robot base frame,
  2. keep cube candidates: dark, above the table, not skin, near a constant-velocity prediction,
  3. ICP the SuGaR cube mesh (cube frame, metres) onto them from the prediction.
When the hand hides most of the cube, the prediction is kept. The first pose comes from the
calibrated RSRD init (track_to_base_traj.py output), which both cameras confirm.

Output: <output>.poses_all.npy (T, 4, 4) base-frame cube poses at 10 Hz plus a simulator
trajectory json (same format as make_synthetic_traj.py).

Example:
    python scripts/ur_cube/track_cube_rgbd.py --init-traj outputs/ur_cube/traj_ep0_f1_to_f0.json \
        --mesh data/ur_cube/cube_mesh.obj --episode 0 --steps 52 85 \
        --output outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import tyro
from scipy.spatial.transform import Rotation, Slerp

from rgbd_utils import backproject, load_frame

reg = o3d.pipelines.registration


def candidates(frame: dict, table_z: float, centre: np.ndarray, radius: float) -> np.ndarray:
    pts, cols = [], []
    for cam in frame["cameras"].values():
        p, c, _ = backproject(cam, depth_range=(0.2, 1.6))
        pts.append(p)
        cols.append(c)
    p, c = np.concatenate(pts), np.concatenate(cols).astype(np.int32)
    near = np.linalg.norm(p - centre, axis=1) < radius
    p, c = p[near], c[near]
    r, g, b = c.T
    dark = c.max(1) < 70
    skin = (r > 95) & (g > 40) & (b > 20) & (r - g > 15) & (r > b)
    keep = dark & ~skin & (p[:, 2] > table_z + 0.004)
    return p[keep]


def cube_symmetries() -> np.ndarray:
    """The 24 rotations mapping a cube onto itself (signed permutation matrices with det +1)."""
    mats = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.ndindex(2, 2, 2):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[i, p] = -1.0 if s else 1.0
            if np.linalg.det(M) > 0:
                mats.append(M)
    return np.stack(mats)


SYMMETRIES = cube_symmetries()


def closest_symmetric(R_new: np.ndarray, R_prev: np.ndarray) -> np.ndarray:
    """ICP can lock onto any face assignment of a cube; keep the one nearest the previous pose."""
    cands = R_new @ SYMMETRIES  # (24, 3, 3): same cube, relabelled axes
    cos = (np.trace(np.swapaxes(cands, 1, 2) @ R_prev, axis1=1, axis2=2) - 1) / 2
    return cands[int(np.argmax(cos))]


def smooth_poses(poses: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of positions and (sign-aligned, renormalised) quaternions over time."""
    if sigma <= 0:
        return poses
    n = len(poses)
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    for i in range(1, n):  # keep quaternions on one hemisphere so averaging is meaningful
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    t = np.arange(n)
    W = np.exp(-0.5 * ((t[:, None] - t[None, :]) / sigma) ** 2)
    W /= W.sum(1, keepdims=True)
    out = poses.copy()
    out[:, :3, 3] = W @ poses[:, :3, 3]
    qs = W @ q
    out[:, :3, :3] = Rotation.from_quat(qs / np.linalg.norm(qs, axis=1, keepdims=True)).as_matrix()
    return out


def redetect(frame: dict, table_z: float, pred_pos: np.ndarray, max_extent: float = 0.07):
    """Cube-sized dark clusters anywhere in the workspace; returns the one nearest the prediction."""
    obs = candidates(frame, table_z, pred_pos, radius=0.45)
    if len(obs) < 40:
        return None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs))
    labels = np.asarray(pc.cluster_dbscan(eps=0.012, min_points=30))
    best = None
    for k in set(labels[labels >= 0].tolist()):
        c = obs[labels == k]
        ext = np.percentile(c, 97, axis=0) - np.percentile(c, 3, axis=0)
        if ext.max() > max_extent:  # sleeve / arm, not the cube
            continue
        d = np.linalg.norm(np.median(c, axis=0) - pred_pos)
        if best is None or d < best[0]:
            best = (d, c)
    return None if best is None else best[1]


def main(
    init_traj: Path,
    mesh: Path,
    episode: int,
    steps: tuple[int, int],
    output: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    layout: Path = Path("outputs/ur_cube/scene_layout.json"),
    search_radius: float = 0.06,
    min_points: int = 40,
    capture_hz: float = 10.0,
    smooth_sigma: float = 1.0,  # capture steps; removes single-frame ICP jitter
    sim_steps_per_second: float = 30.0,
    static_speed: float = 0.02,
    keep_static_s: float = 0.2,
    grasp_height_above_center: float = 0.01,
    gripper_close: float = 0.42,
) -> None:
    table_z = json.loads(layout.read_text())["table_z"]
    model = trimesh.load(mesh, process=False)
    model_pts = model.sample(3000)
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model_pts))
    src.estimate_normals()

    T = np.load(Path(init_traj).with_suffix(".poses_all.npy"))[0]  # calibrated first pose
    poses, n_obs, fitness = [T.copy()], [], []
    vel = np.zeros(3)
    for step in range(steps[0], steps[1] + 1):
        frame = load_frame(capture_dir, episode, step)
        pred = T.copy()
        pred[:3, 3] += vel
        obs = candidates(frame, table_z, pred[:3, 3], search_radius + np.linalg.norm(vel))
        if len(obs) < min_points:  # lost (fast lift / occlusion): search the whole workspace
            found = redetect(frame, table_z, pred[:3, 3])
            if found is not None:
                obs = found
                pred[:3, 3] = np.median(found, axis=0)
        n_obs.append(len(obs))
        if len(obs) >= min_points:
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs))
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
            Tn = pred
            for d in (0.03, 0.015, 0.008):
                r = reg.registration_icp(src, tgt, d, Tn, reg.TransformationEstimationPointToPoint())
                Tn = r.transformation
            # Visible faces only: accept if enough of the observed points are explained.
            back = reg.evaluate_registration(tgt, src, 0.008, np.linalg.inv(Tn))
            fitness.append(back.fitness)
            Tn = Tn.copy()
            Tn[:3, :3] = closest_symmetric(Tn[:3, :3], T[:3, :3])
            if back.fitness > 0.5:
                vel = 0.5 * vel + 0.5 * (Tn[:3, 3] - T[:3, 3])
                T = Tn
            else:
                T = pred
        else:
            fitness.append(0.0)
            T = pred
            vel *= 0.5  # occluded: coast, then settle
        if step > steps[0]:
            poses.append(T.copy())
        else:
            poses[0] = T.copy()
    poses = smooth_poses(np.stack(poses), smooth_sigma)
    pos = poses[:, :3, 3]
    print("step  obs  fit   x      y      z")
    for i, (s, n, f, p) in enumerate(zip(range(steps[0], steps[1] + 1), n_obs, fitness, pos)):
        print(f"{s:4d} {n:5d} {f:4.2f} {p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}")

    np.save(output.with_suffix(".poses_all.npy"), poses)

    # Trim to the motion, resample to the simulator rate, write the trajectory json.
    speed = np.linalg.norm(np.gradient(pos, axis=0), axis=1) * capture_hz
    moving = np.where(speed > static_speed)[0]
    pad = int(round(keep_static_s * capture_hz))
    i0, i1 = max(moving[0] - pad, 0), min(moving[-1] + pad, len(poses) - 1)
    seg = poses[i0 : i1 + 1]
    t_src = np.arange(len(seg)) / capture_hz
    t_dst = np.arange(0, t_src[-1] + 1e-9, 1.0 / sim_steps_per_second)
    rots = Slerp(t_src, Rotation.from_matrix(seg[:, :3, :3]))(t_dst)
    trans = np.stack([np.interp(t_dst, t_src, seg[:, k, 3]) for k in range(3)], axis=1)
    quats = rots.as_quat()  # xyzw
    traj = np.concatenate([quats[:, [3, 0, 1, 2]], trans], axis=1)
    init = json.loads(Path(init_traj).read_text())
    out = {
        "T_base_obj": traj.tolist(),
        "grasp_T_obj_tcp": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, grasp_height_above_center],
        "gripper_close": gripper_close,
        "source": f"RGB-D ICP track, ep{episode} steps {steps[0] + i0}-{steps[0] + i1} @ {capture_hz} Hz, resampled to {sim_steps_per_second} Hz",
        "cube_size_m": [float(x) for x in (model.bounds[1] - model.bounds[0])],
        "demo_frames": [int(i0), int(i1)],
        "T_objinit_cube": init["T_objinit_cube"],
        "objinit_centroid_model": init["objinit_centroid_model"],
        "m_per_model_unit": init["m_per_model_unit"],
    }
    output.write_text(json.dumps(out))
    print(f"moving steps {steps[0] + moving[0]}-{steps[0] + moving[-1]}; start {pos[0].round(3)} end {pos[-1].round(3)} "
          f"max z {pos[:, 2].max():.3f}; {len(traj)} sim steps; wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
