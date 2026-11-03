"""Check the UR5e + 2F-85 URDF (incl. gripper mount) against the RealSense capture.

The capture stores only the controller-reported TCP pose (robot0_eef_pos / robot0_eef_quat, which
is a rotation vector; export it with export_capture_robot_state.py). We solve IK for
`--pose-link` at that pose from many seeds, keep the
solution whose surface points best agree with the fused RealSense depth (base frame), and draw
the robot's mesh points over both camera images.

Example:
    python scripts/ur_cube/check_robot_overlay.py --urdf data/ur5e_robotiq/ur5e_robotiq85.urdf \
        --robot-state outputs/ur_cube/capture_robot_state.json --output-dir /tmp/overlay
"""

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh
import tyro
import yourdfpy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rgbd_utils import backproject, load_frame, project

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


def load_urdf(path: Path) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        path, filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=path.parent)
    )


def solve_ik(urdf, link: str, T_target: np.ndarray, seeds: int = 64, rng=np.random.default_rng(0)):
    """All distinct converged arm solutions for `link` at T_target (base frame)."""
    lo = np.array([urdf.joint_map[j].limit.lower for j in ARM_JOINTS])
    hi = np.array([urdf.joint_map[j].limit.upper for j in ARM_JOINTS])

    def residual(q):
        urdf.update_cfg(dict(zip(ARM_JOINTS, q)))
        T = urdf.get_transform(link, urdf.base_link)
        rot_err = Rotation.from_matrix(T_target[:3, :3].T @ T[:3, :3]).as_rotvec()
        return np.concatenate([T[:3, 3] - T_target[:3, 3], 0.1 * rot_err])

    sols = []
    for _ in range(seeds):
        q0 = rng.uniform(np.maximum(lo, -np.pi), np.minimum(hi, np.pi))
        r = least_squares(residual, q0, bounds=(lo, hi))
        if np.linalg.norm(r.fun) < 1e-4 and not any(np.allclose(r.x, s, atol=1e-2) for s in sols):
            sols.append(r.x)
    return sols


def surface_points(urdf, n_per_link: int = 400) -> np.ndarray:
    """Sample points on every visual mesh at the URDF's current configuration (base frame)."""
    pts = []
    scene = urdf.scene
    for node in scene.graph.nodes_geometry:
        T, geom_name = scene.graph.get(node)
        mesh = scene.geometry[geom_name]
        if isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            p, _ = trimesh.sample.sample_surface(mesh, n_per_link, seed=0)
            pts.append(p @ T[:3, :3].T + T[:3, 3])
    return np.concatenate(pts)


def main(
    urdf: Path,
    robot_state: Path,
    output_dir: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    pose_link: str = "tool0",
) -> None:
    """
    Args:
        pose_link: URDF link the controller-reported TCP pose refers to (tool0 if the
            controller's TCP offset is zero).
    """
    state = json.loads(robot_state.read_text())["episodes"][episode]
    pos, rotvec = np.array(state["tcp_pos"]), np.array(state["tcp_rotvec"])
    T_target = np.eye(4)
    T_target[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T_target[:3, 3] = pos
    print(f"reported TCP pose: pos {np.round(pos, 4)}, rotvec {np.round(rotvec, 4)}")

    robot = load_urdf(urdf)
    sols = solve_ik(robot, pose_link, T_target)
    print(f"{len(sols)} IK solutions")

    meta = load_frame(capture_dir, episode, 0)
    rs = np.concatenate([backproject(c, stride=2)[0] for c in meta["cameras"].values()])
    rs_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rs))

    best = None
    for q in sols:
        robot.update_cfg(dict(zip(ARM_JOINTS, q)))
        pts = surface_points(robot)
        d = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).compute_point_cloud_distance(rs_pc))
        score = np.median(d)
        print(f"  q={np.round(q, 3)}  median dist to depth {score * 1000:.1f} mm")
        if best is None or score < best[0]:
            best = (score, q)
    score, q = best
    robot.update_cfg(dict(zip(ARM_JOINTS, q)))
    print(f"best: q={np.round(q, 4).tolist()} ({score * 1000:.1f} mm median)")

    output_dir.mkdir(parents=True, exist_ok=True)
    pts = surface_points(robot, 1500)
    tiles = []
    for name, cam in meta["cameras"].items():
        img = cam["color"].copy()
        uv, depth = project(cam, pts)
        for u, v in uv[depth > 0].astype(int):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                img[v, u] = (255, 0, 255)
        uv, _ = project(cam, robot.get_transform("tcp", robot.base_link)[None, :3, 3])
        cv2.drawMarker(img, tuple(uv[0].astype(int)), (0, 255, 255), cv2.MARKER_CROSS, 25, 2)
        tiles.append(img)
    out = output_dir / f"robot_overlay_ep{episode}.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))
    (output_dir / f"robot_joints_ep{episode}.json").write_text(
        json.dumps({"joint_names": ARM_JOINTS, "q": q.tolist(), "median_depth_dist_m": float(score)}, indent=2)
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    tyro.cli(main)
