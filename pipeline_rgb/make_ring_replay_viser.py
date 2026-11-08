"""Interactive 3D replay of a generated tape / nut demonstration (the counterpart of the cube's
make_viser_recording.py, same viser version and layout): the robot URDF animated from the rollout's
joints, the measured object following its logged poses, the measured fixture (the mug tree turned as
in the twin), the scanned bench, and both RealSense frustums showing the rendered views.

    uv run --no-project --with viser==0.2.23 --with yourdfpy --with pycollada --with h5py --with trimesh --with tyro \
        python scripts/ur_tape/make_ring_replay_viser.py --task tape \
        --rollout output_data/tape/replay/successes/env_0_* --output outputs/ur_tape/viser/tape_replay.viser
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import tyro
import viser
import yourdfpy
from PIL import Image
from viser.extras import ViserUrdf

sys.path.insert(0, str(Path(__file__).parent))
from ring_viser import REPO, add_bench, add_fixture, add_object, TASKS  # noqa: E402

CAMERA_NAMES = ("front", "right")


def load_jpg(path: Path, width: int) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im.resize((width, int(im.height * width / im.width))))


def main(task: str, rollout: Path, output: Path, splats: Path = REPO / "outputs/ur_cube/splats_base.npz",
         urdf: Path = REPO / "data/ur5e_robotiq/ur5e_robotiq85.urdf",
         camera_poses: Path = REPO / "data/camera_poses/ur_realsense_front_right.h5",
         intrinsics: Path = REPO / "data/camera_poses/ur_realsense_intrinsics.json",
         fps: float = 30.0, frustum_image_width: int = 320, still_step: int | None = None) -> None:
    """still_step: write a single-frame recording at that step instead (for the poster image)."""
    sp = np.load(splats)
    intr = json.loads(intrinsics.read_text())
    with h5py.File(camera_poses, "r") as f:
        cam_poses = f["poses"][:]
    with h5py.File(rollout / "robot_data/robot_data.h5", "r") as f:
        joints = f["joint_angles"][:]
    obj = np.load(rollout / "robot_data/object_poses.npy")
    plans = [json.loads(line) for line in (rollout.parents[1] / "plans.jsonl").read_text().splitlines()]
    env = int(rollout.name.split("_")[1])
    yaw = next((p.get("tree_yaw_deg", 0.0) for p in plans if p["env"] == env), 0.0)
    n = min(len(joints), len(obj), *(len(list((rollout / f"camera_{i}/rgb").glob("*.jpg"))) for i in range(2)))

    server = viser.ViserServer(port=8158)
    rec = server.get_scene_serializer()
    add_bench(server, sp, TASKS[task]["centre"], radius=1.2)
    add_fixture(server, task, yaw)
    o = add_object(server, task, "/object", obj[0])
    robot = ViserUrdf(server, yourdfpy.URDF.load(
        urdf, filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=urdf.parent)), root_node_name="/robot")
    order = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
             "wrist_3_joint", "finger_joint"]
    idx = [order.index(j) for j in robot.get_actuated_joint_names()]
    fov = 2 * np.arctan(intr["height"] / 2 / intr["shared_K"][4])
    frustums = []
    for i, name in enumerate(CAMERA_NAMES):
        p = cam_poses[i]
        frustums.append(server.scene.add_camera_frustum(
            f"/realsense_{name}", fov=fov, aspect=intr["width"] / intr["height"], scale=0.05, wxyz=p[3:], position=p[:3],
            color=(255, 0, 200), image=load_jpg(rollout / f"camera_{i}/rgb/0000.jpg", frustum_image_width), format="jpeg"))
    steps = list(range(n)) if still_step is None else [still_step] * 3
    for k, t in enumerate(steps):
        robot.update_cfg(joints[t][idx])
        if k == 0:  # force a pose message on frame 0 (viser skips unchanged setters; see make_viser_recording.py)
            o.wxyz, o.position = -obj[t, :4], obj[t, 4:] + 1e-4
        o.wxyz, o.position = obj[t, :4], obj[t, 4:]
        for i, fr in enumerate(frustums):
            fr.image = load_jpg(rollout / f"camera_{i}/rgb/{t:04d}.jpg", frustum_image_width)
        rec.insert_sleep(1.0 / fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rec.serialize())
    server.stop()
    print(f"{task}: {len(steps)} frames, tree yaw {yaw} -> {output} ({output.stat().st_size / 1e6:.1f} MB); "
          f"camera {TASKS[task]['camera']}")


if __name__ == "__main__":
    tyro.cli(main)
