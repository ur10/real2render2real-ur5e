"""Spawn the UR5e bench scene headless, render both RealSense views, and blend with the real frames.

Checks cameras (poses + intrinsics), robot model/home pose and table/fixture layout in one image.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/ur_cube/test_ur5e_scene.py --output-dir /tmp/scene_check
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--capture-dir", default="~/ur_human_capture")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import (  # noqa: E402
    UR_CAMERA_POSES_FILE, INTR, UR5eCubePnPCfg,
)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))  # must match the app's device
scene = InteractiveScene(UR5eCubePnPCfg(num_envs=1, env_spacing=3.0))
sim.reset()

robot = scene.articulations["robot"]
print("joint names:", robot.joint_names)
print("body names:", robot.body_names)

with h5py.File(UR_CAMERA_POSES_FILE, "r") as f:
    poses = torch.tensor(f["poses"][:], dtype=torch.float32)
cam = scene.sensors["viewport_camera"]
for _ in range(30):
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    cam.set_world_poses(poses[:, :3] + scene.env_origins[0].cpu(), poses[:, 3:], convention="ros")
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(sim.get_physics_dt())

os.makedirs(args.output_dir, exist_ok=True)
rgb = cam.data.output["rgb"].cpu().numpy()  # (cams, H, W, 3)
meta = json.load(open(os.path.expanduser(f"{args.capture_dir}/rgbd/0/000000/meta.json")))
for i, name in enumerate(INTR["camera_order"]):
    c = meta["cameras"][f"realsense_{name}"]
    ci = c["intrinsics"]
    K = np.array([[ci["fx"], 0, ci["ppx"]], [0, ci["fy"], ci["ppy"]], [0, 0, 1]])
    real = cv2.cvtColor(cv2.imread(os.path.expanduser(f"{args.capture_dir}/rgbd/0/000000/{c['files']['color']}")), cv2.COLOR_BGR2RGB)
    real = cv2.undistort(real, K, np.array(ci["coeffs"]))
    render = rgb[i][..., :3].astype(np.uint8)
    edges = cv2.Canny(cv2.cvtColor(render, cv2.COLOR_RGB2GRAY), 40, 120)
    overlay = real.copy()
    overlay[edges > 0] = (255, 0, 255)
    out = np.vstack([np.hstack([real, render]), np.hstack([(0.5 * real + 0.5 * render).astype(np.uint8), overlay])])
    path = os.path.join(args.output_dir, f"scene_{name}.jpg")
    cv2.imwrite(path, cv2.cvtColor(cv2.resize(out, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
    print("wrote", path)
app.close()
