"""Closed-loop evaluation of a trained policy (scripts/ur_cube/train_policy.py) in the UR5e cube twin.

    OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_ur5e_cube_policy.py \
        --policy outputs/ur_cube/policies/r2r2r_400/policy.pt --set blocked \
        --scenes outputs/ur_cube/eval/scenes.json --output-dir outputs/ur_cube/eval/r2r2r_400_blocked

Writes <output-dir>/results.jsonl (one line per scene) and summary.json. See
real2render2real/isaaclab_viser/ur5e_simulators/ur5e_cube_eval.py for the scoring.
Run with conda deactivated and without CUDA_VISIBLE_DEVICES (see run_ur5e_cube.py).
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=Path, required=True)
parser.add_argument("--scenes", type=Path, default=Path("outputs/ur_cube/eval/scenes.json"))
parser.add_argument("--set", choices=["clean", "blocked"], required=True)
parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N scenes")
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--traj", type=Path, default=Path("outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json"))
parser.add_argument("--obstacle-asset", default="data/obstacles/sauce_bottle/obstacle.json")
parser.add_argument("--exec-steps", type=int, default=8)
parser.add_argument("--max-steps", type=int, default=180)
parser.add_argument("--save-video-envs", type=int, default=0, help="save camera_0 frames of the first batch's first N envs")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
simulation_app = AppLauncher(args).app

# JAX (IK controller, planner) would otherwise open a context on every GPU and fail when another job
# holds the other one ("Failed to initialize BLAS support"): keep it on the simulation's GPU.
import jax  # noqa: E402
jax.config.update("jax_cuda_visible_devices", args.device.split(":")[-1] if args.device.startswith("cuda:") else "0")

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import (  # noqa: E402
    UR5eCubePnPObstacleCfg, obstacle_spawn_from_asset,
)
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_cube_eval import UR5eCubeEval  # noqa: E402

import os  # noqa: E402
import traceback  # noqa: E402

repo = Path(__file__).resolve().parents[1]
scenes = json.loads(args.scenes.read_text())[args.set][: args.limit]
meta_file = str(repo / args.obstacle_asset)
meta = json.load(open(meta_file))
scene_cfg = UR5eCubePnPObstacleCfg(num_envs=args.num_envs, env_spacing=100.0)
scene_cfg.obstacle.spawn = obstacle_spawn_from_asset(meta_file)
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / "results.jsonl").unlink(missing_ok=True)
code = 0
try:
    UR5eCubeEval(
        simulation_app, scene_cfg,
        urdf_path={"robot": repo / "data/ur5e_robotiq/ur5e_robotiq85.urdf"},
        save_data=False, output_dir=str(args.output_dir.resolve()), traj_path=args.traj,
        obstacle=dict(radius=meta["radius"], height=meta["height"], origin=meta["origin"]),
        policy_path=args.policy.resolve(), scenes=scenes, exec_steps=args.exec_steps,
        max_steps=args.max_steps, save_video_envs=args.save_video_envs,
    )
except Exception:
    traceback.print_exc()
    code = 1
finally:
    os._exit(code)  # simulation_app.close() can hang after a run
