"""Generate UR5e + Robotiq 2F-85 demonstrations for the ring tasks (headless): hanging the tape roll
on the mug tree (--task tape) or putting the green nut on the bolt stand (--task nut), from the
tracked object trajectory (scripts/ur_tape/track_tape_rgb.py + clean_tape_track.py / clean_nut_track.py).

    scripts/ur_cube/isaac_run.sh outputs/ur_tape/logs/gen.log "^\\[INFO\\]: Episode 6:" scripts/run_ur5e_tape.py \
        --traj outputs/ur_tape/traj_ep1_hang_clean.json --num-envs 4 --max-episodes 6 --output-dir output_data/tape/r2r2r

Run with conda deactivated and without CUDA_VISIBLE_DEVICES (see run_ur5e_cube.py).
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--max-successes", type=int, default=100000)
parser.add_argument("--max-episodes", type=int, default=None)
parser.add_argument("--output-dir", type=Path, default=Path("output_data/tape/r2r2r"))
parser.add_argument("--no-save", action="store_true")
parser.add_argument("--no-augment", action="store_true", help="replay the tracked trajectory exactly")
parser.add_argument("--save-scale", type=int, default=2)
parser.add_argument("--task", choices=["tape", "nut"], default="tape")
parser.add_argument("--start-offset", type=float, nargs=2, default=(0.0, 0.0),
                    help="shift every sampled start (m), e.g. into reach")
parser.add_argument("--tree-yaw", type=float, nargs=2, default=(250.0, 250.0),
                    help="range for turning the mug tree about its post (deg); see sweep_tree_yaw.py")
parser.add_argument("--obstacle-xy", type=float, nargs=2, default=None,
                    help="place the scanned bottle here (base frame) and plan with ring_repair.py")
parser.add_argument("--no-repair", action="store_true", help="with --obstacle-xy: reject colliding envs instead of repairing")
parser.add_argument("--compare", action="store_true",
                    help="figures: env 0 replays the plain generator's motion next to the repaired one (--num-envs 2 --no-augment)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
simulation_app = AppLauncher(args).app

import jax  # noqa: E402
jax.config.update("jax_cuda_visible_devices", args.device.split(":")[-1] if args.device.startswith("cuda:") else "0")

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_tape_scene_cfg import (  # noqa: E402
    UR5eNutCfg, UR5eNutObstacleCfg, UR5eTapeCfg, UR5eTapeObstacleCfg,
)
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_tape_pnp import UR5eTapePnP  # noqa: E402

import os  # noqa: E402
import traceback  # noqa: E402

repo = Path(__file__).resolve().parents[1]
if args.task == "nut":
    cfg_cls, task_kw = UR5eNutCfg, dict(
        tree_yaw_range=(0.0, 0.0), start_offset_xy=tuple(args.start_offset),
        planner_kwargs=dict(scene_file=repo / "outputs/ur_nut/scene.json", dims_file=repo / "outputs/ur_nut/nut_dims.json",
                            grasp_style="outer", tcp_above=0.017))
else:
    cfg_cls, task_kw = UR5eTapeCfg, dict(tree_yaw_range=tuple(args.tree_yaw), start_offset_xy=tuple(args.start_offset))
task_kw["task"] = args.task
if args.obstacle_xy is not None:
    import json as _json
    cfg_cls = UR5eNutObstacleCfg if args.task == "nut" else UR5eTapeObstacleCfg
    meta = _json.load(open(repo / "data/obstacles/sauce_bottle/obstacle.json"))
    task_kw.update(obstacle=dict(radius=meta["radius"], height=meta["height"], origin=meta["origin"],
                                 xy=tuple(args.obstacle_xy), compare=args.compare), repair=not args.no_repair)
sim, code = None, 0
try:
    sim = UR5eTapePnP(
        simulation_app,
        cfg_cls(num_envs=args.num_envs, env_spacing=100.0),
        urdf_path={"robot": repo / "data/ur5e_robotiq/ur5e_robotiq85.urdf"},
        save_data=not args.no_save,
        output_dir=str(args.output_dir.resolve()),
        traj_path=args.traj,
        max_successes=args.max_successes,
        max_episodes=args.max_episodes,
        **({"start_xy_noise": 0.0, "start_yaw_noise": 0.0, "randomize_look": False} if args.no_augment else {}),
        save_scale=args.save_scale,
        **task_kw,
    )
except Exception:
    traceback.print_exc()
    code = 1
finally:
    if sim is not None and sim.data_logger is not None:
        sim.data_logger.shutdown()
    os._exit(code)
