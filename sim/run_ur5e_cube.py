"""Generate UR5e + Robotiq 2F-85 cube pick-and-place demonstrations (headless).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/run_ur5e_cube.py \
        --traj outputs/ur_cube/traj_ep0_f1_to_f0.json --num-envs 4 --max-successes 50 \
        --output-dir output_data/ur5e_cube_pnp

Obstacle insertion + demo repair (scripts/ur_cube/demo_repair.py):

    ... --obstacle                 # scanned bottle on the route, colliding motion repaired
    ... --obstacle --no-repair     # same scenes, colliding envs rejected (plain generator)
    ... --obstacle --obstacle-asset cylinder --obstacle-radius 0.035 --obstacle-height 0.26
    ... --obstacle --obstacle-xy 0.10 -0.76 --compare --no-augment --num-envs 2   # before/after figure

Run with conda deactivated (isaaclab.sh/Kit pick up CONDA_PREFIX) and without CUDA_VISIBLE_DEVICES
(Omniverse and CUDA enumerate GPUs differently); the simulation runs on cuda:0.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--traj", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--max-successes", type=int, default=50)
parser.add_argument("--output-dir", type=Path, default=Path("output_data/ur5e_cube_pnp"))
parser.add_argument("--no-save", action="store_true", help="dry run: no images or robot data written")
parser.add_argument("--max-episodes", type=int, default=None, help="stop after this many resets")
parser.add_argument("--no-augment", action="store_true", help="replay the demo trajectory exactly (no start jitter)")
parser.add_argument("--wait-for-client", action="store_true", help="block until a viser client connects")
parser.add_argument("--obstacle", action="store_true", help="insert an obstacle on the transfer route per env")
parser.add_argument("--obstacle-asset", default="data/obstacles/sauce_bottle/obstacle.json",
                    help="obstacle.json from prep_gso_obstacle.py, or 'cylinder' for a plain cylinder")
parser.add_argument("--obstacle-radius", type=float, default=0.035, help="cylinder only")
parser.add_argument("--obstacle-height", type=float, default=0.26, help="cylinder only")
parser.add_argument("--obstacle-fraction", type=float, nargs=2, default=(0.3, 0.7), help="range along the cube route")
parser.add_argument("--obstacle-lateral", type=float, nargs=2, default=(-0.06, 0.06), help="range across it (m)")
parser.add_argument("--no-repair", action="store_true", help="with --obstacle: reject colliding envs instead of repairing")
parser.add_argument("--obstacle-xy", type=float, nargs=2, default=None, help="fixed obstacle position (base frame)")
parser.add_argument("--obstacle-prob", type=float, default=1.0, help="chance that an env gets the obstacle")
parser.add_argument("--reject-resample", action="store_true",
                    help="reject-only baseline: redraw each placement until the unrepaired motion is collision-free")
parser.add_argument("--save-scale", type=int, default=1, help="save images downscaled by this factor (2: 640x360)")
parser.add_argument("--compare", action="store_true",
                    help="figures: env 0 replays the plain generator's motion (unfiltered) next to the repaired one; "
                         "use with --obstacle-xy --no-augment --num-envs 2")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
simulation_app = AppLauncher(args).app

# JAX (IK controller, planner) would otherwise open a context on every GPU and fail when another job
# holds the other one ("Failed to initialize BLAS support"): keep it on the simulation's GPU.
import jax  # noqa: E402
jax.config.update("jax_cuda_visible_devices", args.device.split(":")[-1] if args.device.startswith("cuda:") else "0")

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import (  # noqa: E402
    UR5eCubePnPCfg, UR5eCubePnPObstacleCfg, obstacle_spawn_from_asset,
)
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_cube_pnp import UR5eCubePnP  # noqa: E402

import json  # noqa: E402
import os  # noqa: E402
import traceback  # noqa: E402

repo = Path(__file__).resolve().parents[1]
scene_cfg = (UR5eCubePnPObstacleCfg if args.obstacle else UR5eCubePnPCfg)(num_envs=args.num_envs, env_spacing=100.0)
obstacle = None
if args.obstacle:
    if args.obstacle_asset == "cylinder":
        scene_cfg.obstacle.spawn.radius, scene_cfg.obstacle.spawn.height = args.obstacle_radius, args.obstacle_height
        obstacle = dict(radius=args.obstacle_radius, height=args.obstacle_height, origin="center")
    else:  # collision proxy (cylinder containing the object) from the asset's json
        meta_file = str(repo / args.obstacle_asset) if not os.path.isabs(args.obstacle_asset) else args.obstacle_asset
        meta = json.load(open(meta_file))
        scene_cfg.obstacle.spawn = obstacle_spawn_from_asset(meta_file)
        obstacle = dict(radius=meta["radius"], height=meta["height"], origin=meta["origin"])
    obstacle.update(fraction=tuple(args.obstacle_fraction), lateral=tuple(args.obstacle_lateral),
                    xy=args.obstacle_xy, compare=args.compare, prob=args.obstacle_prob,
                    resample_until_valid=args.reject_resample)
sim, code = None, 0
try:
    sim = UR5eCubePnP(
        simulation_app,
        scene_cfg,  # envs 100 m apart: other envs must not show up in the renders
        urdf_path={"robot": repo / "data/ur5e_robotiq/ur5e_robotiq85.urdf"},
        save_data=not args.no_save,
        output_dir=str(args.output_dir.resolve()),
        traj_path=args.traj,
        max_successes=args.max_successes,
        max_episodes=args.max_episodes,
        **({'start_xy_noise': 0.0, 'start_yaw_noise': 0.0, 'randomize_look': False} if args.no_augment else {}),
        wait_for_client=args.wait_for_client,
        obstacle=obstacle,
        repair=not args.no_repair,
        save_scale=args.save_scale,
    )
except Exception:
    traceback.print_exc()
    code = 1
finally:
    # Flush queued images, then hard-exit: the logger's worker processes otherwise keep a crashed
    # run alive (and holding the GPU).
    if sim is not None and sim.data_logger is not None:
        sim.data_logger.shutdown()
    # No simulation_app.close(): it can hang indefinitely after a run and keep the GPU.
    os._exit(code)
