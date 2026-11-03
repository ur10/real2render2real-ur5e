"""Generate UR5e + Robotiq 2F-85 cube pick-and-place demonstrations (headless).

    OMNI_KIT_ACCEPT_EULA=YES python scripts/run_ur5e_cube.py \
        --traj outputs/ur_cube/traj_ep0_f1_to_f0.json --num-envs 4 --max-successes 50 \
        --output-dir output_data/ur5e_cube_pnp

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
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless, args.enable_cameras = True, True
simulation_app = AppLauncher(args).app

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import UR5eCubePnPCfg  # noqa: E402
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_cube_pnp import UR5eCubePnP  # noqa: E402

import os  # noqa: E402
import traceback  # noqa: E402

repo = Path(__file__).resolve().parents[1]
sim, code = None, 0
try:
    sim = UR5eCubePnP(
        simulation_app,
        UR5eCubePnPCfg(num_envs=args.num_envs, env_spacing=100.0),  # far apart: other envs must not show up in the renders
        urdf_path={"robot": repo / "data/ur5e_robotiq/ur5e_robotiq85.urdf"},
        save_data=not args.no_save,
        output_dir=str(args.output_dir.resolve()),
        traj_path=args.traj,
        max_successes=args.max_successes,
        max_episodes=args.max_episodes,
        **({'start_xy_noise': 0.0, 'start_yaw_noise': 0.0, 'randomize_look': False} if args.no_augment else {}),
        wait_for_client=args.wait_for_client,
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
