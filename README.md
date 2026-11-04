# One human demo → robot demonstrations, on a UR5e

A single human demonstration on a real UR5e bench, recorded by two calibrated RealSense cameras and
a phone scan, turned into robot demonstrations for a UR5e + Robotiq 2F-85 without teleoperation.
This is a port of [Real2Render2Real](https://github.com/uynitsuj/real2render2real) (released for
YuMi and Franka) to a new robot, new cameras and my own bench.

**Project page with videos and an interactive 3D replay: https://ur10.github.io/projects/real2render2real/**

![Real demo, tracked cube pose and the rendered UR5e replay, time-aligned](docs/scan_track_render.gif)

*Left to right: the human demo from the front RealSense, the tracked 6-DoF cube pose drawn on the
same frame, and the UR5e replay rendered in IsaacLab from the same calibrated camera.*

## What it does

| Step | What happens | Result on this bench |
|---|---|---|
| **Scan** | Gaussian splat of the bench from a phone video. The RealSense views are registered into the scan's COLMAP model to recover metric scale and the scan-to-robot transform. | Two independent scale estimates agree to 0.2%; 4 mm median registration residual; camera centres within 8 mm |
| **Track** | The cube's pose per frame from both depth cameras: the scanned cube model fitted with ICP, re-detected on fast lifts, kept consistent across the cube's 24 symmetric orientations. | Box overlay stays on the cube in the camera that was not used for tracking |
| **Embody** | UR5e + Robotiq 2F-85 URDF in the calibrated base frame, gripper mount fitted against the real images. | Simulated cameras and robot line up with the real ones |
| **Render** | IsaacLab kinematic replay: IK follows a grasp on the tracked cube; start pose jitter + trajectory interpolation, randomized lighting, camera and table appearance; a feasibility filter rejects IK error, joint-limit violations and finger–fixture contact. | 12 / 12 randomized rollouts pass |

![Randomized rollouts generated from the one demonstration](docs/randomized.jpg)

### Why the tracker is different from the paper

The paper tracks objects with RSRD (DINO features + a differentiable splat). On this bench the cube
is small (≈4 cm), black and about a metre from the cameras, and RSRD slid off it onto the fixture
even after the fixes in `patches/` (metric depth loss, calibrated first-frame pose, NaN handling).
With two calibrated RGB-D cameras the motion is easier to recover geometrically, so
`pipeline/track_cube_rgbd.py` replaces it. The second camera validates the track independently.

## Repository layout

```
pipeline/   bench calibration, scan alignment, cube tracking, robot model, media (→ scripts/ur_cube/)
sim/        UR5e cube pick-and-place simulator, scene and robot configs, run script
patches/    fixes to upstream Real2Render2Real, RSRD, DiG and hamer_helper
assets/     small bench files: calibrated cameras, scene layout, tracked trajectory, cube mesh, skybox
launchers/  generate.sh, headless data generation
install.sh  installs all of the above into a Real2Render2Real checkout
```

Notable patches:
- `real2render2real.patch`: camera manager takes a per-robot camera file; works with viser 0.2.x.
- `rsrd.patch`: metric-depth loss and options for small, distant objects; NaN fixes for missing depth.
- `dig.patch`: stop forcing `TORCH_CUDA_ARCH_LIST=8.9`, which broke JIT builds on other GPUs.
- `hamer_helper.patch`: ViTPose import path; optional lighter person detector for 12 GB GPUs.

## Install

Follow the upstream environment setup first (`env_data_gen.sh` for IsaacLab 4.5 data generation,
`env_real_to_sim.sh` for scanning and tracking). Then:

```bash
git clone https://github.com/ur10/real2render2real-ur5e.git
cd real2render2real-ur5e
./install.sh                      # clones upstream at the pinned commit into ./real2render2real
# or: ./install.sh /path/to/your/real2render2real
```

## Reproduce

Paths below are relative to the Real2Render2Real checkout. The capture is a folder of RGB-D steps
with calibrated `T_cam2base` per camera (`pipeline/rgbd_utils.py` documents the layout); the
bench files from my capture are already in `assets/`, so you can start from step 3.

```bash
# 1. Scan -> robot base (needs the phone scan and the capture)
bash scripts/ur_cube/register_realsense_in_colmap.sh <scan_dir> <capture_dir> 0 <work_dir>
python scripts/ur_cube/scan_to_base_from_colmap.py --help
python scripts/ur_cube/measure_scene_from_rgbd.py --help     # table plane, fixtures, cube

# 2. Track the cube with both RealSense cameras
python scripts/ur_cube/track_cube_rgbd.py --init-traj outputs/ur_cube/traj_ep0_f1_to_f0.json \
    --mesh data/ur_cube/cube_mesh.obj --episode 0 --steps 52 84 \
    --output outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json

# 3. Robot model and USDs
python scripts/ur_cube/build_ur5e_robotiq_urdf.py
python dependencies/IsaacLab/scripts/tools/convert_urdf.py data/ur5e_robotiq/ur5e_robotiq85.urdf \
    data/ur5e_robotiq/usd/ur5e_robotiq85.usd --fix-base --headless
python dependencies/IsaacLab/scripts/tools/convert_mesh.py data/ur_cube/cube_mesh.obj \
    data/ur_cube/usd/cube_mesh.usd --make-instanceable --collision-approximation none --mass 1.0 --headless

# 4. Generate demonstrations (RGB from both cameras + joint data per rollout)
OMNI_KIT_ACCEPT_EULA=YES launchers/generate.sh . output_data/ur5e_randomized 2 6
OMNI_KIT_ACCEPT_EULA=YES launchers/generate.sh . output_data/ur5e_demo_replay 1 1 --no-augment

# 5. Figures and the interactive replay
python scripts/ur_cube/make_triptych.py --rollout <replay rollout> \
    --traj outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json --output-dir outputs/ur_cube/media
python scripts/ur_cube/make_rollout_grid.py --successes output_data/ur5e_randomized/successes \
    --output outputs/ur_cube/media/randomized.mp4
python scripts/ur_cube/make_viser_recording.py --rollout <replay rollout> --output-dir outputs/ur_cube/review
python scripts/ur_cube/make_splat_orbit.py --output outputs/ur_cube/media/splat_orbit.mp4  # splat orbit + RealSense wipe
```

Tested on Ubuntu 22.04, driver 580, Isaac Sim 4.5 / IsaacLab, RTX 3080 Ti (12 GB). One GPU job per
card: the tracker and Isaac Sim don't fit together on 12 GB.

## Status

The robot motion is rendered, not yet executed on hardware. Next: replaying generated
demonstrations on the real UR5e against the calibrated cameras, then training a policy on the
generated data.

## Acknowledgements

Built on Real2Render2Real by Justin Yu, Letian Fu, Huang Huang, Karim El-Refai, Rares Andrei Ambrus,
Richard Cheng, Muhammad Zubair Irshad and Ken Goldberg, and on RSRD, DiG, trajgen and IsaacLab.

```bibtex
@misc{yu2025real2render2realscalingrobotdata,
  title={Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware},
  author={Justin Yu and Letian Fu and Huang Huang and Karim El-Refai and Rares Andrei Ambrus and Richard Cheng and Muhammad Zubair Irshad and Ken Goldberg},
  year={2025},
  eprint={2505.09601},
  archivePrefix={arXiv},
  primaryClass={cs.RO}
}
```

MIT License; see `LICENSE`.
