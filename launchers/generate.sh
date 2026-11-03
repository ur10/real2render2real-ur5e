#!/usr/bin/env bash
# Generate UR5e cube pick-and-place demonstrations from the tracked human demo (headless IsaacLab).
#
#   launchers/generate.sh <real2render2real dir> <output_dir> [num_envs] [max_episodes] [extra args...]
#
# Extra args go to scripts/run_ur5e_cube.py, e.g. --no-augment to replay the demo exactly.
# Isaac Sim needs its EULA accepted (OMNI_KIT_ACCEPT_EULA=YES) and must not inherit a conda env.
set -euo pipefail
R2R="$(realpath "$1")"; OUT="$(realpath -m "$2")"; N="${3:-2}"; EPS="${4:-6}"
TRAJ="${TRAJ:-outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json}"
cd "$R2R"
env -u CONDA_PREFIX -u CUDA_VISIBLE_DEVICES OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-}" \
  .venv/bin/python scripts/run_ur5e_cube.py \
    --traj "$TRAJ" --num-envs "$N" --max-episodes "$EPS" --max-successes 100000 --output-dir "$OUT" \
    "${@:5}" --device cuda:0 \
    --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=0" \
    < /dev/null
