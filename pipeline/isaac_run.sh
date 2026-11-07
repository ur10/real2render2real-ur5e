#!/usr/bin/env bash
# Run an Isaac Sim script headless and stop it once it is done.
#
#   scripts/ur_cube/isaac_run.sh <log> <done_regex> <script.py> [args...]
#
# Isaac Sim processes can hang on exit after their work is finished. This waits until the process
# exits or <done_regex> appears in the log, gives it 30 s to flush, then kills its process group.
# Runs on GPU $ISAAC_GPU (default 0) with conda and CUDA_VISIBLE_DEVICES removed from the environment
# (see run_ur5e_cube.py). JAX allocates on demand: a preallocated pool leaves cuBLAS no room next to
# the renderer on one GPU.
set -uo pipefail
log=$1; done_re=$2; shift 2
gpu=${ISAAC_GPU:-0}
cd "$(dirname "$0")/../.."
env -u CONDA_PREFIX -u CUDA_VISIBLE_DEVICES OMNI_KIT_ACCEPT_EULA=YES XLA_PYTHON_CLIENT_PREALLOCATE=false setsid .venv/bin/python "$@" \
  --device cuda:$gpu --kit_args="--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/activeGpu=$gpu" \
  > "$log" 2>&1 < /dev/null &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if grep -qE "$done_re" "$log"; then sleep 30; break; fi
  sleep 5
done
kill -KILL -"$pid" 2>/dev/null
sleep 3
if grep -qE "$done_re" "$log"; then exit 0; fi
echo "isaac_run: '$done_re' not found in $log" >&2
grep -E "Traceback|Error" "$log" | tail -5 >&2
exit 1
