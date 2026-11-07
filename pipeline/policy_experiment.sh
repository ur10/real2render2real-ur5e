#!/usr/bin/env bash
# Does more diverse generated data give a better policy? Generate datasets of increasing diversity
# from the one tracked human demo, train the same policy on each, evaluate all of them closed-loop on
# the same held-out scenes in the twin.
#
#   scripts/ur_cube/policy_experiment.sh gen      # rollouts (Isaac, GPU 0)
#   scripts/ur_cube/policy_experiment.sh build    # pack training sets
#   scripts/ur_cube/policy_experiment.sh train [names...]   # policies (GPU from $TRAIN_GPU, default 1)
#   scripts/ur_cube/policy_experiment.sh eval  [names...]   # closed-loop eval (Isaac, GPU 0)
#   scripts/ur_cube/policy_experiment.sh all
#
# Datasets (all planned motion from the cleaned demo track, so they differ only in what is varied):
#   single          the demonstration itself, replayed once (no variation): what one demo gives you
#   r2r2r_{25,100,400}   cube start pose + lighting / camera / table randomized (Real2Render2Real)
#   edit_repair_400 200 of those + 200 with a scanned bottle on the route, motion repaired
#   edit_reject_400 200 of those + 200 with a bottle, placements kept only where the unrepaired motion
#                   is already clear (what filtering alone yields). Filtering keeps about half the
#                   scenes, so it takes two generation batches (117 + 83 of 214 rollouts)
# Eval sets (outputs/ur_cube/eval/scenes.json, make_eval_scenes.py): clean, blocked.
# Every step is skipped if its output exists.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
RAW=${RAW:-output_data/pol}
DATA=outputs/ur_cube/policy_data
POL=outputs/ur_cube/policies
EVAL=outputs/ur_cube/eval
TRAJ=outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json
LOGS=outputs/ur_cube/logs
TRAIN_GPU=${TRAIN_GPU:-1}
NAMES=(single r2r2r_25 r2r2r_100 r2r2r_400 edit_repair_400 edit_reject_400)
mkdir -p "$RAW" "$DATA" "$POL" "$EVAL" "$LOGS"

gen() {  # name episodes [args...]
  local name=$1 eps=$2; shift 2
  [ -f "$RAW/$name/.done" ] && return 0
  rm -rf "${RAW:?}/$name"
  echo "== gen $name ($eps episodes)"
  scripts/ur_cube/isaac_run.sh "$LOGS/gen_$name.log" "^\[INFO\]: Episode $eps:" scripts/run_ur5e_cube.py \
    --traj "$TRAJ" --max-episodes "$eps" --max-successes 100000 --output-dir "$RAW/$name" --save-scale 2 --obstacle "$@"
  touch "$RAW/$name/.done"
  echo "   $(ls "$RAW/$name/successes" | wc -l) rollouts"
}

build() {  # name sources...
  local name=$1; shift
  [ -f "$DATA/$name/meta.json" ] && return 0
  echo "== build $name"
  (cd scripts/ur_cube && CUDA_VISIBLE_DEVICES="" ../../$PY build_policy_dataset.py --output "../../$DATA/$name" \
     --sources $(for s in "$@"; do echo "../../$s"; done))
}

train() {
  local name=$1
  [ -f "$POL/$name/policy.pt" ] && return 0
  echo "== train $name (GPU $TRAIN_GPU)"
  (cd scripts/ur_cube && CUDA_VISIBLE_DEVICES=$TRAIN_GPU ../../$PY train_policy.py --data "../../$DATA/$name" \
     --output "../../$POL/$name") > "$LOGS/train_$name.log" 2>&1
}

evaluate() {
  local name=$1 set=$2 out="$EVAL/${1}_$2"
  [ -f "$out/summary.json" ] && return 0
  echo "== eval $name on $set"
  scripts/ur_cube/isaac_run.sh "$LOGS/eval_${name}_$set.log" "^\[EVAL\] \{" scripts/eval_ur5e_cube_policy.py \
    --policy "$POL/$name/policy.pt" --set "$set" --output-dir "$out" --save-video-envs 4 --num-envs 4
  cat "$out/summary.json"
}

case "${1:-all}" in
  gen|all)
    gen single 1 --obstacle-prob 0 --no-augment --num-envs 1
    gen r2r2r 112 --obstacle-prob 0 --num-envs 4   # 4 envs: 8 do not fit in 12 GB with two 1280x720 cameras
    gen bottle_repair 64 --obstacle-prob 1 --num-envs 4
    gen bottle_reject 56 --obstacle-prob 1 --reject-resample --num-envs 4
    gen bottle_reject2 48 --obstacle-prob 1 --reject-resample --num-envs 4
    ;;&
  build|all)
    build single "$RAW/single/successes:1"
    for n in 25 100 400; do build "r2r2r_$n" "$RAW/r2r2r/successes:$n"; done
    build edit_repair_400 "$RAW/r2r2r/successes:200" "$RAW/bottle_repair/successes:200"
    build edit_reject_400 "$RAW/r2r2r/successes:200" "$RAW/bottle_reject/successes:117" "$RAW/bottle_reject2/successes:83"
    ;;&
  train|all)
    shift || true
    for n in "${@:-${NAMES[@]}}"; do [ "$n" = all ] && continue; train "$n"; done
    ;;&
  eval|all)
    shift || true
    [ -f "$EVAL/scenes.json" ] || $PY scripts/ur_cube/make_eval_scenes.py --output "$EVAL/scenes.json"
    for n in "${@:-${NAMES[@]}}"; do [ "$n" = all ] && continue; for s in clean blocked; do evaluate "$n" "$s"; done; done
    ;;
esac
