#!/usr/bin/env bash
# Install the UR5e + Robotiq 2F-85 port into a Real2Render2Real checkout.
#
#   ./install.sh                      # clones upstream next to this repo (./real2render2real)
#   ./install.sh /path/to/real2render2real   # or uses an existing checkout
#
# Steps: clone upstream at the commit this work was built on (if needed), apply the patches in
# patches/ to upstream and its submodules, copy the pipeline, simulator and bench assets into place,
# and fetch the font the media scripts use. Safe to re-run: applied patches are detected.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2R="$(realpath -m "${1:-$HERE/real2render2real}")"
UPSTREAM=https://github.com/uynitsuj/real2render2real.git
UPSTREAM_COMMIT=0ac79eb912a1346a24f2b62a75ed6fd2c0ff90de

if [ ! -d "$R2R/real2render2real" ]; then
  echo "Cloning upstream Real2Render2Real into $R2R"
  git clone "$UPSTREAM" "$R2R"
  git -C "$R2R" checkout -q "$UPSTREAM_COMMIT"
  git -C "$R2R" submodule update --init --recursive
fi

apply_patch() {  # <dir> <patch>
  local dir="$1" patch="$HERE/patches/$2"
  if git -C "$dir" apply --check "$patch" 2>/dev/null; then
    git -C "$dir" apply "$patch" && echo "  patched        ${dir#$R2R/}"
  elif git -C "$dir" apply -R --check "$patch" 2>/dev/null; then
    echo "  already patched ${dir#$R2R/}"
  else
    echo "ERROR: $2 does not apply to $dir (is the submodule at the pinned commit?)" >&2
    exit 1
  fi
}
echo "Applying patches"
apply_patch "$R2R"                                          real2render2real.patch
apply_patch "$R2R/dependencies/rsrd"                        rsrd.patch
apply_patch "$R2R/dependencies/rsrd/dependencies/dig"       dig.patch
apply_patch "$R2R/dependencies/rsrd/dependencies/hamer_helper" hamer_helper.patch

echo "Copying pipeline, simulator and assets"
V="$R2R/real2render2real/isaaclab_viser"
mkdir -p "$R2R/scripts/ur_cube" "$V/ur5e_simulators"
cp "$HERE"/pipeline/*.py "$HERE"/pipeline/*.sh "$R2R/scripts/ur_cube/"
cp "$HERE/sim/run_ur5e_cube.py" "$HERE/sim/eval_ur5e_cube_policy.py" "$R2R/scripts/"
cp "$HERE"/sim/ur5e_simulators/*.py "$V/ur5e_simulators/"
cp "$HERE/sim/configs/ur5e_robotiq_cfg.py" "$V/configs/articulation_configs/"
cp "$HERE/sim/configs/ur5e_scene_cfg.py" "$V/configs/scene_configs/"
cp -r "$HERE/assets/data/." "$R2R/data/"
mkdir -p "$R2R/outputs" && cp -r "$HERE/assets/outputs/." "$R2R/outputs/"

FONTS="$R2R/outputs/ur_cube/fonts"
mkdir -p "$FONTS"
fetch_font() {  # <google/fonts path> <file name>
  [ -s "$FONTS/$2" ] || curl -fsSL -o "$FONTS/$2" "https://github.com/google/fonts/raw/main/$1"
}
fetch_font "ofl/sourcesans3/SourceSans3%5Bwght%5D.ttf" SourceSans3.ttf

echo "Done. Next: build the robot model and USDs (see README, 'Reproduce')."
