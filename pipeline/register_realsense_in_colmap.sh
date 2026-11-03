#!/bin/bash
# Registers the two fixed RealSense views (first frame of a demo episode) into a scan's COLMAP model.
# Works on a copy of the scan's database + sparse model, so the scan itself is untouched.
#
# Usage: register_realsense_in_colmap.sh <scan_dir> <capture_dir> <episode> <work_dir>
# Output: <work_dir>/sparse_txt/{cameras,images,points3D}.txt with rs_front.jpg / rs_right.jpg registered.
set -eo pipefail
SCAN=$(realpath "$1"); CAPTURE=$(realpath "$2"); EP=$3; W=$4
mkdir -p "$W/images" "$W/sparse_reg" "$W/sparse_txt"
W=$(realpath "$W")

cp "$SCAN/colmap/database.db" "$W/database.db"
for f in "$SCAN"/images/*; do ln -sf "$f" "$W/images/$(basename "$f")"; done

for cam in front right; do
    cp "$CAPTURE/rgbd/$EP/000000/realsense_${cam}_color.jpg" "$W/images/rs_${cam}.jpg"
    # RealSense color distortion is OpenCV (k1, k2, p1, p2, k3); FULL_OPENCV adds k4..k6 = 0.
    params=$(python3 -c "
import json; c = json.load(open('$CAPTURE/rgbd/$EP/000000/meta.json'))['cameras']['realsense_$cam']['intrinsics']
k = c['coeffs']; print(','.join(map(str, [c['fx'], c['fy'], c['ppx'], c['ppy'], k[0], k[1], k[2], k[3], k[4], 0, 0, 0])))")
    echo "rs_${cam}.jpg" > "$W/list_${cam}.txt"
    colmap feature_extractor --database_path "$W/database.db" --image_path "$W/images" \
        --image_list_path "$W/list_${cam}.txt" --ImageReader.single_camera 1 \
        --ImageReader.camera_model FULL_OPENCV --ImageReader.camera_params "$params" \
        --FeatureExtraction.use_gpu 1
done

# Match only the new views against every scan frame (and each other).
: > "$W/pairs.txt"
for cam in front right; do
    for f in "$SCAN"/images/*; do echo "rs_${cam}.jpg $(basename "$f")" >> "$W/pairs.txt"; done
done
echo "rs_front.jpg rs_right.jpg" >> "$W/pairs.txt"
colmap matches_importer --database_path "$W/database.db" --match_list_path "$W/pairs.txt" \
    --match_type pairs --FeatureMatching.use_gpu 1

# Keep the new cameras' intrinsics fixed: they come from the RealSense factory calibration.
colmap image_registrator --database_path "$W/database.db" --input_path "$SCAN/colmap/sparse/0" \
    --output_path "$W/sparse_reg" --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
colmap model_converter --input_path "$W/sparse_reg" --output_path "$W/sparse_txt" --output_type TXT
grep -A1 "rs_" "$W/sparse_txt/images.txt" | head -4
