"""Estimate the similarity transform (metric scale + pose) from a splat scan to the UR base frame.

The splat lives in nerfstudio's normalized, arbitrarily scaled frame. The RealSense depth from the
human-demo capture is metric and already in the base frame (via the calibrated T_cam2base).

Scaled ICP collapses here (the overlap is dominated by the table plane, so shrinking always
"fits"), and the orange fixtures are split up by the tape, so color landmarks are unreliable.
Instead, for each candidate scale we scale the splat, globally register it to the RealSense
cloud with FPFH + RANSAC, refine with rigid point-to-plane ICP, and keep the best-fitting scale.

Writes <output> (json) with `scale` (meters per scan unit) and `T_base_scan` (4x4, includes scale).

Example:
    python scripts/ur_cube/align_scan_to_base.py \
        --ckpt outputs/table_new/splatfacto/2026-11-01_205333/nerfstudio_models/step-000029999.ckpt \
        --output outputs/table_new/scan_to_base.json
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import tyro

from rgbd_utils import backproject, load_frame

reg = o3d.pipelines.registration
VOXEL_M = 0.01


def preprocess(pc: o3d.geometry.PointCloud, voxel: float):
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    fpfh = reg.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=100))
    return pc, fpfh


def register_at_scale(scan_pts: np.ndarray, target, target_fpfh, scale: float, seed: int = 0):
    """Returns (rigid T base<-scaled_scan, fitness, rmse) for one candidate scale."""
    src, src_fpfh = preprocess(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_pts * scale)), VOXEL_M)
    o3d.utility.random.seed(seed)
    ransac = reg.registration_ransac_based_on_feature_matching(
        src, target, src_fpfh, target_fpfh, True, VOXEL_M * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(VOXEL_M * 1.5)],
        reg.RANSACConvergenceCriteria(200000, 0.999),
    )
    T = ransac.transformation
    for max_dist in (VOXEL_M * 2, VOXEL_M, VOXEL_M / 2):
        r = reg.registration_icp(src, target, max_dist, T, reg.TransformationEstimationPointToPlane())
        T = r.transformation
    return T, r.fitness, r.inlier_rmse


def main(
    ckpt: Path,
    output: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    step: int = 0,
    min_opacity: float = 0.5,
    scale_min: float = 0.10,
    scale_max: float = 0.30,
    num_scales: int = 21,
    scene_radius_m: float = 1.5,
) -> None:
    # RealSense side: metric, base frame. Fuse both cameras.
    meta = load_frame(capture_dir, episode, step)
    rs = np.concatenate([backproject(cam, stride=2)[0] for cam in meta["cameras"].values()])
    target, target_fpfh = preprocess(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rs)), VOXEL_M)

    # Splat side: confident gaussian centers. Nerfstudio centers the scan on the camera poses,
    # which orbit the table, so cropping around the origin keeps the bench and drops far walls.
    params = torch.load(ckpt, map_location="cpu", weights_only=False)["pipeline"]
    means = params["_model.gauss_params.means"].numpy()
    opac = torch.sigmoid(params["_model.gauss_params.opacities"]).squeeze(-1).numpy()
    scan = means[opac > min_opacity]

    def crop(scale):
        return scan[np.linalg.norm(scan, axis=1) < scene_radius_m / scale]

    def evaluate(scales):
        out = []
        for s in scales:
            T, fit, rmse = register_at_scale(crop(s), target, target_fpfh, s)
            out.append((fit, s, T, rmse))
            print(f"  scale {s:.4f} m/unit: fitness {fit:.3f}, rmse {rmse * 1000:.1f} mm")
        return out

    print("coarse scale search")
    coarse = evaluate(np.linspace(scale_min, scale_max, num_scales))
    best = max(coarse, key=lambda x: x[0])
    step_size = (scale_max - scale_min) / (num_scales - 1)
    print("fine scale search")
    fine = evaluate(np.linspace(best[1] - step_size, best[1] + step_size, 11))
    fit, scale, T_rigid, rmse = max(coarse + fine, key=lambda x: x[0])

    T = T_rigid @ np.diag([scale, scale, scale, 1.0])  # base <- scan (unscaled units)
    print(f"best: scale {scale:.4f} m/unit, fitness {fit:.3f}, rmse {rmse * 1000:.1f} mm")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "ckpt": str(ckpt),
                "scale": float(scale),
                "T_base_scan": T.tolist(),
                "fitness": fit,
                "rmse_m": rmse,
                "scale_search": [(float(s), float(f)) for f, s, _, _ in sorted(coarse + fine, key=lambda x: x[1])],
                "capture": {"dir": str(capture_dir), "episode": episode, "step": step},
            },
            indent=2,
        )
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
