"""Metric scan->base similarity transform from RealSense views registered into the scan's COLMAP model.

Run register_realsense_in_colmap.sh first. Two independent estimates are computed and compared:
  * baseline: scale from the distance between the two RealSense camera centers (COLMAP vs T_cam2base),
  * depth: RANSAC Umeyama on COLMAP 3D points observed by the RealSense, lifted to 3D with its depth.
The depth estimate is the one written out; the baseline one is a consistency check.

Coordinates of a trained nerfstudio model ("model" frame) relate to COLMAP's via the model's
dataparser_transforms.json, whose `transform` already includes the scan's `applied_transform`
(see nerfstudio_dataparser.py):
    X_model = dp_scale * (dp_T @ X_colmap)

Example:
    python scripts/ur_cube/scan_to_base_from_colmap.py \
        --colmap-txt <work>/sparse_txt \
        --dataparser-transforms outputs/table_new/splatfacto/2026-11-01_205333/dataparser_transforms.json \
        --output outputs/table_new/scan_to_base.json
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro

from rgbd_utils import intrinsics_matrix, load_frame


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_colmap_txt(path: Path):
    images = {}
    lines = [l for l in (path / "images.txt").read_text().splitlines() if not l.startswith("#")]
    for header, pts in zip(lines[0::2], lines[1::2]):
        h = header.split()
        name = h[9]
        T = np.eye(4)
        T[:3, :3] = qvec2rotmat(np.array(h[1:5], float))
        T[:3, 3] = np.array(h[5:8], float)
        p = np.array(pts.split(), float).reshape(-1, 3) if pts.strip() else np.zeros((0, 3))
        images[name] = {"T_cam_world": T, "xy": p[:, :2], "pid": p[:, 2].astype(int)}
    points = {}
    for l in (path / "points3D.txt").read_text().splitlines():
        if l.startswith("#") or not l.strip():
            continue
        s = l.split()
        points[int(s[0])] = np.array(s[1:4], float)
    return images, points


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimizing ||dst - (s R src + t)||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def ransac_umeyama(src, dst, thresh, iters=5000, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(len(src), 4, replace=False)
        s, R, t = umeyama(src[idx], dst[idx])
        inl = np.linalg.norm(dst - (s * src @ R.T + t), axis=1) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    s, R, t = umeyama(src[best], dst[best])
    return s, R, t, best


def sim_matrix(s, R, t):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def main(
    colmap_txt: Path,
    output: Path,
    dataparser_transforms: Optional[Path] = None,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    inlier_thresh_m: float = 0.01,
) -> None:
    images, points = read_colmap_txt(colmap_txt)
    meta = load_frame(capture_dir, episode, 0)

    centers_colmap, centers_base, src, dst = [], [], [], []
    for cam_name in ("front", "right"):
        img = images.get(f"rs_{cam_name}.jpg")
        assert img is not None, f"rs_{cam_name}.jpg was not registered"
        cam = meta["cameras"][f"realsense_{cam_name}"]
        T_cam_colmap = img["T_cam_world"]
        T_base_cam = np.asarray(cam["T_cam2base"])
        centers_colmap.append(np.linalg.inv(T_cam_colmap)[:3, 3])
        centers_base.append(T_base_cam[:3, 3])

        # Lift each observed COLMAP point with the RealSense depth at its (distorted) pixel.
        has3d = img["pid"] >= 0
        xy, pid = img["xy"][has3d], img["pid"][has3d]
        u, v = np.round(xy).astype(int).T
        ok = (u >= 0) & (v >= 0) & (u < cam["depth_m"].shape[1]) & (v < cam["depth_m"].shape[0])
        xy, pid, u, v = xy[ok], pid[ok], u[ok], v[ok]
        z = cam["depth_m"][v, u]
        ok = (z > 0.1) & (z < 2.0)
        norm = cv2.undistortPoints(
            xy[ok].reshape(-1, 1, 2), intrinsics_matrix(cam), np.array(cam["intrinsics"]["coeffs"])
        ).reshape(-1, 2)
        p_cam = np.concatenate([norm, np.ones((len(norm), 1))], 1) * z[ok, None]
        dst.append(p_cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3])
        src.append(np.stack([points[i] for i in pid[ok]]))
        print(f"{cam_name}: {has3d.sum()} observed 3D points, {ok.sum()} with valid depth")

    # Estimate 1: camera baseline.
    s_base = np.linalg.norm(centers_base[0] - centers_base[1]) / np.linalg.norm(centers_colmap[0] - centers_colmap[1])
    # Estimate 2: depth-lifted correspondences.
    src, dst = np.concatenate(src), np.concatenate(dst)
    s, R, t, inl = ransac_umeyama(src, dst, inlier_thresh_m)
    resid = np.linalg.norm(dst[inl] - (s * src[inl] @ R.T + t), axis=1)
    T_base_colmap = sim_matrix(s, R, t)
    pred_centers = [s * R @ c + t for c in centers_colmap]
    center_err = [np.linalg.norm(p - c) for p, c in zip(pred_centers, centers_base)]
    print(f"depth Umeyama: scale {s:.5f} m/colmap-unit, {inl.sum()}/{len(inl)} inliers, "
          f"median residual {np.median(resid) * 1000:.1f} mm")
    print(f"baseline scale {s_base:.5f} m/colmap-unit ({100 * (s_base / s - 1):+.2f}% vs depth)")
    print("camera center error under depth fit (mm): front %.1f, right %.1f" % tuple(1000 * np.array(center_err)))

    result = {
        "scale_m_per_colmap_unit": s,
        "scale_baseline_m_per_colmap_unit": s_base,
        "T_base_colmap": T_base_colmap.tolist(),
        "num_inliers": int(inl.sum()),
        "median_residual_m": float(np.median(resid)),
        "camera_center_error_m": {"front": float(center_err[0]), "right": float(center_err[1])},
    }

    if dataparser_transforms is not None:
        dp = json.loads(dataparser_transforms.read_text())
        D = np.eye(4)
        D[:3, :] = np.array(dp["transform"])
        M_model_colmap = np.diag([dp["scale"]] * 3 + [1.0]) @ D
        T_base_model = T_base_colmap @ np.linalg.inv(M_model_colmap)
        result.update(
            {
                "dataparser_transforms": str(dataparser_transforms),
                "scale_m_per_model_unit": s / dp["scale"],
                "T_base_model": T_base_model.tolist(),
            }
        )
        print(f"model frame ({dataparser_transforms.parent.name}): {s / dp['scale']:.5f} m/model-unit")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
