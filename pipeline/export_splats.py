"""Export Gaussian splats in robot-base metres for the viser "Scan, Track, Render" recording.

  cube:  DiG cube state -> cube frame (the frame the trajectory poses), metres
  bench: splatfacto scene scan -> robot base frame, cropped, with the scan's (differently posed)
         arm and the cube at its start pose removed so they don't clash with the animated ones.

Writes an .npz with {cube,bench}_{centers,covariances,rgbs,opacities}.

Example:
    python scripts/ur_cube/export_splats.py --traj outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json \
        --output outputs/ur_cube/splats_base.npz
"""

import json
from pathlib import Path

import numpy as np
import torch
import tyro
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814


def gaussians(params: dict, prefix: str = ""):
    g = lambda k: params[prefix + k].detach().float().numpy()  # noqa: E731
    means, scales, quats = g("means"), np.exp(g("scales")), g("quats")
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]] / np.linalg.norm(quats, axis=1, keepdims=True)).as_matrix()
    cov = R @ (scales[:, :, None] ** 2 * np.swapaxes(R, 1, 2))
    rgb = np.clip(0.5 + SH_C0 * g("features_dc").reshape(len(means), -1)[:, :3], 0, 1)
    opa = 1 / (1 + np.exp(-g("opacities").reshape(-1, 1)))
    return means, cov, rgb, opa


def transform(means, cov, T):
    """Similarity T (4x4, may include scale) applied to centres and covariances."""
    A = T[:3, :3]
    return means @ A.T + T[:3, 3], A @ cov @ A.T


def main(
    traj: Path,
    output: Path,
    cube_state: Path = Path("outputs/table_new/state_rigid_20261102_190817.pt"),
    bench_ckpt: Path = Path("outputs/table_new/splatfacto/2026-11-01_205333/nerfstudio_models/step-000029999.ckpt"),
    scan_to_base: Path = Path("outputs/table_new/scan_to_base.json"),
    layout: Path = Path("outputs/ur_cube/scene_layout.json"),
    bench_radius: float = 2.5,
    max_bench_gaussians: int = 300_000,
) -> None:
    t = json.loads(traj.read_text())
    s, c = t["m_per_model_unit"], np.array(t["objinit_centroid_model"])
    T_obj_cube = np.array(t["T_objinit_cube"])
    T_base_model = np.array(json.loads(scan_to_base.read_text())["T_base_model"])
    table_z = json.loads(layout.read_text())["table_z"]

    # Cube: model -> object frame (centroid, metres) -> cube frame.
    m, cov, rgb, opa = gaussians(torch.load(cube_state, map_location="cpu"))
    T_model_to_obj = np.eye(4)
    T_model_to_obj[:3, :3] *= s
    T_model_to_obj[:3, 3] = -s * c
    cm, ccov = transform(m, cov, np.linalg.inv(T_obj_cube) @ T_model_to_obj)

    # Bench: model -> base (similarity), crop, drop the scanned arm and the start-pose cube.
    p = torch.load(bench_ckpt, map_location="cpu", weights_only=False)["pipeline"]
    bm, bcov, brgb, bopa = gaussians(p, "_model.gauss_params.")
    bm, bcov = transform(bm, bcov, T_base_model)
    keep = (np.linalg.norm(bm[:, :2], axis=1) < bench_radius) & (bm[:, 2] > table_z - 0.8) & (bm[:, 2] < 2.0)
    keep &= bopa[:, 0] > 0.05
    arm = (bm[:, 0] > -0.45) & (bm[:, 0] < 0.45) & (bm[:, 1] > -0.65) & (bm[:, 1] < 0.3) & (bm[:, 2] > table_z + 0.04)
    # The scanned gripper hung further out, toward the fixtures; cut it above fixture/cube height.
    arm |= (bm[:, 0] > -0.45) & (bm[:, 0] < 0.45) & (bm[:, 1] > -0.76) & (bm[:, 1] < 0.3) & (bm[:, 2] > table_z + 0.12)
    cube0 = np.array(t["T_base_obj"][0][4:])
    on_cube = (np.linalg.norm(bm[:, :2] - cube0[:2], axis=1) < 0.04) & (bm[:, 2] > table_z + 0.02) & (bm[:, 2] < cube0[2] + 0.04)
    keep &= ~arm & ~on_cube
    idx = np.where(keep)[0]
    if len(idx) > max_bench_gaussians:  # keep the most opaque
        idx = idx[np.argsort(-bopa[idx, 0])[:max_bench_gaussians]]
    print(f"cube: {len(cm)} gaussians, extent {np.ptp(cm, 0).round(3)} m; bench: {len(idx)} of {len(bm)} kept")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        cube_centers=cm.astype(np.float32), cube_covariances=ccov.astype(np.float32),
        cube_rgbs=rgb.astype(np.float32), cube_opacities=opa.astype(np.float32),
        bench_centers=bm[idx].astype(np.float32), bench_covariances=bcov[idx].astype(np.float32),
        bench_rgbs=brgb[idx].astype(np.float32), bench_opacities=bopa[idx].astype(np.float32),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
