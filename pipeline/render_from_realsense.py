"""Render a trained splat from the calibrated RealSense poses and blend it with the real frames.

A sharp, ghost-free blend means the scan->base transform (scale included) is right.

Example:
    python scripts/ur_cube/render_from_realsense.py \
        --config outputs/table_new/splatfacto/2026-11-01_205333/config.yml \
        --scan-to-base outputs/table_new/scan_to_base.json --output-dir /tmp/check
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.eval_utils import eval_setup

from rgbd_utils import load_frame


def camera_in_model(T_base_model: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """OpenCV camera pose in base -> nerfstudio (OpenGL) camera-to-world in the model frame."""
    s = np.cbrt(np.linalg.det(T_base_model[:3, :3]))
    R_bm, t_bm = T_base_model[:3, :3] / s, T_base_model[:3, 3]
    c2w = np.eye(4)
    c2w[:3, :3] = R_bm.T @ T_base_cam[:3, :3]
    c2w[:3, 3] = R_bm.T @ (T_base_cam[:3, 3] - t_bm) / s
    c2w[:3, 1:3] *= -1  # OpenCV (y down, z forward) -> OpenGL (y up, z back)
    return c2w


def main(
    config: Path,
    scan_to_base: Path,
    output_dir: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    step: int = 0,
) -> None:
    T_base_model = np.array(json.loads(scan_to_base.read_text())["T_base_model"])
    _, pipeline, _, _ = eval_setup(config, test_mode="inference")
    model = pipeline.model
    meta = load_frame(capture_dir, episode, step)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, cam in meta["cameras"].items():
        i = cam["intrinsics"]
        c2w = camera_in_model(T_base_model, np.asarray(cam["T_cam2base"]))
        ns_cam = Cameras(
            camera_to_worlds=torch.tensor(c2w[None, :3, :], dtype=torch.float32),
            fx=i["fx"], fy=i["fy"], cx=i["ppx"], cy=i["ppy"], width=i["width"], height=i["height"],
        ).to(model.device)
        with torch.no_grad():
            rgb = model.get_outputs_for_camera(ns_cam)["rgb"].cpu().numpy()
        render = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        # The real frame is distorted; undistort it so both are pinhole images with the same K.
        K = np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]])
        real = cv2.undistort(cam["color"], K, np.array(i["coeffs"]))
        blend = (0.5 * real + 0.5 * render).astype(np.uint8)
        edges = cv2.Canny(cv2.cvtColor(render, cv2.COLOR_RGB2GRAY), 60, 150)
        overlay = real.copy()
        overlay[edges > 0] = (255, 0, 255)
        out = np.vstack([np.hstack([real, render]), np.hstack([blend, overlay])])
        path = output_dir / f"{name}_ep{episode}_s{step}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(cv2.resize(out, None, fx=0.5, fy=0.5), cv2.COLOR_RGB2BGR))
        print(f"wrote {path}")


if __name__ == "__main__":
    tyro.cli(main)
