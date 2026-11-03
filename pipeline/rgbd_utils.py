"""Helpers for reading the RealSense RGB-D capture in ~/ur_human_capture."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ("front", "right")


def load_frame(capture_dir: Path, episode: int, step: int) -> dict:
    """Returns meta.json for one step, with color/depth arrays added per camera."""
    frame_dir = capture_dir / "rgbd" / str(episode) / f"{step:06d}"
    meta = json.loads((frame_dir / "meta.json").read_text())
    for cam in meta["cameras"].values():
        cam["color"] = np.asarray(Image.open(frame_dir / cam["files"]["color"]).convert("RGB"))
        cam["depth_m"] = np.asarray(Image.open(frame_dir / cam["files"]["depth"])).astype(np.float32) * cam["depth_scale"]
    return meta


def intrinsics_matrix(cam: dict) -> np.ndarray:
    i = cam["intrinsics"]
    return np.array([[i["fx"], 0, i["ppx"]], [0, i["fy"], i["ppy"]], [0, 0, 1]])


def backproject(cam: dict, depth_range=(0.1, 2.0), stride: int = 1):
    """Back-projects a camera's depth into the robot base frame.

    Returns (points_base [N, 3], colors [N, 3] uint8, pixel_uv [N, 2]). Depth is assumed
    aligned to the color stream (same 1280x720 grid). Lens distortion is ignored here,
    which is sub-pixel for these RealSense coefficients near the image center.
    """
    depth = cam["depth_m"][::stride, ::stride]
    color = cam["color"][::stride, ::stride]
    v, u = np.mgrid[0 : cam["depth_m"].shape[0] : stride, 0 : cam["depth_m"].shape[1] : stride]
    valid = (depth > depth_range[0]) & (depth < depth_range[1])
    K = intrinsics_matrix(cam)
    z = depth[valid]
    x = (u[valid] - K[0, 2]) / K[0, 0] * z
    y = (v[valid] - K[1, 2]) / K[1, 1] * z
    pts_cam = np.stack([x, y, z], axis=1)
    T = np.asarray(cam["T_cam2base"])
    pts_base = pts_cam @ T[:3, :3].T + T[:3, 3]
    return pts_base, color[valid], np.stack([u[valid], v[valid]], axis=1)


def project(cam: dict, pts_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projects base-frame points into a camera. Returns (uv [N, 2], depth [N])."""
    T = np.linalg.inv(np.asarray(cam["T_cam2base"]))
    pts_cam = pts_base @ T[:3, :3].T + T[:3, 3]
    uvw = pts_cam @ intrinsics_matrix(cam).T
    return uvw[:, :2] / uvw[:, 2:3], pts_cam[:, 2]
