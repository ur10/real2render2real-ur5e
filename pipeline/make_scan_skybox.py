"""Equirectangular skybox of the real lab rendered from the phone-scan splat.

The paper lights/backgrounds its renders with HDR photos of its own lab. Here the background is
the scanned bench itself: we render six 90-degree cube-map faces with gsplat from a point above the
workspace (the scan's arm and start-pose cube are already removed in splats_base.npz, and the table
top is dropped too since the simulator draws its own), then resample to a lat-long image for the
dome light. Z is up, longitude 0 looks along +X (base frame).

Example:
    python scripts/ur_cube/make_scan_skybox.py --output data/ur_cube/skybox_scan.jpg
"""

import json
from pathlib import Path

import numpy as np
import torch
import tyro
from gsplat import rasterization
from PIL import Image


def look_at_viewmat(eye, forward, up):
    f = forward / np.linalg.norm(forward)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    R = np.stack([r, -u, f])  # OpenCV camera: x right, y down, z forward
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = -R @ eye
    return V


def main(
    output: Path,
    splats: Path = Path("outputs/ur_cube/splats_base.npz"),
    layout: Path = Path("outputs/ur_cube/scene_layout.json"),
    eye: tuple[float, float, float] = (0.1, -0.55, 0.45),
    face: int = 1024,
    width: int = 4096,
) -> None:
    sp = np.load(splats)
    table_z = json.loads(layout.read_text())["table_z"]
    c = sp["bench_centers"]
    std = np.sqrt(np.linalg.eigvalsh(sp["bench_covariances"]).max(-1).clip(0))
    on_table = (np.abs(c[:, 0] - 0.15) < 0.95) & (np.abs(c[:, 1] + 0.45) < 0.85) & (c[:, 2] < table_z + 0.07)
    keep = ~on_table  # the simulator draws its own table and fixtures
    keep &= (sp["bench_opacities"][:, 0] > 0.3) & (std < 0.04)  # drop floaters / smeared blobs
    keep &= c[:, 2] > table_z - 0.25  # no dark floor under the bench
    dev = "cuda"
    means = torch.tensor(c[keep], device=dev)
    covars = torch.tensor(sp["bench_covariances"][keep], device=dev)
    colors = torch.tensor(sp["bench_rgbs"][keep], device=dev)
    opac = torch.tensor(sp["bench_opacities"][keep, 0], device=dev)
    K = torch.tensor([[face / 2, 0, face / 2], [0, face / 2, face / 2], [0, 0, 1]], dtype=torch.float32, device=dev)

    eye = np.array(eye)
    dirs = {"+x": ([1, 0, 0], [0, 0, 1]), "-x": ([-1, 0, 0], [0, 0, 1]), "+y": ([0, 1, 0], [0, 0, 1]),
            "-y": ([0, -1, 0], [0, 0, 1]), "+z": ([0, 0, 1], [0, 1, 0]), "-z": ([0, 0, -1], [0, 1, 0])}
    faces, Rs = {}, {}
    for name, (fwd, up) in dirs.items():
        V = look_at_viewmat(eye, np.array(fwd, float), np.array(up, float))
        img, _, _ = rasterization(means, None, None, opac, colors, torch.tensor(V, dtype=torch.float32, device=dev)[None],
                                  K[None], face, face, covars=covars, near_plane=0.05, far_plane=50.0)
        faces[name] = img[0].clamp(0, 1).cpu().numpy()
        Rs[name] = V[:3, :3]

    # Lat-long: rows = elevation (+90 .. -90), cols = azimuth (0 .. 360 deg about +Z from +X).
    h = width // 2
    az = (np.arange(width) + 0.5) / width * 2 * np.pi
    el = np.pi / 2 - (np.arange(h) + 0.5) / h * np.pi
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
    out = np.zeros((h, width, 3), np.float32)
    best = np.full((h, width), -np.inf)
    for name, R in Rs.items():
        cam = d @ R.T  # direction in that face's camera frame
        z = cam[..., 2]
        u = face / 2 * cam[..., 0] / np.maximum(z, 1e-6) + face / 2
        v = face / 2 * cam[..., 1] / np.maximum(z, 1e-6) + face / 2
        ok = (z > best) & (z > 0) & (u >= 0) & (u < face) & (v >= 0) & (v < face)
        out[ok] = faces[name][v[ok].astype(int), u[ok].astype(int)]
        best = np.where(ok, z, best)
    # Fill holes (no splats in that direction) with a soft neutral so the dome isn't black.
    empty = out.sum(-1) < 0.02
    sky = np.clip(0.80 + 0.12 * np.sin(E), 0, 1)[..., None] * np.array([1.0, 1.0, 1.02])  # brighter towards the ceiling
    out[empty] = sky[empty]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((out * 255).astype(np.uint8)).save(output, quality=92)
    print(f"{keep.sum()} gaussians; empty {empty.mean():.1%}; wrote {output} ({width}x{h})")


if __name__ == "__main__":
    tyro.cli(main)
