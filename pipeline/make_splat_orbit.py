"""Orbit video of the phone-scan Gaussian splat, ending on a wipe against a real RealSense frame.

The clip shows the three things the scan step delivers: a Gaussian splat of the whole bench from a
phone video, metric registration into the robot base frame (the two RealSense cameras and the base
axes are drawn where calibration puts them), and agreement with a real camera (the orbit flies into
one RealSense pose and wipes between the splat render and the real frame).

Example:
    python scripts/ur_cube/make_splat_orbit.py --output outputs/ur_cube/media/splat_orbit.mp4
    python scripts/ur_cube/make_splat_orbit.py --stills /tmp/orbit_stills   # framing check only
"""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import tyro
from gsplat import rasterization
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from make_scan_skybox import look_at_viewmat  # noqa: E402
from rgbd_utils import intrinsics_matrix, load_frame  # noqa: E402

FONT = Path("outputs/ur_cube/fonts/SourceSans3.ttf")
LABELS = True


def load_scene(ckpt: Path, scan_to_base: Path, dev: str):
    """All gaussians of the trained splatfacto scan, kept in its own (COLMAP) frame with full SH."""
    p = torch.load(ckpt, map_location=dev, weights_only=False)["pipeline"]
    G = {k.split(".")[-1]: v.float() for k, v in p.items() if k.startswith("_model.gauss_params.")}
    T = np.array(json.loads(scan_to_base.read_text())["T_base_model"])
    return dict(
        means=G["means"], quats=G["quats"] / G["quats"].norm(dim=-1, keepdim=True), scales=torch.exp(G["scales"]),
        opac=torch.sigmoid(G["opacities"]).squeeze(-1), sh=torch.cat([G["features_dc"][:, None], G["features_rest"]], 1),
        T_base_model=T,
    )


def render(scene, V: np.ndarray, K: np.ndarray, w: int, h: int, near: float = 0.05) -> np.ndarray:
    """V is a base-frame world->camera (OpenCV) transform; it is mapped into the scan frame through the
    registered similarity T_base_model, dividing out the scale (projection is scale-invariant)."""
    T = scene["T_base_model"]
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    Vm = np.eye(4)
    Vm[:3, :3] = V[:3, :3] @ (T[:3, :3] / s)
    Vm[:3, 3] = (V[:3, :3] @ T[:3, 3] + V[:3, 3]) / s
    dev = scene["means"].device
    g = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    img, _, _ = rasterization(scene["means"], scene["quats"], scene["scales"], scene["opac"], scene["sh"],
                              g(Vm)[None], g(K)[None], w, h, sh_degree=3, near_plane=near / s, far_plane=50.0 / s,
                              backgrounds=torch.full((1, 3), 0.09, device=dev))
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def project(V, K, pts):
    p = pts @ V[:3, :3].T + V[:3, 3]
    ok = p[:, 2] > 0.05
    uv = p[:, :2] / np.maximum(p[:, 2:3], 1e-6) @ K[:2, :2].T + K[:2, 2]
    return uv, ok


def draw_overlays(img, V, K, cams, alpha=1.0):
    """RealSense frustums and base-frame axes, drawn where calibration puts them."""
    if alpha <= 0:
        return img
    layer = img.copy()
    for T, Kc, wh in cams:
        d = 0.16  # frustum depth (m)
        corners = np.array([[0, 0], [wh[0], 0], [wh[0], wh[1]], [0, wh[1]]], float)
        rays = np.c_[(corners - Kc[:2, 2]) / np.diag(Kc)[:2], np.ones(4)] * d
        pts = np.vstack([T[:3, 3], rays @ T[:3, :3].T + T[:3, 3]])
        uv, ok = project(V, K, pts)
        if not ok.all():
            continue
        q = uv.astype(int)
        for a, b in [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]:
            cv2.line(layer, tuple(q[a]), tuple(q[b]), (255, 214, 64), 2, cv2.LINE_AA)
    axes = np.array([[0, 0, 0], [0.15, 0, 0], [0, 0.15, 0], [0, 0, 0.15]], float)
    uv, ok = project(V, K, axes)
    if ok.all():
        q = uv.astype(int)
        for i, col in zip((1, 2, 3), ((235, 70, 60), (70, 200, 90), (70, 140, 255))):
            cv2.line(layer, tuple(q[0]), tuple(q[i]), col, 3, cv2.LINE_AA)
    return cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)


def label(img, text, sub=None, font_px=26):
    if not LABELS:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    f1 = ImageFont.truetype(str(FONT), font_px)
    f1.set_variation_by_axes([650])
    f2 = ImageFont.truetype(str(FONT), int(font_px * 0.78))
    f2.set_variation_by_axes([500])
    x, y = 22, img.shape[0] - 22
    lines = [(text, f1)] + ([(sub, f2)] if sub else [])
    hs = [d.textbbox((0, 0), t, font=f)[3] + 8 for t, f in lines]
    ws = [d.textbbox((0, 0), t, font=f)[2] for t, f in lines]
    y0 = y - sum(hs) - 8
    d.rounded_rectangle((x - 10, y0 - 6, x + max(ws) + 12, y), 8, fill=(12, 14, 16, 190))
    for (t, f), hh in zip(lines, hs):
        d.text((x, y0), t, font=f, fill=(255, 255, 255, 255) if f is f1 else (220, 225, 230, 255))
        y0 += hh
    return np.asarray(im)


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def slerp_view(V0, V1, s):
    """Interpolate two world->camera transforms through their camera centres and rotations."""
    from scipy.spatial.transform import Rotation, Slerp
    c0, c1 = -V0[:3, :3].T @ V0[:3, 3], -V1[:3, :3].T @ V1[:3, 3]
    R = Slerp([0, 1], Rotation.from_matrix(np.stack([V0[:3, :3], V1[:3, :3]])))([s]).as_matrix()[0]
    c = (1 - s) * c0 + s * c1
    V = np.eye(4)
    V[:3, :3], V[:3, 3] = R, -R @ c
    return V


def main(
    output: Path = Path("outputs/ur_cube/media/splat_orbit.mp4"),
    ckpt: Path = Path("outputs/table_new/splatfacto/2026-11-01_205333/nerfstudio_models/step-000029999.ckpt"),
    scan_to_base: Path = Path("outputs/table_new/scan_to_base.json"),
    capture_dir: Path = Path.home() / "ur_human_capture",
    wipe_camera: str = "realsense_right",
    center: tuple[float, float, float] = (0.05, -0.55, 0.05),
    radius: float = 1.55,
    height: float = 0.95,
    fov_deg: float = 40.0,
    az_start: float = 45.0,
    az_sweep: float = -95.0,
    width: int = 1280,
    height_px: int = 720,
    fps: int = 30,
    stills: Path | None = None,
    labels: bool = True,
) -> None:
    global LABELS
    LABELS = labels
    dev = "cuda"
    scene = load_scene(ckpt, scan_to_base, dev)
    meta = load_frame(capture_dir, 0, 0)
    cams = [(np.asarray(c["T_cam2base"]), intrinsics_matrix(c), (c["intrinsics"]["width"], c["intrinsics"]["height"]))
            for c in meta["cameras"].values()]
    C = np.array(center)
    fov_f = width / 2 / np.tan(np.deg2rad(fov_deg) / 2)
    K = np.array([[fov_f, 0, width / 2], [0, fov_f, height_px / 2], [0, 0, 1]])

    def orbit_view(az):
        a = np.deg2rad(az)
        eye = C + np.array([radius * np.cos(a), radius * np.sin(a), height])
        return look_at_viewmat(eye, C - eye, np.array([0, 0, 1.0]))

    if stills:
        stills.mkdir(parents=True, exist_ok=True)
        for az in range(-180, 180, 45):
            V = orbit_view(az)
            Image.fromarray(draw_overlays(render(scene, V, K, width, height_px, near=0.5), V, K, cams)).save(stills / f"az{az:+04d}.jpg")
        cam = meta["cameras"][wipe_camera]
        Kc = intrinsics_matrix(cam)
        Vc = np.linalg.inv(np.asarray(cam["T_cam2base"]))
        r = render(scene, Vc, Kc, cam["intrinsics"]["width"], cam["intrinsics"]["height"])
        real = cv2.undistort(cam["color"], Kc, np.array(cam["intrinsics"]["coeffs"]))
        Image.fromarray(np.hstack([r, (0.5 * r + 0.5 * real).astype(np.uint8)])).save(stills / "realsense.jpg")
        print("wrote stills to", stills)
        return

    # Timeline (s): orbit 0-5.2, fly into the RealSense pose 5.2-6.6, hold + wipe 6.6-9.4.
    T_orbit, T_fly, T_wipe = 5.2, 1.4, 2.8
    cam = meta["cameras"][wipe_camera]
    Kc = intrinsics_matrix(cam)
    cw, ch = cam["intrinsics"]["width"], cam["intrinsics"]["height"]
    Vc = np.linalg.inv(np.asarray(cam["T_cam2base"]))
    # Real frames are 1280x720 like the output; scale K if not.
    sx, sy = width / cw, height_px / ch
    Kc_out = np.diag([sx, sy, 1.0]) @ Kc
    real = cv2.resize(cv2.undistort(cam["color"], Kc, np.array(cam["intrinsics"]["coeffs"])), (width, height_px))
    V_end_orbit = orbit_view(az_start + az_sweep)
    n = int((T_orbit + T_fly + T_wipe) * fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{width}x{height_px}", "-r", str(fps), "-i", "-",
                            "-vf", "scale=960:-2", "-c:v", "libx264", "-preset", "slow", "-crf", "26",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], stdin=subprocess.PIPE)
    render_c = None
    for i in range(n):
        t = i / fps
        if t < T_orbit:
            V = orbit_view(az_start + az_sweep * (t / T_orbit))
            img = draw_overlays(render(scene, V, K, width, height_px, near=0.5), V, K, cams, alpha=ease(t / 0.8))
            img = label(img, "Gaussian splat of the bench, from a phone video",
                        "registered to the robot base frame · RealSense cameras in yellow")
        elif t < T_orbit + T_fly:
            s = ease((t - T_orbit) / T_fly)
            V = slerp_view(V_end_orbit, Vc, s)
            Kt = (1 - s) * K + s * Kc_out
            img = draw_overlays(render(scene, V, Kt, width, height_px, near=0.5 - 0.45 * s), V, Kt, cams, alpha=1 - s)
            img = label(img, f"Into the calibrated {wipe_camera.split('_')[-1]} RealSense view")
        else:
            if render_c is None:
                render_c = render(scene, Vc, Kc_out, width, height_px)
            s = ease((t - T_orbit - T_fly - 0.3) / (T_wipe - 0.9))
            x = int(width * (1 - s))
            img = render_c.copy()
            img[:, x:] = real[:, x:]
            if 0 < x < width:
                cv2.line(img, (x, 0), (x, height_px), (255, 255, 255), 3, cv2.LINE_AA)
            img = label(img, "Splat render  |  real RealSense frame",
                        "registered to within 4 mm (median)")
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()
    print("wrote", output)


if __name__ == "__main__":
    tyro.cli(main)
