"""Time-aligned Real | Track | Render video (plus a single-panel teaser) for the project page.

Panels share the front RealSense crop, so they line up geometrically. The real and tracked frames
(10 Hz capture) are retimed onto the rendered rollout's phases: before the grasp the human video
plays through the reach, during the carry it runs in real time, after the place it holds.

Example:
    python scripts/ur_cube/make_triptych.py --rollout <demo replay successes/env_...> \
        --traj outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json --output-dir outputs/ur_cube/media
"""

import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from rgbd_utils import load_frame, project

REPO = Path(__file__).resolve().parents[2]
FONTS = REPO / "outputs/ur_cube/fonts"
ACCENT = (95, 181, 192)  # site --accent (dark theme)
INK, MUTED, BG = (230, 228, 224), (154, 161, 168), (19, 21, 23)
CROP = (200, 60, 1040, 585)  # x0, y0, w, h in the 1280x720 front frame (16:9)
PHASES = (("reach", 30), ("grasp", 8), ("carry", None), ("place", 8), ("retreat", 14))


def font(name: str, size: int, weight: int | None = None):
    f = ImageFont.truetype(str(FONTS / name), size)
    if weight is not None:
        try:
            f.set_variation_by_axes([weight] if "Sans" in name else [12, weight])
        except Exception:
            pass
    return f


def crop(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x0, y0, w, h = CROP
    return cv2.resize(img[y0 : y0 + h, x0 : x0 + w], size, interpolation=cv2.INTER_AREA)


def to_crop_px(uv: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    x0, y0, w, h = CROP
    return (uv - [x0, y0]) * [size[0] / w, size[1] / h]


def box_corners(T: np.ndarray, size) -> np.ndarray:
    h = np.asarray(size) / 2
    c = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return c @ T[:3, :3].T + T[:3, 3]


EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def main(
    rollout: Path,
    traj: Path,
    output_dir: Path,
    capture_dir: Path = Path.home() / "ur_human_capture",
    episode: int = 0,
    first_step: int = 52,
    fps: int = 30,
    panel: tuple[int, int] = (624, 351),
) -> None:
    t = json.loads(traj.read_text())
    poses = np.load(traj.with_suffix(".poses_all.npy"))  # 10 Hz, from first_step
    i0, i1 = t["demo_frames"]
    cube = t["cube_size_m"]
    sim = sorted((rollout / "camera_0/rgb").glob("*.jpg"))
    n_carry = len(t["T_base_obj"])
    bounds, k0 = [], 0
    for name, length in PHASES:
        length = n_carry if length is None else length
        bounds.append((name, k0, k0 + length))
        k0 += length
    n = min(len(sim), k0)

    def real_index(k: int) -> float:  # fractional index into the 10 Hz track for sim frame k
        reach_end = bounds[1][2]  # grasp closes at the end of "grasp"
        if k < reach_end:
            return i0 * k / reach_end
        if k < bounds[2][2]:
            return i0 + (k - reach_end) / fps * 10.0
        return float(i1)

    cam = load_frame(capture_dir, episode, first_step)["cameras"]["realsense_front"]
    f_lab, f_cap, f_phase = font("SourceSans3.ttf", 16, 700), font("SourceSans3.ttf", 18, 500), font("SourceSans3.ttf", 15, 600)
    W = 3 * panel[0] + 4 * 12
    H = 62 + panel[1] + 48
    H += H % 2  # H.264 needs even dimensions
    labels = [("1  Real", "Human demo · RealSense"), ("2  Track", "Cube pose · RGB-D tracking"),
              ("3  Render", "UR5e replay · IsaacLab")]
    teaser_frames = []
    f_tag = font("SourceSans3.ttf", 18, 600)
    TW = panel[0] + 24
    TH = 12 + 3 * (panel[1] + 12) + 40
    TW, TH = TW + TW % 2, TH + TH % 2

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k in range(n):
            ri = real_index(k)
            step = first_step + int(round(ri))
            frame = load_frame(capture_dir, episode, step)["cameras"]["realsense_front"]["color"]
            real = crop(frame, panel)

            # Track panel: the real frame, dimmed, with the cube box and its trail so far.
            trk = (real * 0.72).astype(np.uint8)
            Tk = poses[min(int(round(ri)), len(poses) - 1)]
            trail = poses[: min(int(round(ri)), len(poses) - 1) + 1, :3, 3]
            if len(trail) > 1:
                uv = to_crop_px(project(cam, trail)[0], panel).astype(np.int32)
                cv2.polylines(trk, [uv], False, ACCENT, 2, cv2.LINE_AA)
            uvb = to_crop_px(project(cam, box_corners(Tk, cube))[0], panel).astype(int)
            for a, b in EDGES:
                cv2.line(trk, tuple(uvb[a]), tuple(uvb[b]), ACCENT, 2, cv2.LINE_AA)

            render = crop(cv2.cvtColor(cv2.imread(str(sim[k])), cv2.COLOR_BGR2RGB), panel)
            panels = [real, trk, render]

            canvas = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(canvas)
            for j, (img, (lab, cap)) in enumerate(zip(panels, labels)):
                x = 12 + j * (panel[0] + 12)
                canvas.paste(Image.fromarray(img), (x, 62))
                draw.text((x, 14), lab, font=f_lab, fill=ACCENT)
                draw.text((x, 34), cap, font=f_cap, fill=INK)
            # Progress bar with the phase name.
            phase = next(name for name, a, b in bounds if a <= k < b) if k < bounds[-1][2] else "retreat"
            y = 62 + panel[1] + 20
            draw.rectangle((12, y, W - 12, y + 3), fill=(42, 46, 50))
            draw.rectangle((12, y, 12 + int((W - 24) * k / (n - 1)), y + 3), fill=ACCENT)
            for name, a, b in bounds:
                xa = 12 + int((W - 24) * a / (n - 1))
                draw.text((xa + 4, y + 9), name.capitalize(), font=f_phase, fill=INK if name == phase else MUTED)
            canvas.save(tmp / f"t{k:04d}.png")
            teaser_frames.append(panels)

            # Tall variant (phones): the same three panels stacked, labels as tags, same timeline.
            tall = Image.new("RGB", (TW, TH), BG)
            dt = ImageDraw.Draw(tall)
            for j, (img, (lab, cap)) in enumerate(zip(panels, labels)):
                y0 = 12 + j * (panel[1] + 12)
                tall.paste(Image.fromarray(img), (12, y0))
                tag = f"{lab}  {cap}"
                dt.rectangle((20, y0 + 8, 36 + dt.textlength(tag, font=f_tag), y0 + 36), fill=BG)
                dt.text((28, y0 + 11), tag, font=f_tag, fill=ACCENT)
            y = 12 + 3 * (panel[1] + 12) + 6
            dt.rectangle((12, y, TW - 12, y + 3), fill=(42, 46, 50))
            dt.rectangle((12, y, 12 + int((TW - 24) * k / (n - 1)), y + 3), fill=ACCENT)
            dt.text((12, y + 10), phase.capitalize(), font=f_tag, fill=INK)
            tall.save(tmp / f"v{k:04d}.png")

        output_dir.mkdir(parents=True, exist_ok=True)
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart"]
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "t%04d.png"),
                        *enc, str(output_dir / "triptych.mp4")], check=True)
        Image.open(tmp / f"t{bounds[2][1] + n_carry // 2:04d}.png").save(output_dir / "triptych.jpg", quality=88)
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "v%04d.png"),
                        *enc, str(output_dir / "triptych_tall.mp4")], check=True)
        Image.open(tmp / f"v{bounds[2][1] + n_carry // 2:04d}.png").save(output_dir / "triptych_tall.jpg", quality=86)

        # Teaser (single 16:9 panel, for a 330 px column): real -> track -> render, labelled.
        size = (640, 360)
        idx = 0
        for which, lab in ((0, "Real · human demo"), (1, "Track · cube pose"), (2, "Render · UR5e replay")):
            for k in range(0, n, 1):
                im = Image.fromarray(cv2.resize(teaser_frames[k][which], size, interpolation=cv2.INTER_AREA))
                d = ImageDraw.Draw(im)
                tw = d.textlength(lab, font=f_tag)
                d.rectangle((12, 12, 28 + tw, 40), fill=BG)
                d.text((20, 15), lab, font=f_tag, fill=ACCENT)
                im.save(tmp / f"s{idx:04d}.png")
                idx += 1
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "s%04d.png"),
                        "-vf", "scale=640:-2", *enc, str(output_dir / "teaser.mp4")], check=True)
        Image.open(tmp / f"s{2 * n + bounds[2][1] + n_carry // 2:04d}.png").save(output_dir / "teaser.jpg", quality=88)
    for f in ("triptych.mp4", "triptych_tall.mp4", "teaser.mp4", "triptych.jpg", "triptych_tall.jpg", "teaser.jpg"):
        print(f"{output_dir / f}: {(output_dir / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    tyro.cli(main)
