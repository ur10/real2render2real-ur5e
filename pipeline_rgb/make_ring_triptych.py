"""Real | Track | Render video for the RGB-only tasks (tape roll, nut), in the style of the cube's
triptych (scripts/ur_cube/make_triptych.py): same fonts, colours, panel labels, phase bar and tall
phone variant, front camera crop.

Timing: before the grasp the human video runs in real time up to the start of the tracked segment,
during the carry it runs frame for frame (30 Hz video, 30 Hz rollout), after the place it holds.

    python scripts/ur_tape/make_ring_triptych.py --rollout output_data/tape/replay/successes/env_0_* \
        --track outputs/ur_tape/traj_ep1_hang.json --clean outputs/ur_tape/traj_ep1_hang_clean.json \
        --episode 1 --what "Tape roll pose" --output-dir outputs/ur_tape/media --name tape_triptych
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tyro
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ur_cube"))
from cams import load_cams  # noqa: E402
from make_triptych import ACCENT, BG, INK, MUTED, font  # noqa: E402
from tape_model import BLUE, TapeDims, frame_from_axis, render_labels, sample_surface  # noqa: E402
from track_tape_rgb import read_all  # noqa: E402

PHASES = (("reach", 30), ("grasp", 8), ("carry", None), ("place", 8), ("retreat", 14))


def main(rollout: Path, track: Path, clean: Path, episode: int, what: str, output_dir: Path, name: str,
         render_note: str = "UR5e replay · IsaacLab", crop_box: tuple[int, int, int, int] = (200, 100, 1040, 585),
         panel: tuple[int, int] = (624, 351), fps: int = 30) -> None:
    d = json.loads(track.read_text())
    n_carry = len(json.loads(clean.read_text())["T_base_obj"])
    cam = load_cams()[0]
    pts, lab = sample_surface(TapeDims.from_json(d["dims"]))
    C, A = np.array(d["full_track"]["centre"]), np.array(d["full_track"]["axis"])
    vid = read_all(episode, 0)
    sim = sorted((rollout / "camera_0/rgb").glob("*.jpg"))
    bounds, k0 = [], 0
    for ph, length in PHASES:
        length = n_carry if length is None else length
        bounds.append((ph, k0, k0 + length))
        k0 += length
    n = min(len(sim), k0)
    f0 = d["frames"][0]
    reach_end = bounds[1][2]
    x0, y0, w, h = crop_box

    def crop(img):
        return cv2.resize(img[y0:y0 + h, x0:x0 + w], panel, interpolation=cv2.INTER_AREA)

    def to_px(uv):
        return ((uv - [x0, y0]) * [panel[0] / w, panel[1] / h]).astype(np.int32)

    f_lab, f_cap, f_phase = font("SourceSans3.ttf", 16, 700), font("SourceSans3.ttf", 18, 500), font("SourceSans3.ttf", 15, 600)
    f_tag = font("SourceSans3.ttf", 18, 600)
    W = 3 * panel[0] + 4 * 12
    H = 62 + panel[1] + 48
    H += H % 2
    TW, TH = panel[0] + 24, 12 + 3 * (panel[1] + 12) + 40
    TW, TH = TW + TW % 2, TH + TH % 2
    labels = [("1  Real", "Human demo · RealSense"), ("2  Track", f"{what} · two-view RGB"), ("3  Render", render_note)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k in range(n):
            f = int(np.clip(f0 - (reach_end - k) if k < reach_end else min(f0 + (k - reach_end), f0 + n_carry - 1), 0, len(vid) - 1))
            real_bgr = vid[f]
            real = cv2.cvtColor(crop(real_bgr), cv2.COLOR_BGR2RGB)
            trk = (real * 0.72).astype(np.uint8)
            trail = C[f0:f + 1] if f >= f0 else C[f:f + 1]
            if len(trail) > 1:
                cv2.polylines(trk, [to_px(cam.project(trail)[0])], False, ACCENT, 2, cv2.LINE_AA)
            m = render_labels(cam, frame_from_axis(A[f]), C[f], pts, lab, (0, 0, cam.W, cam.H)) == BLUE
            m = cv2.resize(m[y0:y0 + h, x0:x0 + w].astype(np.uint8), panel, interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(trk, cnts, -1, ACCENT, 2, cv2.LINE_AA)
            r = cv2.imread(str(sim[k]))
            r = cv2.resize(r, (cam.W, cam.H), interpolation=cv2.INTER_CUBIC) if r.shape[1] != cam.W else r
            render = cv2.cvtColor(crop(r), cv2.COLOR_BGR2RGB)
            panels = [real, trk, render]
            phase = next(p for p, a, b in bounds if a <= k < b) if k < bounds[-1][2] else "retreat"

            canvas = Image.new("RGB", (W, H), BG)
            draw = ImageDraw.Draw(canvas)
            for j, (img, (lb, cp)) in enumerate(zip(panels, labels)):
                x = 12 + j * (panel[0] + 12)
                canvas.paste(Image.fromarray(img), (x, 62))
                draw.text((x, 14), lb, font=f_lab, fill=ACCENT)
                draw.text((x, 34), cp, font=f_cap, fill=INK)
            y = 62 + panel[1] + 20
            draw.rectangle((12, y, W - 12, y + 3), fill=(42, 46, 50))
            draw.rectangle((12, y, 12 + int((W - 24) * k / (n - 1)), y + 3), fill=ACCENT)
            for p, a, b in bounds:
                xa = 12 + int((W - 24) * a / (n - 1))
                draw.text((xa + 4, y + 9), p.capitalize(), font=f_phase, fill=INK if p == phase else MUTED)
            canvas.save(tmp / f"t{k:04d}.png")

            tall = Image.new("RGB", (TW, TH), BG)
            dt = ImageDraw.Draw(tall)
            for j, (img, (lb, cp)) in enumerate(zip(panels, labels)):
                yy = 12 + j * (panel[1] + 12)
                tall.paste(Image.fromarray(img), (12, yy))
                tag = f"{lb}  {cp}"
                dt.rectangle((20, yy + 8, 36 + dt.textlength(tag, font=f_tag), yy + 36), fill=BG)
                dt.text((28, yy + 11), tag, font=f_tag, fill=ACCENT)
            y = 12 + 3 * (panel[1] + 12) + 6
            dt.rectangle((12, y, TW - 12, y + 3), fill=(42, 46, 50))
            dt.rectangle((12, y, 12 + int((TW - 24) * k / (n - 1)), y + 3), fill=ACCENT)
            dt.text((12, y + 10), phase.capitalize(), font=f_tag, fill=INK)
            tall.save(tmp / f"v{k:04d}.png")
        output_dir.mkdir(parents=True, exist_ok=True)
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-preset", "slow", "-movflags", "+faststart"]
        mid = bounds[2][1] + int(n_carry * 0.6)
        for pre, suf in (("t", ""), ("v", "_tall")):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / f"{pre}%04d.png"), *enc,
                            str(output_dir / f"{name}{suf}.mp4")], check=True)
            Image.open(tmp / f"{pre}{mid:04d}.png").save(output_dir / f"{name}{suf}.jpg", quality=86)
    for suf in ("", "_tall"):
        print(output_dir / f"{name}{suf}.mp4", f"{(output_dir / f'{name}{suf}.mp4').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    tyro.cli(main)
