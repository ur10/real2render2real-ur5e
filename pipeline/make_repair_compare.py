"""Before/after video of one obstacle placement: the plain generator's motion (env 0, through the
obstacle) next to the repaired motion (env 1), both rendered in the same scene by

    python scripts/run_ur5e_cube.py --traj outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json --obstacle \
        --obstacle-xy X Y --compare --no-augment --num-envs 2 --max-episodes 1 --output-dir <run>

Columns: generator | repaired. Rows: one per camera. Each column's tag turns red on the frames where
the planner's collision check finds the arm, gripper or held cube inside the obstacle (clearance
from <run>/plans.jsonl; the obstacle is visual-only in the renders, so the arm passes through it).

Example:
    python scripts/ur_cube/make_repair_compare.py --run output_data/ur5e_repair_compare \
        --output outputs/ur_cube/media/repair_compare.mp4
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parents[2] / "outputs/ur_cube/fonts"
BG, ACCENT, HIT, MUTED, TEXT = (19, 21, 23), (95, 181, 192), (226, 104, 90), (120, 128, 134), (226, 230, 233)


def main(
    run: Path,
    output: Path,
    cameras: tuple[str, ...] = ("camera_0", "camera_1"),
    crops: tuple[tuple[int, int, int, int], ...] = ((300, 60, 1100, 510), (140, 240, 940, 690)),
    tile: tuple[int, int] = (640, 360),
    gap: int = 8,
    fps: int = 30,
    hold_end: float = 0.8,
) -> None:
    plans = [json.loads(l) for l in (run / "plans.jsonl").read_text().splitlines()]
    by_env = {p["env"]: p for p in plans if p["episode"] == plans[0]["episode"]}  # one episode per compare run
    succ = sorted((run / "successes").iterdir())
    dirs = {e: next(d for d in succ if d.name.startswith(f"env_{e}_")) for e in (0, 1)}
    clear = {e: np.asarray(by_env[e]["obstacle_clearance_per_step"]) for e in (0, 1)}
    frames = {(e, c): sorted((dirs[e] / c / "rgb").glob("*.jpg")) for e in (0, 1) for c in cameras}
    n = min(len(f) for f in frames.values())
    T = len(clear[0])
    step = lambda k: min(int(round(k * (T - 1) / max(n - 1, 1))), T - 1)  # noqa: E731

    head = 44
    W = 2 * tile[0] + 3 * gap
    H = head + len(cameras) * tile[1] + (len(cameras) + 1) * gap
    W, H = W + W % 2, H + H % 2
    f_title = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), 20)
    f_title.set_variation_by_name("Bold")
    f_tag = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), 16)
    f_tag.set_variation_by_name("Bold")
    min_rep = float(clear[1].min()) * 100
    titles = ["Plain generator", "Repaired (free segments re-optimized)"]

    def tag(e: int, k: int) -> tuple[str, tuple]:
        if e == 0:
            hit_now = clear[0][step(k)] < 0
            return ("in collision" if hit_now else "collides: rejected by the filter"), (HIT if hit_now else MUTED)
        return f"clear, min {min_rep:.1f} cm", ACCENT

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        n_out = n + int(hold_end * fps)
        for k_out in range(n_out):
            k = min(k_out, n - 1)
            canvas = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(canvas)
            for e in (0, 1):
                x0 = gap + e * (tile[0] + gap)
                d.text((x0 + 2, 10), titles[e], font=f_title, fill=TEXT)
                lab, col = tag(e, k)
                tw = d.textlength(lab, font=f_tag)
                d.rounded_rectangle((x0 + tile[0] - tw - 22, 10, x0 + tile[0], 34), radius=5, outline=col, width=2)
                d.text((x0 + tile[0] - tw - 11, 12), lab, font=f_tag, fill=col)
                for r, (cam, crop) in enumerate(zip(cameras, crops)):
                    y0 = head + gap + r * (tile[1] + gap)
                    im = Image.open(frames[(e, cam)][k]).convert("RGB").crop(crop).resize(tile, Image.LANCZOS)
                    canvas.paste(im, (x0, y0))
                    if e == 0 and clear[0][step(k)] < 0:
                        d.rectangle((x0, y0, x0 + tile[0] - 1, y0 + tile[1] - 1), outline=HIT, width=3)
            canvas.save(tmp / f"g{k_out:04d}.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "g%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart",
                        str(output)], check=True)
        hits = np.where(clear[0] < 0)[0]
        k_poster = int(round(hits[len(hits) // 2] * (n - 1) / (T - 1))) if len(hits) else n // 2
        Image.open(tmp / f"g{k_poster:04d}.png").save(output.with_suffix(".jpg"), quality=86)
    print(f"{n} frames -> {output} ({output.stat().st_size / 1e6:.2f} MB); generator min clearance "
          f"{clear[0].min() * 100:.1f} cm over {int((clear[0] < 0).sum())} steps, repaired {min_rep:.1f} cm")


if __name__ == "__main__":
    tyro.cli(main)
