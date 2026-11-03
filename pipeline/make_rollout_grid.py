"""Tile generated rollouts (one camera each) into a grid video for the project page.

Example:
    python scripts/ur_cube/make_rollout_grid.py --successes <output_data/.../successes> \
        --output outputs/ur_cube/media/randomized.mp4
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parents[2] / "outputs/ur_cube/fonts"
BG, ACCENT = (19, 21, 23), (95, 181, 192)


def main(
    successes: Path,
    output: Path,
    camera: str = "camera_0",
    cols: int = 4,
    rows: int = 3,
    tile: tuple[int, int] = (474, 266),
    gap: int = 8,
    fps: int = 30,
) -> None:
    eps = sorted(p for p in successes.iterdir() if (p / camera / "rgb").is_dir())[: cols * rows]
    frames = [sorted((e / camera / "rgb").glob("*.jpg")) for e in eps]
    n = min(len(f) for f in frames)
    W, H = cols * tile[0] + (cols + 1) * gap, rows * tile[1] + (rows + 1) * gap
    W, H = W + W % 2, H + H % 2
    f_lab = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), 16)
    f_lab.set_variation_by_name("Bold")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k in range(n):
            canvas = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(canvas)
            for i, fr in enumerate(frames):
                r, c = divmod(i, cols)
                x, y = gap + c * (tile[0] + gap), gap + r * (tile[1] + gap)
                canvas.paste(Image.open(fr[k]).convert("RGB").resize(tile, Image.LANCZOS), (x, y))
                lab = f"#{i + 1:02d}"
                d.rectangle((x + 6, y + 6, x + 14 + d.textlength(lab, font=f_lab), y + 26), fill=BG)
                d.text((x + 10, y + 7), lab, font=f_lab, fill=ACCENT)
            canvas.save(tmp / f"g{k:04d}.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "g%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart",
                        str(output)], check=True)
        Image.open(tmp / f"g{n // 2:04d}.png").save(output.with_suffix(".jpg"), quality=86)
    print(f"{len(eps)} rollouts x {n} frames -> {output} ({output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    tyro.cli(main)
