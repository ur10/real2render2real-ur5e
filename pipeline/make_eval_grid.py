"""Grid video of closed-loop policy rollouts: one row per policy, one column per held-out scene (the
same scenes in every row), each tile tagged with its outcome. Frames come from the evaluator's
--save-video-envs output (front camera, first batch).

Example:
    python scripts/ur_cube/make_eval_grid.py --set blocked --policies single r2r2r_400 edit_repair_400 \
        --output outputs/ur_cube/media/policy_blocked.mp4
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parents[2] / "outputs/ur_cube/fonts"
BG, ACCENT, HIT, MUTED, TEXT = (19, 21, 23), (95, 181, 192), (226, 104, 90), (150, 156, 162), (226, 230, 233)
LABELS = {"single": "Trained on 1 demo", "r2r2r_25": "R2R2R, 25 rollouts", "r2r2r_100": "R2R2R, 100 rollouts",
          "r2r2r_400": "R2R2R, 400 rollouts", "edit_repair_400": "R2R2R + repaired bottle scenes",
          "edit_reject_400": "R2R2R + filtered bottle scenes"}


def outcome(r: dict) -> tuple[str, tuple]:
    if r["success"]:
        return "placed", ACCENT
    if r["bottle_collision"]:
        return "hit the bottle", HIT
    if not r["grasped"]:
        return "missed the grasp", HIT
    return "not placed", HIT


def main(set: str, policies: tuple[str, ...], output: Path, eval_dir: Path = Path("outputs/ur_cube/eval"),
         n_scenes: int = 4, crop: tuple[int, int, int, int] = (150, 60, 550, 285), tile: tuple[int, int] = (400, 225),
         label_w: int = 190, gap: int = 6, fps: int = 30, hold_end: float = 1.0, lab_size: int = 19, tag_size: int = 15) -> None:
    runs = {p: eval_dir / f"{p}_{set}" for p in policies}
    res = {p: {json.loads(l)["scene"]: json.loads(l) for l in (d / "results.jsonl").read_text().splitlines()} for p, d in runs.items()}
    scenes = sorted(p.name for p in (runs[policies[0]] / "video").iterdir())[:n_scenes]
    frames = {(p, s): sorted((runs[p] / "video" / s).glob("*.jpg")) for p in policies for s in scenes}
    n = min(len(f) for f in frames.values())
    W = label_w + len(scenes) * (tile[0] + gap) + gap
    H = len(policies) * (tile[1] + gap) + gap
    W, H = W + W % 2, H + H % 2
    f_lab = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), lab_size)
    f_lab.set_variation_by_name("Bold")
    f_tag = ImageFont.truetype(str(FONTS / "SourceSans3.ttf"), tag_size)
    f_tag.set_variation_by_name("Bold")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for k_out in range(n + int(hold_end * fps)):
            k = min(k_out, n - 1)
            last = k == n - 1
            canvas = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(canvas)
            for r, p in enumerate(policies):
                y = gap + r * (tile[1] + gap)
                words, line, lines = LABELS.get(p, p).split(), "", []
                for w in words:
                    if d.textlength(line + " " + w, font=f_lab) > label_w - 18 and line:
                        lines.append(line)
                        line = w
                    else:
                        line = (line + " " + w).strip()
                lines.append(line)
                for i, t in enumerate(lines):
                    lh = int(lab_size * 1.3)
                    d.text((gap + 4, y + tile[1] / 2 - lh * len(lines) / 2 + lh * i), t, font=f_lab, fill=TEXT)
                for c, s in enumerate(scenes):
                    x = label_w + gap + c * (tile[0] + gap)
                    im = Image.open(frames[(p, s)][k]).convert("RGB").crop(crop).resize(tile, Image.LANCZOS)
                    canvas.paste(im, (x, y))
                    if last or k_out >= n:
                        lab, col = outcome(res[p][s])
                        tw = d.textlength(lab, font=f_tag)
                        pad = tag_size // 2
                        d.rounded_rectangle((x + 8, y + 8, x + 8 + 2 * pad + tw, y + 8 + int(tag_size * 1.6)), radius=5, fill=BG)
                        d.text((x + 8 + pad, y + 8 + int(tag_size * 0.2)), lab, font=f_tag, fill=col)
            canvas.save(tmp / f"g{k_out:04d}.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "g%04d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "slow", "-movflags", "+faststart",
                        str(output)], check=True)
        Image.open(tmp / f"g{n + int(hold_end * fps) - 1:04d}.png").save(output.with_suffix(".jpg"), quality=86)
    print(f"{len(policies)} policies x {len(scenes)} scenes x {n} frames -> {output} ({output.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
