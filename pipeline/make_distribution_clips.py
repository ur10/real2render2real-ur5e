"""Data-distribution clips in the style of the Real2Render2Real project page: every video frame is the
first frame of a different generated rollout, so playing the clip flicks through the spread of start
poses, lighting, camera poses, table looks (and bottle placements) in the dataset.

Example:
    python scripts/ur_cube/make_distribution_clips.py --out-dir outputs/ur_cube/media_v2/distribution
"""

import random
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

# crops in the saved (640x360) frames, 16:9, around the workspace
CROPS = {"camera_0": (90, 30, 570, 300), "camera_1": (80, 90, 560, 360)}


def clip(src: Path, camera: str, output: Path, fps: int, size: tuple[int, int], frame: int, limit: int | None, seed: int):
    runs = sorted(p for p in src.iterdir() if (p / camera / "rgb").is_dir())
    random.Random(seed).shuffle(runs)
    runs = runs[:limit]
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(runs):
            f = sorted((r / camera / "rgb").glob("*.jpg"))[frame]
            Image.open(f).convert("RGB").crop(CROPS[camera]).resize(size, Image.LANCZOS).save(f"{tmp}/f{i:04d}.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["/usr/bin/ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/f%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow", "-movflags", "+faststart",
                        str(output)], check=True)
        Image.open(f"{tmp}/f0000.png").save(output.with_suffix(".jpg"), quality=86)
    print(f"{src} {camera}: {len(runs)} rollouts -> {output} ({output.stat().st_size / 1e6:.2f} MB)")


def main(out_dir: Path = Path("outputs/ur_cube/media_v2/distribution"), raw: Path = Path("output_data/pol"),
         fps: int = 15, size: tuple[int, int] = (480, 270), frame: int = 0, limit: int | None = None, seed: int = 0) -> None:
    for name, src in (("r2r2r", "r2r2r"), ("bottle", "bottle_repair")):
        for cam, tag in (("camera_0", "front"), ("camera_1", "right")):
            clip(raw / src / "successes", cam, out_dir / f"{name}_{tag}.mp4", fps, size, frame, limit, seed)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
