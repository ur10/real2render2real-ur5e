"""Human demo vs robot replay for the tape task, time-aligned over the carry: the real view with the
tracked roll outline, and the rendered UR5e replay (run_ur5e_tape.py --no-augment) from the same
calibrated camera. The twin's mug tree is turned (see sweep_tree_yaw.py), so the hang differs.

    python scripts/ur_tape/make_tape_compare.py --rollout output_data/tape/smoke/successes/env_0_* \
        --output outputs/ur_tape/media/tape_compare.mp4
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from tape_model import BLUE, TapeDims, frame_from_axis, render_labels, sample_surface  # noqa: E402
from track_tape_rgb import read_all  # noqa: E402

LEAD = 30 + 8  # logged steps before the carry (approach + close)


def label(im, text):
    cv2.rectangle(im, (0, 0), (len(text) * 11 + 16, 30), (23, 21, 19), -1)
    cv2.putText(im, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)


def main(rollout: Path, output: Path = Path("outputs/ur_tape/media/tape_compare.mp4"),
         track: Path = Path("outputs/ur_tape/traj_ep1_hang.json"), clean: Path = Path("outputs/ur_tape/traj_ep1_hang_clean.json"),
         episode: int = 1, camera: int = 0, fps: int = 20, hold: int = 20) -> None:
    d, dc = json.loads(track.read_text()), json.loads(clean.read_text())
    cam = load_cams()[camera]
    pts, lab = sample_surface(TapeDims(**d["dims"]))
    f0 = d["frames"][0]
    n = len(dc["T_base_obj"])
    vid = read_all(episode, camera)
    renders = sorted((rollout / f"camera_{camera}" / "rgb").glob("*.jpg"))
    tmp = Path(tempfile.mkdtemp())
    ks = list(range(-LEAD, n + 22)) + [n + 21] * hold
    for j, k in enumerate(ks):
        f = int(np.clip(f0 + k, 0, len(vid) - 1))
        real = vid[f].copy()
        if 0 <= k < len(d["full_track"]["centre"]) - f0:
            c, a = np.array(d["full_track"]["centre"][f]), np.array(d["full_track"]["axis"][f])
            m = render_labels(cam, frame_from_axis(a), c, pts, lab, (0, 0, cam.W, cam.H))
            cnts, _ = cv2.findContours((m == BLUE).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(real, cnts, -1, (0, 220, 255), 2)
        real = cv2.resize(real, (640, 360))
        r = cv2.imread(str(renders[int(np.clip(k + LEAD, 0, len(renders) - 1))]))
        r = cv2.resize(r, (640, 360))
        label(real, "Human demo, tracked roll (real camera)")
        label(r, "UR5e replay, rendered (same camera)")
        cv2.imwrite(str(tmp / f"{j:05d}.png"), np.hstack([real, np.full((360, 8, 3), 255, np.uint8), r]))
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                    "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    mid = ks.index(int(n * 0.7))
    cv2.imwrite(str(output.with_suffix(".jpg")), cv2.imread(str(tmp / f"{mid:05d}.png")))
    print(output)


if __name__ == "__main__":
    tyro.cli(main)
