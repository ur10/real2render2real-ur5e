"""The tracked demo trajectory (teal) and N interpolated variants (amber) drawn on the real front
camera view, for the RGB-only tasks (the cube's make_traj_fan.py figure). The tape's variants end
on the twin's turned tree, so its fan is drawn in the twin frame on the real image of the bench,
with the real (unturned) tree in view; the caption says so.

    python scripts/ur_tape/make_ring_traj_fan.py --task nut --output outputs/ur_nut/media/nut_traj_fan.jpg
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from ring_repair import REPO, TASKS  # noqa: E402
from tape_plan import rotate_traj_about_post  # noqa: E402
from track_tape_rgb import VIDEOS  # noqa: E402
from trajgen.traj_interp import traj_interp_batch  # noqa: E402

TEAL, AMBER = (192, 181, 95), (70, 160, 230)  # BGR of the site's accent colours
EPISODE = {"tape": 1, "nut": 4}


def main(task: str = "tape", output: Path | None = None, n: int = 64, xy_noise: float = 0.04, seed: int = 0,
         crop: tuple[int, int, int, int] = (200, 100, 1040, 585)) -> None:
    cfg = TASKS[task]
    output = output or REPO / f"outputs/ur_{task}/media/{task}_traj_fan.jpg"
    demo = np.asarray(json.loads((REPO / cfg["traj"]).read_text())["T_base_obj"], np.float64)
    scene = json.loads((REPO / cfg["scene"]).read_text())
    post = np.asarray(scene["mug_tree"]["post"]["xy"] if "mug_tree" in scene else scene["bolt_stand"]["xy"])
    twin = rotate_traj_about_post(demo, np.radians(cfg["tree_yaw"]), post) if cfg["tree_yaw"] else demo
    rng = np.random.default_rng(seed)
    starts = np.repeat(demo[:1], n, 0)
    starts[:, 4:6] += np.asarray(cfg["start_offset"]) + rng.uniform(-xy_noise, xy_noise, (n, 2))
    fan = np.asarray(traj_interp_batch(traj=twin, new_starts=starts, proportion=0.6))
    cam = load_cams()[0]
    cap = cv2.VideoCapture(str(VIDEOS / str(EPISODE[task]) / "0.mp4"))
    im = cap.read()[1]
    over = im.copy()
    for f in fan:
        cv2.polylines(over, [cam.project(f[:, 4:])[0].astype(np.int32)], False, AMBER, 2, cv2.LINE_AA)
    im = cv2.addWeighted(over, 0.75, im, 0.25, 0)
    cv2.polylines(im, [cam.project(demo[:, 4:])[0].astype(np.int32)], False, TEAL, 4, cv2.LINE_AA)
    for f in fan:
        cv2.circle(im, tuple(cam.project(f[0, 4:])[0].astype(int)), 4, AMBER, -1, cv2.LINE_AA)
    x0, y0, w, h = crop
    im = im[y0:y0 + h, x0:x0 + w]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.resize(im, (1040, 585)), [cv2.IMWRITE_JPEG_QUALITY, 86])
    print(output, f"{n} variants")


if __name__ == "__main__":
    tyro.cli(main)
