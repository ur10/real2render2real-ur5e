"""Check a roll track by drawing the posed model's outline on both camera views, frame by frame.

    python scripts/ur_tape/overlay_tape_track.py --track outputs/ur_tape/traj_ep1_hang.json \
        --output outputs/ur_tape/track_overlay_ep1.mp4
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
from tape_model import BLUE, CORE, TapeDims, axis_from_angles, frame_from_axis, render_labels, sample_surface  # noqa: E402
from track_tape_rgb import read_all  # noqa: E402


def main(track: Path = Path("outputs/ur_tape/traj_ep1_hang.json"), output: Path = Path("outputs/ur_tape/track_overlay_ep1.mp4"),
         episode: int = 1, pad: int = 15, crop: tuple[int, int, int, int] = (0, 160, 1280, 720), stills: int = 6) -> None:
    d = json.loads(track.read_text())
    cams = load_cams()
    dims = TapeDims(**d["dims"])
    pts, lab = sample_surface(dims)
    ft = d["full_track"]
    f0, f1 = d["frames"]
    vids = [read_all(episode, c) for c in range(2)]
    frames = range(max(f0 - pad, 0), min(f1 + pad, len(ft["centre"]) - 1) + 1)
    tmp = Path(tempfile.mkdtemp())
    x0, y0, x1, y1 = crop
    for k, f in enumerate(frames):
        c, a = np.array(ft["centre"][f]), np.array(ft["axis"][f])
        R = frame_from_axis(a)
        tiles = []
        for i, cam in enumerate(cams):
            im = vids[i][f].copy()
            m = render_labels(cam, R, c, pts, lab, (0, 0, cam.W, cam.H))
            for lbl, col in ((BLUE, (0, 255, 255)), (CORE, (255, 0, 255))):
                cnts, _ = cv2.findContours((m == lbl).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cnts, -1, col, 2)
            uv = cam.project(np.stack([c, c + a * 0.08]))[0].astype(int)
            cv2.arrowedLine(im, tuple(uv[0]), tuple(uv[1]), (0, 0, 255), 2, tipLength=0.25)
            tiles.append(cv2.resize(im[y0:y1, x0:x1], ((x1 - x0) // 2, (y1 - y0) // 2)))
        frame = np.hstack(tiles)
        txt = f"ep{episode} frame {f}  1-IoU {d['fit_1_minus_iou'][f]:.2f}" + ("  (demo segment)" if f0 <= f <= f1 else "")
        cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(tmp / f"{k:05d}.png"), frame)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "15", "-i", str(tmp / "%05d.png"), "-pix_fmt", "yuv420p",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(output)], check=True)
    # a strip of stills across the segment for quick viewing
    sel = np.linspace(f0, f1, stills).astype(int)
    ims = [cv2.imread(str(tmp / f"{list(frames).index(f):05d}.png")) for f in sel]
    cv2.imwrite(str(output.with_suffix(".jpg")), np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]))
    print(output)


if __name__ == "__main__":
    tyro.cli(main)
