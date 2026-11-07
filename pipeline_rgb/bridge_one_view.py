"""Bridge a stretch of a track where only one calibrated view sees the object (the nut task: the
human's hand hides the nut from the right camera between lifting it and letting go on the rod).

Each bridged frame's centre lies on the seeing camera's ray through the object's colour blob, so
it reprojects exactly in that view. The depth along the ray is interpolated between the two anchor
poses at the gap's ends (both views agree there), in proportion to the blob's travelled image path,
so a pause in the image (the nut held on the rod) keeps its depth. The axis is kept vertical; the
nut is levelled in clean_nut_track.py anyway.

    python scripts/ur_tape/bridge_one_view.py --track outputs/ur_nut/traj_ep4_place.json --episode 4 \
        --gap 183 225 --view 0 --output outputs/ur_nut/traj_ep4_place.json
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from tape_model import COLOUR_MASKS, TapeDims, angles_from_axis, skin_mask  # noqa: E402
from track_tape_rgb import read_all, segment_and_smooth, static_masks  # noqa: E402


def main(track: Path, episode: int, gap: tuple[int, int], output: Path, view: int = 0, colour: str = "green",
         dims_file: Path = Path("outputs/ur_nut/nut_dims.json"), search_px: int = 120) -> None:
    d = json.loads(track.read_text())
    cams = load_cams()
    cam = cams[view]
    vid = read_all(episode, view)
    dj = json.loads(dims_file.read_text())
    p0 = np.r_[dj["poses"][0]["centre"], angles_from_axis(np.array(dj["poses"][0]["axis"]))]
    st = static_masks(cams, [read_all(episode, i)[0] if i != view else vid[0] for i in range(2)],
                      TapeDims.from_json(dj), colour, p0)[view]
    C = np.array(d["full_track"]["centre"])
    A = np.array(d["full_track"]["axis"])
    a, b = gap
    prev = cam.project(C[a])[0]
    uv = []
    for f in range(a, b + 1):
        m = COLOUR_MASKS[colour](vid[f]) & ~st & ~skin_mask(vid[f])
        n, _, s, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        cand = [(np.linalg.norm(cen[k] - prev), cen[k]) for k in range(1, n) if s[k, 4] >= 25]
        cand = [c for c in cand if c[0] < search_px]
        if cand:
            prev = min(cand, key=lambda c: c[0])[1]
        uv.append(prev.copy())
    uv = np.array(uv)
    # anchors: blob centroid offset vs the projected centre at each end, removed along the gap
    off_a, off_b = cam.project(C[a])[0] - uv[0], cam.project(C[b])[0] - uv[-1]
    path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    s = path / max(path[-1], 1e-9)
    o = cam.T_cam2base[:3, 3]
    lam_a, lam_b = np.linalg.norm(C[a] - o), np.linalg.norm(C[b] - o)
    for k, f in enumerate(range(a, b + 1)):
        _, dvec = cam.ray(uv[k] + off_a + s[k] * (off_b - off_a))
        C[f] = o + dvec * (lam_a + s[k] * (lam_b - lam_a))
        A[f] = [0.0, 0.0, 1.0 if A[a][2] >= 0 else -1.0]
    f0, f1, T, win = segment_and_smooth(C, A, d["fps"])
    d.update(T_base_obj=T.tolist(), frames=[f0, f1],
             full_track=dict(centre=np.round(C, 5).tolist(), axis=np.round(A, 4).tolist()),
             source=d["source"] + f"; frames {a}-{b} bridged from view {view} only (right view occluded by the hand)",
             bridged=dict(frames=[a, b], view=view, image_path_px=float(path[-1])))
    output.write_text(json.dumps(d))
    print(f"bridged {a}-{b} from view {view}: image path {path[-1]:.0f} px, depth {lam_a:.3f} -> {lam_b:.3f} m; "
          f"segment {f0}-{f1} ({len(T)} steps), start {T[0, 4:].round(3)}, end {T[-1, 4:].round(3)}")


if __name__ == "__main__":
    tyro.cli(main)
