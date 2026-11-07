"""Clean up the tracked nut trajectory for the generator (as clean_tape_track.py does for the roll).

  * Level: the human carries the nut flat (the tracked tilt stays small, printed below), and a
    flat nut is what a top-down grasp holds, so every pose is levelled (axis vertical).
  * Rest at the start: centre half a thickness above the bench before it moves.
  * On the rod: the hole leaves ~3 mm around the rod, about the tracking error, so the final xy is
    snapped onto the measured rod axis, blended in over the approach to the rod (the last
    `snap_s` seconds).
  * Onto the rod from above: a nut can only go onto the vertical rod from over its tip, but where
    only the front camera saw the nut (bridge_one_view.py) its height is about a centimetre
    uncertain and the bridged path reaches the rod slightly below the tip. So within `enter_r` of
    the rod axis the nut is kept at least `tip_clear` above the tip, and the last `descent_s`
    seconds are a straight descent down the rod axis into the tracked final pose.
  * Spin about the axis is unobservable for a round nut; the frame is kept fixed.

    python scripts/ur_tape/clean_nut_track.py --track outputs/ur_nut/traj_ep4_place.json \
        --output outputs/ur_nut/traj_ep4_place_clean.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))


def main(track: Path = Path("outputs/ur_nut/traj_ep4_place.json"),
         output: Path = Path("outputs/ur_nut/traj_ep4_place_clean.json"),
         scene: Path = Path("outputs/ur_nut/scene.json"), rest_tol: float = 0.004, snap_s: float = 0.6,
         gripper_close: float = 0.55, enter_r: float = 0.05, tip_clear: float = 0.006, descent_s: float = 0.35) -> None:
    d = json.loads(track.read_text())
    sc = json.loads(scene.read_text())
    T = np.asarray(d["T_base_obj"], np.float64)
    fps = d["fps"]
    pos = T[:, 4:].copy()
    axes = Rotation.from_quat(T[:, [1, 2, 3, 0]]).as_matrix()[:, :, 2]
    tilt = np.degrees(np.arccos(np.clip(np.abs(axes[:, 2]), -1, 1)))
    dims = d["dims"]

    moved = np.linalg.norm(pos - pos[0], axis=1) > rest_tol
    k_rest = int(np.argmax(moved)) if moved.any() else len(pos)
    dz = sc["table_z"] + dims["w"] / 2 - pos[:k_rest, 2].mean()
    pos[:, 2] += dz * np.clip(1 - (np.arange(len(pos)) - k_rest) / 10, 0, 1)

    rod = sc["bolt_stand"]
    off = np.array(rod["xy"]) - pos[-1, :2]
    n_snap = int(snap_s * fps)
    ramp = np.clip((np.arange(len(pos)) - (len(pos) - n_snap)) / max(n_snap - 1, 1), 0, 1)
    ramp = ramp * ramp * (3 - 2 * ramp)
    pos[:, :2] += off[None] * ramp[:, None]

    # onto the rod from above
    tip = rod["rod"]["z1"]
    z_above = tip + tip_clear + dims["w"] / 2
    n_d = int(descent_s * fps)
    k_d = len(pos) - n_d
    dist = np.linalg.norm(pos[:, :2] - np.array(rod["xy"]), axis=1)
    near = np.clip((enter_r - dist) / (0.4 * enter_r), 0, 1)  # 0 far away -> 1 within 0.6 enter_r
    lift = np.maximum(z_above - pos[:, 2], 0) * near
    lift[k_d:] = 0
    pos[:, 2] += lift
    end = pos[-1].copy()
    start_d = np.r_[rod["xy"], max(z_above, pos[k_d - 1, 2])]
    for i in range(n_d):
        t = (i + 1) / n_d
        t = t * t * (3 - 2 * t)
        pos[k_d + i] = start_d + t * (end - start_d)
    pos[k_d - 1, :2] = pos[k_d - 1, :2] + 0.5 * (np.array(rod["xy"]) - pos[k_d - 1, :2])
    max_lift = float(lift.max())

    q = np.tile([1.0, 0, 0, 0], (len(pos), 1))
    T_clean = np.concatenate([q, pos], 1)
    clearance = dims["r_in"] - rod["rod"]["radius"]
    print(f"steps {len(T_clean)}; rest {k_rest}, height correction {dz * 1000:+.1f} mm; tracked tilt median "
          f"{np.median(tilt):.1f} deg, max {tilt.max():.1f} deg (levelled); final xy {np.linalg.norm(off) * 1000:.1f} mm "
          f"off the rod axis (hole clearance {clearance * 1000:.1f} mm), snapped over {snap_s} s; "
          f"final height {pos[-1, 2]:.3f} (rod tip {rod['rod']['z1']:.3f}); lifted up to {max_lift * 1000:.1f} mm "
          f"to clear the tip, straight {descent_s} s descent")
    out = dict(d, T_base_obj=T_clean.tolist(), gripper_close=gripper_close,
               grasp_T_obj_tcp=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, dims["w"] / 2 + 0.017],  # nominal; the planner picks its own
               cleanup=dict(rest_steps=k_rest, rest_height_correction_m=float(dz), tilt_median_deg=float(np.median(tilt)),
                            tilt_max_deg=float(tilt.max()), final_offset_to_rod_mm=float(np.linalg.norm(off) * 1000),
                            hole_clearance_mm=float(clearance * 1000), snap_s=snap_s, max_lift_mm=max_lift * 1000,
                            descent_s=descent_s, tip_clear_m=tip_clear))
    out.pop("full_track", None)
    output.write_text(json.dumps(out))


if __name__ == "__main__":
    tyro.cli(main)
