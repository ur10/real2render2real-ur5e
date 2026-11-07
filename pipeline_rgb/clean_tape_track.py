"""Clean up the tracked roll trajectory for the generator (the counterpart of demo_repair's
level_poses / minimal_yaw for the cube).

  * Rest at the start: the roll lies flat, so its axis is snapped to vertical and its centre to half
    a width above the bench while it has not moved yet.
  * Flip symmetry: the roll looks the same with its axis reversed. The human tilted it from flat to
    hanging through `angle(a0, a_T)`; reversing the hung pose's axis gives the same placement through
    `angle(a0, -a_T)`. The smaller tilt is used (here 64 deg instead of 116 deg), which is also what
    makes a grasp on the top face end on the outside of the roll instead of between roll and post.
    The tilt follows the human's timing (tilt progress), along the great circle to the chosen axis.
  * Check: in the hung pose the branch must thread the core: distance from the roll's axis line to
    the branch line, at the roll, below r_in - branch radius.

    python scripts/ur_tape/clean_tape_track.py --track outputs/ur_tape/traj_ep1_hang.json \
        --output outputs/ur_tape/traj_ep1_hang_clean.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import tyro
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from track_tape_rgb import minimal_twist_frames  # noqa: E402


def slerp_vec(a, b, t):
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def branch_threading(centre, axis, branch, r_in):
    """Distance between the roll axis line and the branch segment's line near the roll."""
    a, b = np.array(branch["root"]), np.array(branch["tip"])
    u = (b - a) / np.linalg.norm(b - a)
    n = np.cross(axis, u)
    if np.linalg.norm(n) < 1e-6:
        d = np.linalg.norm(np.cross(centre - a, u))
    else:
        d = abs((centre - a) @ n / np.linalg.norm(n))
    s = (centre - a) @ u  # where along the branch the roll sits
    return float(d), float(s), float(np.degrees(np.arccos(abs(axis @ u)))), r_in - branch["radius"]


def main(track: Path = Path("outputs/ur_tape/traj_ep1_hang.json"),
         output: Path = Path("outputs/ur_tape/traj_ep1_hang_clean.json"),
         scene: Path = Path("outputs/ur_tape/scene.json"), rest_tol: float = 0.004,
         gripper_close: float = 0.62, release_along: float | None = 0.045) -> None:
    """release_along: end the demo (release) at the first step where the roll, threaded on the branch,
    is this far along it from the root (m); None keeps the human's full push to the root."""
    d = json.loads(track.read_text())
    sc = json.loads(scene.read_text())
    T = np.asarray(d["T_base_obj"], np.float64)
    pos = T[:, 4:].copy()
    R = Rotation.from_quat(T[:, [1, 2, 3, 0]]).as_matrix()
    axes = R[:, :, 2]
    dims = d["dims"]

    # rest at the start
    moved = np.linalg.norm(pos - pos[0], axis=1) > rest_tol
    k_rest = int(np.argmax(moved)) if moved.any() else len(pos)
    a0 = np.array([0.0, 0.0, 1.0]) * np.sign(axes[0, 2])
    rest_z = sc["table_z"] + dims["w"] / 2
    dz = rest_z - pos[:k_rest, 2].mean()
    ramp = np.clip(1 - (np.arange(len(pos)) - k_rest) / 10, 0, 1)  # blend the height correction out
    pos[:, 2] += dz * ramp

    # flip symmetry: pick the smaller tilt to the hung pose
    aT = axes[-5:].mean(0); aT /= np.linalg.norm(aT)
    tilt_human, tilt_flip = np.degrees(np.arccos(np.clip(a0 @ aT, -1, 1))), np.degrees(np.arccos(np.clip(-a0 @ aT, -1, 1)))
    target = -aT if tilt_flip < tilt_human else aT
    prog = np.arccos(np.clip(axes @ a0, -1, 1)) / np.radians(tilt_human)
    prog = np.clip(np.maximum.accumulate(uniform_filter1d(np.clip(prog, 0, 1), 5)), 0, 1)
    prog[:k_rest] = 0
    new_axes = np.stack([slerp_vec(a0, target, p) for p in prog])
    Rs = minimal_twist_frames(new_axes)
    q = Rotation.from_matrix(Rs).as_quat()[:, [3, 0, 1, 2]]
    T_clean = np.concatenate([q, pos], 1)

    br = sc["mug_tree"]["branches"][sc["mug_tree"]["target_branch"]]
    n_full = len(T_clean)
    if release_along is not None:
        thr = [branch_threading(pos[i], new_axes[i], br, dims["r_in"]) for i in range(len(pos))]
        on = [i for i, (dd, s_, _, lim) in enumerate(thr) if dd < lim and s_ <= release_along and i > len(pos) // 2]
        if on:
            T_clean, pos, new_axes = T_clean[:on[0] + 1], pos[:on[0] + 1], new_axes[:on[0] + 1]
    print(f"steps {len(T_clean)} of {n_full} (release at {release_along} m along the branch)")
    dist, along, ang, limit = branch_threading(pos[-1], new_axes[-1], br, dims["r_in"])
    print(f"rest steps {k_rest}, height correction {dz * 1000:+.1f} mm; tilt human {tilt_human:.0f} deg, "
          f"flipped {tilt_flip:.0f} deg -> using {'flipped' if target is not aT else 'human'}")
    print(f"hung pose: roll axis {dist * 1000:.1f} mm from the branch line (core clearance {limit * 1000:.1f} mm), "
          f"{along * 100:.1f} cm along the branch, axis {ang:.1f} deg off the branch direction")
    from tape_plan import TapePlanner  # grasp at azimuth 0 as the file's nominal grasp (the planner picks its own)
    g = TapePlanner.grasp(SimpleGrasp(dims), 0.0)
    out = dict(d, T_base_obj=T_clean.tolist(), grasp_T_obj_tcp=np.asarray(g.wxyz_xyz).tolist(),
               gripper_close=gripper_close,
               cleanup=dict(rest_steps=k_rest, steps_kept=len(T_clean), steps_tracked=n_full, release_along_m=release_along, rest_height_correction_m=float(dz), tilt_human_deg=float(tilt_human),
                            tilt_used_deg=float(min(tilt_human, tilt_flip)),
                            hung_axis_to_branch_mm=dist * 1000, core_clearance_mm=limit * 1000,
                            hung_along_branch_cm=along * 100, hung_axis_vs_branch_deg=ang))
    out.pop("full_track", None)
    output.write_text(json.dumps(out))


class SimpleGrasp:
    """Just enough of TapePlanner for its grasp() (no robot model)."""
    def __init__(self, dims):
        self.dims, self.tcp_above = dims, 0.010


if __name__ == "__main__":
    tyro.cli(main)
