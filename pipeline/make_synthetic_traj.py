"""Synthetic fixture-to-fixture cube trajectory in the UR base frame, for testing the simulator
before the RSRD track exists. Same file format the tracked trajectory will be converted to:

    {"T_base_obj": [[w, x, y, z, x, y, z], ...],   # cube pose per sim step, metres
     "grasp_T_obj_tcp": [w, x, y, z, x, y, z],      # TCP pose in the cube frame
     "gripper_close": float,                        # Robotiq finger_joint when closed (rad)
     "source": str}

Example:
    python scripts/ur_cube/make_synthetic_traj.py --output outputs/ur_cube/traj_synthetic.json
"""

import json
from pathlib import Path

import numpy as np
import tyro

CUBE = 0.04


def min_jerk(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def main(
    output: Path,
    layout: Path = Path("outputs/ur_cube/scene_layout.json"),
    pick_fixture: int = 1,
    place_fixture: int = 0,
    lift: float = 0.12,
    steps: int = 90,
    gripper_close: float = 0.42,  # stock 2F-85 at ~4 cm opening
) -> None:
    lay = json.loads(layout.read_text())
    f_pick, f_place = lay["fixtures"][pick_fixture], lay["fixtures"][place_fixture]
    p0 = np.array([*f_pick["center_xy"], f_pick["top_z"] + CUBE / 2])
    p1 = np.array([*f_place["center_xy"], f_place["top_z"] + CUBE / 2])

    # Lift (25%), carry (50%), lower (25%), each with minimum-jerk timing.
    n_up, n_lower = steps // 4, steps // 4
    n_carry = steps - n_up - n_lower
    up = p0 + np.outer(min_jerk(n_up), [0, 0, lift])
    top0, top1 = p0 + [0, 0, lift], p1 + [0, 0, lift]
    carry = top0 + np.outer(min_jerk(n_carry), top1 - top0)
    lower = top1 + np.outer(min_jerk(n_lower), p1 - top1)
    pos = np.concatenate([up, carry, lower])
    traj = np.concatenate([np.tile([1.0, 0, 0, 0], (len(pos), 1)), pos], axis=1)

    out = {
        "T_base_obj": traj.tolist(),
        # TCP z points down (rotate 180 deg about x), 1 cm above the cube centre.
        "grasp_T_obj_tcp": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.01],
        "gripper_close": gripper_close,
        "source": f"synthetic: fixture {pick_fixture} -> {place_fixture}, lift {lift} m, {steps} steps",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out))
    print(f"start {p0.round(4)} -> end {p1.round(4)}, {len(traj)} steps; wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
