"""Which turns of the mug tree about its post make the tracked hang plannable for the UR5e?

The human hung the roll on the branch facing away from the robot. Turning the tree about its post
keeps everything the human did relative to the tree (the roll's path onto the branch, the hung
pose); the start is warped back to the real start pose as the generator does (trajgen, leading
60%). For each turn, every grasp azimuth is planned and checked (tape_plan.py).

    CUDA_VISIBLE_DEVICES=1 python scripts/ur_tape/sweep_tree_yaw.py --output outputs/ur_tape/tree_yaw_sweep.json
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as onp  # noqa: E402
import tyro  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from tape_plan import TapePlanner, rotate_about_post, rotate_traj_about_post  # noqa: E402
from trajgen.traj_interp import traj_interp_batch  # noqa: E402


def turned_traj(T: onp.ndarray, psi: float, post_xy) -> onp.ndarray:
    rot = rotate_traj_about_post(T, psi, post_xy)
    return onp.asarray(traj_interp_batch(traj=rot, new_starts=T[:1], proportion=0.6)[0], onp.float64)


def main(traj: Path = Path("outputs/ur_tape/traj_ep1_hang_clean.json"), output: Path = Path("outputs/ur_tape/tree_yaw_sweep.json"),
         step_deg: float = 30.0, yaw_range: tuple[float, float] = (0.0, 330.0)) -> None:
    T = onp.asarray(json.loads(traj.read_text())["T_base_obj"], onp.float64)
    rows = []
    planner = None
    for yaw in onp.arange(yaw_range[0], yaw_range[1] + 1e-6, step_deg):
        planner = TapePlanner(tree_yaw_deg=float(yaw)) if planner is None else planner
        planner.scene = rotate_about_post(json.loads(Path("outputs/ur_tape/scene.json").read_text()), onp.radians(yaw))
        from tape_plan import capsules_from_scene
        planner.caps, planner.base_cyl = capsules_from_scene(planner.scene)
        Ty = turned_traj(T, onp.radians(yaw), planner.scene["mug_tree"]["post"]["xy"])
        p = planner.plan(Ty)
        az = planner.scene["mug_tree"]["branches"]["B"]["azimuth_deg"]
        n_ok = sum(t["status"] == "ok" for t in p["tries"])
        rows.append(dict(tree_yaw_deg=float(yaw), branch_azimuth_deg=az, best=p["status"], n_grasps_ok=n_ok,
                         best_azimuth=p["azimuth_deg"], tree_clearance=round(p["tree_clearance"], 4), tries=p["tries"]))
        best_try = min(p["tries"], key=lambda t: (t["status"] != "ok", t["ik_err_mm"]))
        print(f"tree yaw {yaw:5.0f}  branch az {az:6.1f}  best {p['status']:18s} grasps ok {n_ok:2d}/{len(p['tries'])}  "
              f"lowest IK err {best_try['ik_err_mm']} mm ({best_try['ik_phase']}), tree clr {best_try['tree_clearance']} ({best_try['tree_phase']})",
              flush=True)
    output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
