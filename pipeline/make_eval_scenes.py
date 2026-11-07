"""Held-out evaluation scenes for closed-loop policy evaluation (eval_ur5e_cube_policy.py).

Each scene fixes the cube's start offset on the pick fixture (dx, dy, yaw; the generator's ranges)
and, for the obstacle set, a bottle placement. Every policy is evaluated on the same scenes and,
batch by batch, the same lighting / camera / table draws (seeded in the evaluator).

Sets:
  clean     cube start only, no obstacle
  blocked   a bottle on the route such that the plain generator's motion collides with it, but a
            collision-free demonstration exists (demo_repair finds one): the scenes where an edit
            to the scene matters and the task is still solvable

Example:
    python scripts/ur_cube/make_eval_scenes.py --n 96 --output outputs/ur_cube/eval/scenes.json
"""

import json
import sys
from pathlib import Path

import numpy as onp
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from demo_repair import FIXTURE_RADIUS, DemoRepair, World  # noqa: E402

BOTTLE = json.loads(Path("data/obstacles/sauce_bottle/obstacle.json").read_text())


def warp(rep: DemoRepair, dx: float, dy: float, yaw: float) -> onp.ndarray:
    """Same warp as the generator: rotate the start about base z, shift it, interpolate 60%."""
    from trajgen.traj_interp import traj_interp_batch
    start = rep.demo_obj[0]
    q = start[:4]
    rot = R.from_euler("z", yaw) * R.from_quat([q[1], q[2], q[3], q[0]])
    x, y, z, w = rot.as_quat()
    ns = onp.array([[w, x, y, z, start[4] + dx, start[5] + dy, start[6]]])
    return onp.asarray(traj_interp_batch(traj=rep.demo_obj, new_starts=ns, proportion=0.6))[0]


def sample_bottle(traj, rng, fraction=(0.3, 0.7), lateral=(-0.06, 0.06), fixtures=None):
    p0, p1 = traj[0, 4:6], traj[-1, 4:6]
    d = (p1 - p0) / onp.linalg.norm(p1 - p0)
    for _ in range(50):
        xy = p0 + rng.uniform(*fraction) * (p1 - p0) + rng.uniform(*lateral) * onp.array([-d[1], d[0]])
        if (onp.linalg.norm(fixtures[:, :2] - xy, axis=1) - FIXTURE_RADIUS - BOTTLE["radius"] > 0.01).all():
            return xy
    return None


def main(n: int = 96, seed: int = 12345, output: Path = Path("outputs/ur_cube/eval/scenes.json"),
         xy_noise: float = 0.03, yaw_noise_deg: float = 30.0) -> None:
    rng = onp.random.default_rng(seed)
    rep = DemoRepair()
    base = World.from_layout(rep.layout)
    start = lambda: (float(rng.uniform(-xy_noise, xy_noise)), float(rng.uniform(-xy_noise, xy_noise)),  # noqa: E731
                     float(onp.radians(rng.uniform(-yaw_noise_deg, yaw_noise_deg))))
    clean = []
    for i in range(n):
        dx, dy, yaw = start()
        clean.append(dict(id=f"clean_{i:03d}", dx=dx, dy=dy, yaw=yaw, bottle_xy=None))
    blocked, tries = [], 0
    while len(blocked) < n:
        tries += 1
        dx, dy, yaw = start()
        traj = warp(rep, dx, dy, yaw)
        xy = sample_bottle(traj, rng, fixtures=base.fixtures)
        if xy is None:
            continue
        world = base.with_cylinder(float(xy[0]), float(xy[1]), BOTTLE["radius"], BOTTLE["height"])
        out = rep.run(world, obj_traj=traj)
        if out["before"]["ok"] or out["q"] is None:  # plain motion already clear, or unsolvable
            continue
        blocked.append(dict(id=f"blocked_{len(blocked):03d}", dx=dx, dy=dy, yaw=yaw, bottle_xy=[float(xy[0]), float(xy[1])],
                            repair_status=out["status"], plain_min_clearance=out["before"]["min_obstacle_clearance"]))
        print(len(blocked), tries, out["status"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(seed=seed, bottle=BOTTLE, clean=clean, blocked=blocked,
                                      blocked_draws=tries), indent=1))
    print(f"{output}: {len(clean)} clean, {len(blocked)} blocked ({tries} draws)")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
