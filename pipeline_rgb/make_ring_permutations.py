"""Trajectory interpolation with collision-object permutations for the RGB-only tasks (the cube's
make_traj_permutations.py, same output format, so make_ring_viser.py draws it like the cube's).

1. The tracked demo (cleaned; for the tape in the twin's frame, with the mug tree turned 250 deg).
2. The generator's interpolated variants: K new object starts (the generator's jitter around the
   start it uses; the nut's starts are moved into reach), the leading 60% of the path warped to each.
3. Collision permutations: the scanned bottle, mug and cereal box placed on or beside the route,
   clear of the fixture; for each placement the plain motion for M of those starts is checked and,
   where it collides, repaired (ring_repair.py). Stores the held object's path before and after.

    CUDA_VISIBLE_DEVICES=1 python scripts/ur_tape/make_ring_permutations.py --task tape \
        --output outputs/ur_tape/traj_perm/traj_perm.json
"""

import json
import sys
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).parent))
from ring_repair import REPO, RingRepair  # noqa: E402
from tape_plan import rotate_traj_about_post  # noqa: E402
from trajgen.traj_interp import traj_interp_batch  # noqa: E402

OBJECTS = ("sauce_bottle", "mug", "cereal_box")


def footprint_gap(meta: dict, xy, yaw: float, cyls) -> onp.ndarray:
    """Distance from the object's footprint (disc, or yawed box) to each fixture cylinder's disc."""
    rel = onp.asarray(cyls)[:, :2] - xy
    if meta.get("proxy", "cylinder") == "cylinder":
        return onp.linalg.norm(rel, axis=1) - meta["radius"] - onp.asarray(cyls)[:, 4]
    c, s = onp.cos(yaw), onp.sin(yaw)
    local = rel @ onp.array([[c, -s], [s, c]])
    out = onp.maximum(onp.abs(local) - onp.array(meta["size"][:2]) / 2, 0.0)
    return onp.linalg.norm(out, axis=1) - onp.asarray(cyls)[:, 4]


def main(task: str = "tape", output: Path | None = None, k: int = 12, m: int = 6, candidates: int = 6, keep: int = 3,
         seed: int = 3, xy_noise: float = 0.04, objects: tuple[str, ...] = OBJECTS,
         fraction: tuple[float, float] = (0.3, 0.7), lateral: tuple[float, float] = (-0.06, 0.06),
         wide_fraction: tuple[float, float] = (-0.2, 1.2), wide_lateral: tuple[float, float] = (-0.18, 0.18),
         route_attempts: int = 100, min_lateral: float = 0.0, append: bool = False) -> None:
    """The bottle goes on the route first (fraction / lateral of the start -> goal segment), the mug and
    cereal box anywhere around it (wide_*) when they don't fit; all clear of the fixture."""
    output = output or REPO / ("outputs/ur_tape/traj_perm/traj_perm.json" if task == "tape" else "outputs/ur_nut/traj_perm/traj_perm.json")
    rng = onp.random.default_rng(seed)
    rep = RingRepair(task)
    base = rep.world()
    demo = rep.demo_obj
    if rep.cfg["tree_yaw"]:
        demo = rotate_traj_about_post(demo, onp.radians(rep.cfg["tree_yaw"]), rep.planner.post_xy)
        demo = onp.asarray(traj_interp_batch(traj=demo, new_starts=rep.demo_obj[:1], proportion=0.6)[0], onp.float64)
    starts = []
    for _ in range(k):
        s = demo[0].copy()
        s[4:6] += onp.asarray(rep.cfg["start_offset"]) + rng.uniform(-xy_noise, xy_noise, 2)
        starts.append(s)
    fan = [onp.asarray(traj_interp_batch(traj=demo, new_starts=s[None], proportion=0.6)[0], onp.float64) for s in starts]
    p0, p1 = fan[0][0, 4:6], fan[0][-1, 4:6]
    d = (p1 - p0) / onp.linalg.norm(p1 - p0)
    perms = json.loads(output.read_text())["permutations"] if append and output.exists() else []
    for name in objects:
        meta = json.loads((REPO / f"data/obstacles/{name}/obstacle.json").read_text())
        tried = []
        for c in range(candidates):
            for attempt in range(400):
                fr, la = (fraction, lateral) if attempt < route_attempts else (wide_fraction, wide_lateral)
                lat = rng.uniform(*la)
                if abs(lat) < min_lateral:
                    continue
                xy = p0 + rng.uniform(*fr) * (p1 - p0) + lat * onp.array([-d[1], d[0]])
                yaw = float(rng.uniform(0, onp.pi if meta.get("proxy") == "box" else 2 * onp.pi))
                if (footprint_gap(meta, xy, yaw, base.fixtures) > 0.01).all():
                    break
            else:
                continue
            if meta.get("proxy") == "box":
                world = base.with_box(float(xy[0]), float(xy[1]), *meta["size"], yaw=yaw)
            else:
                world = base.with_cylinder(float(xy[0]), float(xy[1]), meta["radius"], meta["height"])
            runs = []
            for i in range(m):
                out = rep.run(world, obj_traj=fan[i])
                held = out["nominal_baseline"]["held"]
                plain = rep.objects_from_q(out["q_baseline"], out["nominal_baseline"])[held]
                fixed = rep.objects_from_q(out["q"], out["nominal"])[out["nominal"]["held"]] if out["q"] is not None else None
                runs.append(dict(start=i, status=out["status"], plain_ok=bool(out["before"]["ok"]),
                                 plain_min_clearance=out["before"]["min_obstacle_clearance"],
                                 min_clearance=out["after"]["min_obstacle_clearance"] if out.get("after") else None,
                                 plain=onp.round(plain, 5).tolist(), path=None if fixed is None else onp.round(fixed, 5).tolist()))
            n_hit = sum(not r["plain_ok"] for r in runs)
            n_fixed = sum(r["status"].startswith("repaired") for r in runs)
            n_ok = sum(r["path"] is not None for r in runs)
            print(task, name, c, onp.round(xy, 3), f"plain hits {n_hit}/{m}, repaired {n_fixed}, valid {n_ok}",
                  sorted({r['status'] for r in runs}), flush=True)
            tried.append(dict(object=name, xy=[float(xy[0]), float(xy[1])], yaw=yaw, meta=meta, runs=runs,
                              n_hit=n_hit, n_fixed=n_fixed, n_ok=n_ok))
        tried.sort(key=lambda t: (-t["n_fixed"], m - t["n_ok"]))
        perms += tried[:keep]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        task=task, demo=onp.round(demo, 5).tolist(), starts=[s.tolist() for s in starts],
        fan=[onp.round(f, 5).tolist() for f in fan], table_z=rep.planner.table_z, fixtures=base.fixtures.tolist(),
        tree_yaw=rep.cfg["tree_yaw"], permutations=perms)))
    print(f"{output}: {len(perms)} placements")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
