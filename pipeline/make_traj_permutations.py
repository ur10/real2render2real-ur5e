"""Trajectory interpolation with collision-object permutations: the data behind the 3D trajectory
figure / viser recording (make_traj_viser.py).

1. The tracked demo (cleaned cube poses).
2. The generator's interpolated variants: K new cube starts on the pick fixture (the generator's
   ranges), the leading 60% of the path warped to each (trajgen.traj_interp_batch).
3. Collision permutations: scanned objects (bottle, mug, cereal box) placed on the route; for each
   placement, the generator's plain motion for M of those starts is checked and, where it collides,
   repaired (demo_repair.py). Stores the held cube's path before and after.

Example (JAX on GPU 1):
    CUDA_VISIBLE_DEVICES=1 python scripts/ur_cube/make_traj_permutations.py \
        --output outputs/ur_cube/traj_perm/traj_perm.json
"""

import json
import sys
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).parent))
from demo_repair import FIXTURE_RADIUS, DemoRepair, World  # noqa: E402
from make_eval_scenes import warp  # noqa: E402

OBJECTS = ("sauce_bottle", "mug", "cereal_box")


def footprint_gap(meta: dict, xy, yaw: float, centers) -> onp.ndarray:
    """Distance from the object's footprint (cylinder disc, or yawed box) to each fixture disc."""
    rel = onp.asarray(centers)[:, :2] - xy
    if meta.get("proxy", "cylinder") == "cylinder":
        return onp.linalg.norm(rel, axis=1) - meta["radius"] - FIXTURE_RADIUS
    c, s = onp.cos(yaw), onp.sin(yaw)
    local = rel @ onp.array([[c, -s], [s, c]])  # world -> box frame
    out = onp.maximum(onp.abs(local) - onp.array(meta["size"][:2]) / 2, 0.0)
    return onp.linalg.norm(out, axis=1) - FIXTURE_RADIUS


def main(output: Path = Path("outputs/ur_cube/traj_perm/traj_perm.json"), k: int = 12, m: int = 6,
         candidates: int = 8, keep: int = 3, seed: int = 3, xy_noise: float = 0.03, yaw_noise_deg: float = 30.0,
         objects: tuple[str, ...] = OBJECTS, append: bool = False,
         fraction: tuple[float, float] = (0.3, 0.7), lateral: tuple[float, float] = (-0.05, 0.05),
         wide_fraction: tuple[float, float] = (-0.2, 1.2), wide_lateral: tuple[float, float] = (-0.16, 0.16),
         route_attempts: int = 100, min_lateral: float = 0.0) -> None:
    """Objects that fit between the fixtures (the bottle) are placed on the route (fraction / lateral of
    the start -> goal segment); bigger ones anywhere around it (wide_*), clear of both fixtures."""
    rng = onp.random.default_rng(seed)
    rep = DemoRepair()
    base = World.from_layout(rep.layout)
    starts = [(float(rng.uniform(-xy_noise, xy_noise)), float(rng.uniform(-xy_noise, xy_noise)),
               float(onp.radians(rng.uniform(-yaw_noise_deg, yaw_noise_deg)))) for _ in range(k)]
    fan = [warp(rep, *s) for s in starts]
    p0, p1 = rep.demo_obj[0, 4:6], rep.demo_obj[-1, 4:6]
    d = (p1 - p0) / onp.linalg.norm(p1 - p0)
    old = json.loads(output.read_text())["permutations"] if append and output.exists() else []
    perms = [p for p in old if p["object"] not in objects]
    for name in objects:
        meta = json.loads(Path(f"data/obstacles/{name}/obstacle.json").read_text())
        tried = []
        for c in range(candidates):
            for attempt in range(400):
                fr, la = (fraction, lateral) if attempt < route_attempts else (wide_fraction, wide_lateral)
                lat = rng.uniform(*la)
                if abs(lat) < min_lateral:
                    continue
                xy = p0 + rng.uniform(*fr) * (p1 - p0) + lat * onp.array([-d[1], d[0]])
                yaw = float(rng.uniform(0, onp.pi if meta.get("proxy") == "box" else 2 * onp.pi))  # cylinders: visual only
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
            print(name, c, onp.round(xy, 3), f"plain hits {n_hit}/{m}, repaired {n_fixed}, valid {n_ok}", flush=True)
            tried.append(dict(object=name, xy=[float(xy[0]), float(xy[1])], yaw=yaw, meta=meta, runs=runs,
                              n_hit=n_hit, n_fixed=n_fixed, n_ok=n_ok))
        # keep the placements that show the most repairs (ties: fewest rejections)
        tried.sort(key=lambda t: (-t["n_fixed"], m - t["n_ok"]))
        perms += tried[:keep]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(
        demo=onp.round(rep.demo_obj, 5).tolist(), starts=starts, fan=[onp.round(f, 5).tolist() for f in fan],
        table_z=rep.layout["table_z"], fixtures=base.fixtures.tolist(), permutations=perms)))
    print(f"{output}: {len(perms)} placements")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
