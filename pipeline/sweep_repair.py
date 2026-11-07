"""Repair-or-reject sweep: move one obstacle over a grid of positions along (and across) the demo's
cube path and record, per placement, what the plain generator would do and what repair achieves.

Outcomes per placement:
  valid_as_generated            generated motion is already collision-free (no edit needed)
  repaired                      free segments re-optimized, grasp/placement kept
  repaired_other_grasp_yaw      the same, after switching to a symmetric grasp yaw of the cube
  valid_as_generated_other_grasp_yaw
  reject_obstacle_in_task_segment   obstacle hits a pinned (task-critical) segment for every grasp yaw
  repair_failed                 no candidate passed the zero-margin re-check
  invalid_placement             obstacle overlaps a fixture

Example:
    python scripts/ur_cube/sweep_repair.py --output outputs/ur_cube/repair/sweep_bottle
"""

import json
import sys
import time
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).parent))
from demo_repair import DemoRepair, World, path_metrics  # noqa: E402


def main(
    output: Path = Path("outputs/ur_cube/repair/sweep_bottle"),
    fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    laterals: tuple[float, ...] = (-0.08, -0.04, 0.0, 0.04, 0.08),
    heights: tuple[float, ...] = (0.1987,),  # defaults: the scanned bottle's collision cylinder
    radius: float = 0.0328,                   # (data/obstacles/sauce_bottle/obstacle.json)
    save_cases: bool = True,
) -> None:
    rep = DemoRepair()
    base = World.from_layout(rep.layout)
    p = rep.demo_obj[:, 4:6]
    p0, p1 = p[0], p[-1]
    d = (p1 - p0) / onp.linalg.norm(p1 - p0)
    side = onp.array([-d[1], d[0]])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in heights:
        for f in fractions:
            for lat in laterals:
                xy = p0 + f * (p1 - p0) + lat * side
                row = dict(fraction=f, lateral=lat, height=h, x=float(xy[0]), y=float(xy[1]), radius=radius)
                gap = onp.linalg.norm(base.fixtures[:, :2] - xy, axis=1) - base.fixtures[:, 4] - radius
                if (gap < 0.01).any():
                    row.update(status="invalid_placement")
                    rows.append(row)
                    print(row, flush=True)
                    continue
                world = base.with_cylinder(float(xy[0]), float(xy[1]), radius, h)
                t0 = time.time()
                out = rep.run(world)
                row.update(status=out["status"], baseline_status=out["baseline_status"], seed=out.get("seed"),
                           grasp_rank=out.get("grasp_rank"), time_s=round(time.time() - t0, 2),
                           baseline_min_clearance=out["before"]["min_obstacle_clearance"],
                           baseline_spans=out["before"]["collision_spans"])
                if out["q"] is not None:
                    row.update(min_clearance=out["after"]["min_obstacle_clearance"],
                               **path_metrics(rep, out["q_baseline"], out["q"], out["nominal"]))
                    if save_cases and out["status"].startswith("repaired"):
                        name = f"h{int(h * 100)}_f{int(f * 100):02d}_l{int(round(lat * 100)):+03d}"
                        onp.savez(output / f"{name}.npz", q_baseline=out["q_baseline"], q=out["q"],
                                  obj=rep.objects_from_q(out["q"], out["nominal"]),
                                  obj_baseline=onp.asarray(out["nominal"]["obj"].wxyz_xyz) if out["grasp_rank"] == 0 else None,
                                  phase=out["nominal"]["phase"], pinned=out["nominal"]["pinned"],
                                  held=out["nominal"]["held"], world_cyl=world.obs_cyl)
                        row["case"] = name
                rows.append(row)
                print({k: row[k] for k in ("fraction", "lateral", "height", "status", "time_s")}, flush=True)
                (output / "sweep.json").write_text(json.dumps(rows, indent=1))
    (output / "sweep.json").write_text(json.dumps(rows, indent=1))
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("counts:", counts)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
