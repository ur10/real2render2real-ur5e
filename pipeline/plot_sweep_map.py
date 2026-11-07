"""Top-down map of a repair sweep (sweep_repair.py): each obstacle placement coloured by outcome.

Example:
    python scripts/ur_cube/plot_sweep_map.py --sweep outputs/ur_cube/repair/sweep_bottle
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "valid_as_generated": ("#9aa0a6", "already clear"),
    "repaired": ("#2ca02c", "repaired"),
    "repaired_other_grasp_yaw": ("#98df8a", "repaired (other grasp yaw)"),
    "reject_obstacle_in_task_segment": ("#d62728", "rejected: blocks grasp or set-down"),
    "repair_failed": ("#000000", "repair failed"),
}


def main(
    sweep: Path = Path("outputs/ur_cube/repair/sweep_bottle"),
    layout: Path = Path("outputs/ur_cube/scene_layout.json"),
    demo: Path = Path("outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json"),
) -> None:
    rows = json.loads((sweep / "sweep.json").read_text())
    lay = json.loads(layout.read_text())
    T = np.array(json.loads(demo.read_text())["T_base_obj"])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    for f in lay["fixtures"]:
        ax.add_patch(plt.Circle(f["center_xy"], 0.045, color="#f28e2b", alpha=0.35))
    ax.plot(T[:, 4], T[:, 5], color="k", lw=1, ls="--", label="demo cube path")
    for s, (c, lab) in STYLE.items():
        pts = [(r["x"], r["y"]) for r in rows if r["status"] == s]
        if pts:
            ax.scatter(*zip(*pts), s=90, color=c, edgecolor="k", lw=0.5, label=f"{lab} ({len(pts)})")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 0.3)
    ax.set_ylim(-0.9, -0.62)
    ax.set_xlabel("x (m, robot base frame)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False)
    r = next(r for r in rows if "radius" in r)
    ax.set_title(f"Obstacle placement (r {r['radius'] * 100:.1f} cm, h {r['height'] * 100:.0f} cm) vs outcome", fontsize=10)
    fig.tight_layout()
    fig.savefig(sweep / "sweep_map.png", dpi=140)
    print(sweep / "sweep_map.png")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
