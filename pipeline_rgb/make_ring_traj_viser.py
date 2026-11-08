"""Trajectory interpolation + collision-object permutations in viser for the RGB-only tasks (the
cube's make_traj_viser.py with the measured object and fixture; data from make_ring_permutations.py).

  demo   the tracked object trajectory with 6-DoF frames along it
  fan    the generator's interpolated variants: new starts (6-DoF frames) and their paths
  perm_<object>  a scanned object on / beside the route: the plain variants that collide (red) and
         the variants the generator keeps after repair (amber)
  all    one looping recording: demo -> fan -> each placement, the object riding one kept path

    uv run --no-project --with viser==0.2.23 --with trimesh --with pillow --with tyro \
        python scripts/ur_tape/make_ring_traj_viser.py --task tape --output-dir outputs/ur_tape/traj_perm/viser
"""

import json
import sys
from pathlib import Path

import numpy as np
import tyro
import viser

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ur_cube"))
from make_traj_viser import AMBER, RED, TEAL, load_object, set_visible, yaw_wxyz  # noqa: E402
from ring_viser import REPO, TASKS, add_bench, add_fixture, add_object  # noqa: E402


class Scene:
    def __init__(self, server, task: str, data: dict, sp):
        self.s, self.d, self.task = server, data, task
        add_bench(server, sp, TASKS[task]["centre"], radius=0.9)
        add_fixture(server, task, data.get("tree_yaw", 0.0))

    def path(self, name, traj, color, width=3.0, visible=True):
        return self.s.scene.add_spline_catmull_rom(name, positions=np.asarray(traj)[:, 4:], color=color, line_width=width,
                                                   visible=visible)

    def axes(self, name, poses, length=0.025, visible=True):
        p = np.asarray(poses)
        return self.s.scene.add_batched_axes(name, batched_wxyzs=p[:, :4], batched_positions=p[:, 4:], axes_length=length,
                                             axes_radius=length * 0.07, visible=visible)

    def demo(self, visible=True):
        dm = np.asarray(self.d["demo"])
        return [self.path("/demo/path", dm, TEAL, 4.0, visible), self.axes("/demo/axes", dm[::5], visible=visible)]

    def fan(self, visible=True):
        fan = [np.asarray(f) for f in self.d["fan"]]
        hs = [self.path(f"/fan/path_{i}", f, AMBER, 2.5, visible) for i, f in enumerate(fan)]
        hs.append(self.axes("/fan/starts", np.stack([f[0] for f in fan]), visible=visible))
        return hs

    def placement(self, i: int, p: dict, visible=True):
        tz = self.d["table_z"]
        hs = [self.s.scene.add_mesh_trimesh(f"/perm_{i}/object", load_object(p["object"]), wxyz=yaw_wxyz(p["yaw"]),
                                            position=(p["xy"][0], p["xy"][1], tz), visible=visible)]
        for r in p["runs"]:
            if not r["plain_ok"]:
                hs.append(self.path(f"/perm_{i}/plain_{r['start']}", r["plain"], RED, 1.5, visible))
            if r["path"] is not None:
                hs.append(self.path(f"/perm_{i}/path_{r['start']}", r["path"], AMBER, 3.0, visible))
        return hs


def main(task: str = "tape", output_dir: Path | None = None, data: tuple[Path, ...] = (),
         splats: Path = REPO / "outputs/ur_cube/splats_base.npz", fps: float = 30.0,
         demo_s: float = 3.5, fan_s: float = 3.5, perm_s: float = 3.0, panels: tuple[str, ...] = ("all", "demo", "fan")) -> None:
    """data: one or more traj_perm json files (placements are concatenated, e.g. on and beside the route)."""
    root = REPO / ("outputs/ur_tape" if task == "tape" else "outputs/ur_nut")
    output_dir = output_dir or root / "traj_perm/viser"
    files = data or (root / "traj_perm/traj_perm.json",)
    d = json.loads(Path(files[0]).read_text())
    for f in files[1:]:
        d["permutations"] += json.loads(Path(f).read_text())["permutations"]
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = np.load(splats)
    perms = []
    for p in d["permutations"]:  # drop near-duplicate placements of the same object
        if all(q["object"] != p["object"] or np.hypot(*np.subtract(q["xy"], p["xy"])) > 0.02 for q in perms):
            perms.append(p)
    objs = []
    for p in perms:
        if p["object"] not in objs:
            objs.append(p["object"])
    for panel in (*panels, *[f"perm_{o}" for o in objs]):
        server = viser.ViserServer(port=8159)
        rec = server.get_scene_serializer()
        sc = Scene(server, task, d, sp)
        dm = np.asarray(d["demo"])
        if panel == "demo":
            sc.demo()
            add_object(server, task, "/obj0", dm[0])
        elif panel == "fan":
            sc.demo()
            sc.fan()
        elif panel.startswith("perm_"):  # the placement of this object with the most repairs
            cands = [(i, p) for i, p in enumerate(perms) if p["object"] == panel[5:]]
            i, p = max(cands, key=lambda ip: (ip[1]["n_fixed"], ip[1]["n_ok"]))
            sc.placement(i, p)
        else:
            demo, fan = sc.demo(), sc.fan(visible=False)
            pl = [sc.placement(i, p, visible=False) for i, p in enumerate(perms)]
            mover = add_object(server, task, "/mover", dm[0])
            for k in range(int(demo_s * fps)):
                if k == 0:
                    mover.position = dm[0, 4:] + 1e-4
                t = min(int(k / (demo_s * fps * 0.8) * (len(dm) - 1)), len(dm) - 1)
                mover.wxyz, mover.position = dm[t, :4], dm[t, 4:]
                rec.insert_sleep(1 / fps)
            mover.visible = False
            set_visible(fan, True)
            rec.insert_sleep(fan_s)
            set_visible(fan, False)
            set_visible(demo, False)
            for i, hs in enumerate(pl):
                set_visible(hs, True)
                runs = [r for r in perms[i]["runs"] if r["path"] is not None]
                if runs:
                    path = np.asarray(runs[0]["path"])
                    mover.visible = True
                    for k in range(int(perm_s * fps)):
                        t = min(int(k / (perm_s * fps * 0.8) * (len(path) - 1)), len(path) - 1)
                        mover.wxyz, mover.position = path[t, :4], path[t, 4:]
                        rec.insert_sleep(1 / fps)
                    mover.visible = False
                else:
                    rec.insert_sleep(perm_s)
                set_visible(hs, False)
            set_visible(demo, True)
        if panel != "all":
            rec.insert_sleep(1.0)
        out = output_dir / f"traj_{panel}.viser"
        out.write_bytes(rec.serialize())
        server.stop()
        print(f"{panel}: {out} ({out.stat().st_size / 1e6:.1f} MB)")
    counts = dict(placements=len(perms), trajectories=sum(len(p["runs"]) for p in perms),
                  plain_collide=sum(not r["plain_ok"] for p in perms for r in p["runs"]),
                  kept_filter_only=sum(r["plain_ok"] for p in perms for r in p["runs"]),
                  kept_with_repair=sum(r["path"] is not None for p in perms for r in p["runs"]),
                  by_object={o: dict(placements=sum(p["object"] == o for p in perms),
                                     collide=sum(not r["plain_ok"] for p in perms if p["object"] == o for r in p["runs"]),
                                     kept=sum(r["path"] is not None for p in perms if p["object"] == o for r in p["runs"]),
                                     shown=max(((p["n_hit"], p["n_fixed"], p["n_ok"], len(p["runs"])) for p in perms if p["object"] == o),
                                               key=lambda t: (t[1], t[2])))
                             for o in objs})
    (output_dir / "counts.json").write_text(json.dumps(counts, indent=1))
    print(json.dumps(counts))


if __name__ == "__main__":
    tyro.cli(main)
