"""Trajectory-interpolation scene in viser (the counterpart of real2render2real.com's "Trajectory
Interpolation"), extended with collision-object permutations. Data: make_traj_permutations.py.

  demo   the tracked cube trajectory with 6-DoF frames along it
  fan    the generator's interpolated variants: new starts (6-DoF frames) and their paths
  perm_<object>  a scanned object placed on / beside the route: the plain variants that collide
         (red) and the variants the generator keeps after repair (amber)
  all    one looping recording: demo -> fan -> each placement in turn (for the project page)

Run with viser 0.2.23:
    uv run --no-project --with viser==0.2.23 --with trimesh --with pillow --with tyro \
        python scripts/ur_cube/make_traj_viser.py --output-dir outputs/ur_cube/traj_perm/viser
"""

import json
from pathlib import Path

import numpy as np
import trimesh
import tyro
import viser
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
TEAL, AMBER, RED = (95, 181, 192), (230, 160, 70), (226, 104, 90)
CAMERA = dict(pos="0.42,-1.22,0.42", look="0.09,-0.77,0.04")


def load_object(name: str, tex: int = 512) -> trimesh.Trimesh:
    d = REPO / "data/obstacles" / name
    m = trimesh.load(d / "model.obj", force="mesh", process=False)
    im = Image.open(d / "texture.png").convert("RGB").resize((tex, tex), Image.LANCZOS)
    m.visual = trimesh.visual.TextureVisuals(uv=m.visual.uv, material=trimesh.visual.material.SimpleMaterial(image=im))
    return m


def yaw_wxyz(yaw: float):
    return (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))


class Scene:
    def __init__(self, server, data: dict, sp, bench_radius: float):
        self.s, self.d, self.sp = server, data, sp
        tz = data["table_z"]
        b = (sp["bench_centers"][:, 2] > tz - 0.03) & (np.linalg.norm(sp["bench_centers"][:, :2] - [0.09, -0.77], axis=1) < bench_radius)
        server.scene.add_gaussian_splats("/bench", centers=sp["bench_centers"][b], covariances=sp["bench_covariances"][b],
                                         rgbs=sp["bench_rgbs"][b], opacities=sp["bench_opacities"][b])

    def cube(self, name: str, pose, visible=True):
        f = self.s.scene.add_frame(name, show_axes=False, wxyz=pose[:4], position=pose[4:], visible=visible)
        sp = self.sp
        self.s.scene.add_gaussian_splats(f"{name}/splat", centers=sp["cube_centers"], covariances=sp["cube_covariances"],
                                         rgbs=sp["cube_rgbs"], opacities=sp["cube_opacities"])
        return f

    def path(self, name, traj, color, width=3.0, visible=True):
        return self.s.scene.add_spline_catmull_rom(name, positions=np.asarray(traj)[:, 4:], color=color, line_width=width,
                                                   visible=visible)

    def axes(self, name, poses, length=0.025, visible=True):
        p = np.asarray(poses)
        return self.s.scene.add_batched_axes(name, batched_wxyzs=p[:, :4], batched_positions=p[:, 4:], axes_length=length,
                                             axes_radius=length * 0.07, visible=visible)

    def demo(self, visible=True):
        dm = np.asarray(self.d["demo"])
        return [self.path("/demo/path", dm, TEAL, 4.0, visible), self.axes("/demo/axes", dm[::6], visible=visible)]

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


def set_visible(hs, v: bool):
    for h in hs:
        h.visible = v


def main(output_dir: Path = Path("outputs/ur_cube/traj_perm/viser"), data: Path = Path("outputs/ur_cube/traj_perm/traj_perm.json"),
         splats: Path = REPO / "outputs/ur_cube/splats_base.npz", bench_radius: float = 0.9, fps: float = 30.0,
         demo_s: float = 3.5, fan_s: float = 3.5, perm_s: float = 3.0,
         panels: tuple[str, ...] = ("all", "demo", "fan", "perm_sauce_bottle", "perm_mug", "perm_cereal_box")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(data.read_text())
    sp = np.load(splats)
    perms = []
    for p in d["permutations"]:  # drop near-duplicate placements of the same object
        if all(q["object"] != p["object"] or np.hypot(*np.subtract(q["xy"], p["xy"])) > 0.02 for q in perms):
            perms.append(p)
    for panel in panels:
        server = viser.ViserServer(port=8157)
        rec = server.get_scene_serializer()
        sc = Scene(server, d, sp, bench_radius)
        if panel == "demo":
            sc.demo()
            sc.cube("/cube0", np.asarray(d["demo"])[0])
        elif panel == "fan":
            sc.demo()
            sc.fan()
        elif panel.startswith("perm_"):  # the first (most repaired) placement of one object
            i = next(i for i, p in enumerate(perms) if p["object"] == panel[5:])
            sc.placement(i, perms[i])
        else:
            demo, fan = sc.demo(), sc.fan(visible=False)
            pl = [sc.placement(i, p, visible=False) for i, p in enumerate(perms)]
            mover = sc.cube("/mover", np.asarray(d["demo"])[0])
            dm = np.asarray(d["demo"])
            for k in range(int(demo_s * fps)):  # demo: the cube follows the track
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
                if runs:  # a cube rides one repaired path
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
        path = output_dir / f"traj_{panel}.viser"
        path.write_bytes(rec.serialize())
        server.stop()
        print(f"{panel}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    import shutil
    shutil.copytree(Path(viser.__file__).parent / "client/build", output_dir / "viser-client", dirs_exist_ok=True)


if __name__ == "__main__":
    tyro.cli(main)
