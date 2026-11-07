"""Interactive viser recordings of the cross-embodiment retargets (retarget_embodiment.py): the
phone-scanned bench, the cube's Gaussian splat following the tracked trajectory, and the arm.

Run with viser 0.2.23 (the project page's client version):
    uv run --no-project --with viser==0.2.23 --with yourdfpy --with pycollada --with tyro \
        python scripts/ur_cube/make_embodiment_viser.py --output-dir outputs/ur_cube/embodiments/viser
Preview:  python3 -m http.server 8091 -d outputs/ur_cube/embodiments/viser
"""

import json
import shutil
from pathlib import Path

import numpy as np
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

REPO = Path(__file__).resolve().parents[2]
URDFS = {"ur5e": "data/ur5e_robotiq/ur5e_robotiq85.urdf", "ur10e": "data/ur10e_robotiq/ur10e_robotiq85.urdf",
         "panda": "data/panda/panda.urdf"}
LABELS = {"ur5e": "UR5e + Robotiq 2F-85", "ur10e": "UR10e + Robotiq 2F-85", "panda": "Franka Panda + Franka Hand"}
CAMERA = dict(pos="0.62,-1.32,0.5", look="0.05,-0.48,0.14")


def add_bench(server, sp, table_z, radius):
    c = sp["bench_centers"]
    b = (c[:, 2] > table_z - 0.03) & (np.linalg.norm(c[:, :2] - [0.12, -0.55], axis=1) < radius)
    b &= (c[:, 1] > -1.02) | (c[:, 2] < table_z + 0.02)  # drop the front camera stand, keep the table top
    server.scene.add_gaussian_splats("/bench/gaussian_splat", centers=sp["bench_centers"][b], covariances=sp["bench_covariances"][b],
                                     rgbs=sp["bench_rgbs"][b], opacities=sp["bench_opacities"][b])


def record(name: str, data: Path, out: Path, sp, table_z: float, radius: float, fps: float, hold: int,
           only_step: int | None = None) -> Path:
    d = np.load(data / f"{name}.npz")
    server = viser.ViserServer(port=8156)
    rec = server.get_scene_serializer()
    add_bench(server, sp, table_z, radius)
    base = d["base"]
    if name == "panda":  # the Panda stands on the table top: a small plate under its base
        server.scene.add_box("/plate", color=(70, 74, 80), dimensions=(0.22, 0.22, 0.012),
                             position=(base[4], base[5], table_z + 0.006))
    root = server.scene.add_frame("/robot", show_axes=False, wxyz=base[:4], position=base[4:])
    urdf = REPO / URDFS[name]
    robot = ViserUrdf(server, yourdfpy.URDF.load(
        urdf, filename_handler=lambda fname: yourdfpy.filename_handler_magic(fname, dir=urdf.parent)), root_node_name="/robot")
    del root
    cube = server.scene.add_frame("/cube", show_axes=False, wxyz=d["obj"][0, :4], position=d["obj"][0, 4:])
    server.scene.add_gaussian_splats("/cube/gaussian_splat", centers=sp["cube_centers"], covariances=sp["cube_covariances"],
                                     rgbs=sp["cube_rgbs"], opacities=sp["cube_opacities"])
    names = robot.get_actuated_joint_names()
    cols = {str(n): i for i, n in enumerate(d["joint_names"])}
    fcols = {str(n): i for i, n in enumerate(d["finger_names"])}
    n = len(d["q"])
    frames = list(range(n)) + [n - 1] * hold if only_step is None else [only_step] * 3
    for k, t in enumerate(frames):
        cfg = [d["q"][t, cols[j]] if j in cols else d["fingers"][t, fcols[j]] for j in names]
        robot.update_cfg(np.array(cfg))
        if k == 0:  # force a pose message on frame 0 (viser skips unchanged setters; see make_viser_recording.py)
            cube.wxyz, cube.position = -d["obj"][0, :4], d["obj"][0, 4:] + 1e-4
        cube.wxyz, cube.position = d["obj"][t, :4], d["obj"][t, 4:]
        rec.insert_sleep(1.0 / fps)
    path = out / (f"{name}.viser" if only_step is None else f"{name}_still.viser")
    path.write_bytes(rec.serialize())
    server.stop()
    print(f"{name}: {n} steps -> {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main(output_dir: Path = Path("outputs/ur_cube/embodiments/viser"), data: Path = Path("outputs/ur_cube/embodiments"),
         robots: tuple[str, ...] = ("ur5e", "ur10e", "panda"), splats: Path = REPO / "outputs/ur_cube/splats_base.npz",
         layout: Path = REPO / "outputs/ur_cube/scene_layout.json", bench_radius: float = 1.2, fps: float = 30.0,
         hold: int = 20, still_step: int = 66) -> None:
    """Also writes <robot>_still.viser: one frame (step `still_step`, just after lift-off) for poster images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sp = np.load(splats)
    table_z = json.loads(layout.read_text())["table_z"]
    for r in robots:
        record(r, data, output_dir, sp, table_z, bench_radius, fps, hold)
        record(r, data, output_dir, sp, table_z, bench_radius, fps, hold, only_step=still_step)
    shutil.copytree(Path(viser.__file__).parent / "client/build", output_dir / "viser-client", dirs_exist_ok=True)
    tiles = "\n".join(
        f'<div><iframe src="viser-client/index.html?playbackPath=../{r}.viser&initialCameraPosition={CAMERA["pos"]}'
        f'&initialCameraLookAt={CAMERA["look"]}&initialCameraUp=0,0,1"></iframe><p>{LABELS[r]}</p></div>' for r in robots)
    (output_dir / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Embodiments</title><style>body{font-family:system-ui;margin:16px}"
        ".g{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}iframe{width:100%;aspect-ratio:4/3;border:1px solid #ccc;"
        f"border-radius:6px}}p{{text-align:center}}</style><div class=g>{tiles}</div>")


if __name__ == "__main__":
    tyro.cli(main)
