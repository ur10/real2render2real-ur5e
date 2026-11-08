"""Shared pieces for the RGB-only tasks' viser recordings (make_ring_replay_viser.py,
make_ring_traj_viser.py): the scanned bench splat without the cube task's fixtures, and the measured
object and fixture meshes in their measured colours. Needs viser 0.2.23 (see make_viser_recording.py).
"""

import json
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
LAYOUT = json.loads((REPO / "outputs/ur_cube/scene_layout.json").read_text())
TASKS = {
    "tape": dict(object=[("data/ur_tape/tape_band.obj", (18, 62, 196)), ("data/ur_tape/tape_core.obj", (225, 228, 228))],
                 fixture=[("data/ur_tape/mug_tree.obj", (214, 178, 128))], post="outputs/ur_tape/scene.json",
                 camera=dict(pos="0.55,-1.30,0.62", look="0.05,-0.72,0.12"), centre=(0.05, -0.75)),
    "nut": dict(object=[("data/ur_nut/nut.obj", (28, 150, 70))],
                fixture=[("data/ur_nut/stand_block.obj", (214, 178, 128)), ("data/ur_nut/stand_base.obj", (78, 84, 92)),
                         ("data/ur_nut/stand_spacers.obj", (28, 150, 70)), ("data/ur_nut/stand_rod.obj", (52, 56, 62))],
                post="outputs/ur_nut/scene.json", camera=dict(pos="0.45,-1.25,0.50", look="0.02,-0.74,0.08"),
                centre=(0.0, -0.76)),
}


def post_xy(task: str) -> np.ndarray:
    s = json.loads((REPO / TASKS[task]["post"]).read_text())
    return np.asarray(s["mug_tree"]["post"]["xy"] if "mug_tree" in s else s["bolt_stand"]["xy"])


def bench_mask(sp, centre, radius: float):
    """Bench Gaussians over the table near `centre`, minus the cube task's two fixtures and cube."""
    c = sp["bench_centers"]
    tz = LAYOUT["table_z"]
    keep = (c[:, 2] > tz - 0.03) & (np.linalg.norm(c[:, :2] - np.asarray(centre), axis=1) < radius)
    for f in LAYOUT["fixtures"]:  # orange fixtures (+ the cube resting on one in the scan)
        near = np.linalg.norm(c[:, :2] - np.asarray(f["center_xy"]), axis=1) < 0.075
        keep &= ~(near & (c[:, 2] > tz + 0.002))
    return keep


def add_bench(server, sp, centre, radius: float = 1.1, name: str = "/bench/gaussian_splat"):
    """The bench without the cube task's fixtures: Gaussians above the table top near them are
    dropped, and the flat remains (fixture bases, the yellow tape crosses) take the colour of the
    surrounding table, so the table has no holes."""
    b = bench_mask(sp, centre, radius)
    c, rgb = sp["bench_centers"][b], sp["bench_rgbs"][b].copy()
    cov, op = sp["bench_covariances"][b].copy(), sp["bench_opacities"][b].copy()
    tz = LAYOUT["table_z"]
    for f in LAYOUT["fixtures"]:
        d = np.linalg.norm(c[:, :2] - np.asarray(f["center_xy"]), axis=1)
        yellow = (rgb[:, 1] - rgb[:, 2]) > 0.3 * float(rgb.max() > 1.5 and 255 or 1)  # the tape crosses
        dark = rgb.mean(1) < 0.5 * float(rgb.max() > 1.5 and 255 or 1)
        ring = (d > 0.15) & (d < 0.23) & (np.abs(c[:, 2] - tz) < 0.01) & ~yellow & ~dark
        if ring.any():
            inner = (d < 0.135) | ((d < 0.22) & yellow)
            k = np.random.default_rng(0).choice(np.where(ring)[0], int(inner.sum()))
            rgb[inner], cov[inner], op[inner] = rgb[k], cov[k], op[k]  # surrounding table's colours and shapes
    server.scene.add_gaussian_splats(name, centers=c, covariances=cov, rgbs=rgb, opacities=op)


def coloured(path: str, rgb) -> trimesh.Trimesh:
    m = trimesh.load(REPO / path, force="mesh", process=False)
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.tile([*rgb, 255], (len(m.faces), 1)))
    return m


def add_fixture(server, task: str, tree_yaw_deg: float = 0.0, prefix: str = "/fixture"):
    """The measured fixture, turned about its post like the twin's (tree_yaw_deg)."""
    c = np.r_[post_xy(task), 0.0]
    psi = np.radians(tree_yaw_deg)
    q = (np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2))
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    frame = server.scene.add_frame(prefix, show_axes=False, wxyz=q, position=tuple(c - Rz @ c))
    for i, (path, rgb) in enumerate(TASKS[task]["fixture"]):
        server.scene.add_mesh_trimesh(f"{prefix}/part{i}", coloured(path, rgb))
    return frame


def add_object(server, task: str, name: str, pose, visible: bool = True):
    """The measured object as a frame (wxyz_xyz pose) with its coloured mesh parts."""
    pose = np.asarray(pose)
    f = server.scene.add_frame(name, show_axes=False, wxyz=pose[:4], position=pose[4:], visible=visible)
    for i, (path, rgb) in enumerate(TASKS[task]["object"]):
        server.scene.add_mesh_trimesh(f"{name}/part{i}", coloured(path, rgb))
    return f
