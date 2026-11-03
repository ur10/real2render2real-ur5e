"""Put the SuGaR cube mesh into the trajectory's object frame (metres) and make it sim-friendly.

SuGaR writes the mesh in the scan's COLMAP world units. The trajectory (track_to_base_traj.py)
poses the cube frame T_objinit_cube relative to RSRD's object frame (scan axes, centred at the
gaussian centroid, metres). We map COLMAP -> DiG model frame (dataparser transform + scale)
-> object frame -> cube frame, keep the largest connected component (drops SuGaR floaters),
decimate, and write an .obj that IsaacLab's convert_mesh.py can turn into a USD.

Example:
    python scripts/ur_cube/mesh_to_object_frame.py --mesh <sugar .obj> \
        --dataparser-transforms outputs/table_new/dig/2026-11-02_184246/dataparser_transforms.json \
        --traj outputs/ur_cube/traj_ep0_f1_to_f0.json --output data/ur_cube/cube_mesh.obj
"""

import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import tyro


def main(
    mesh: Path,
    dataparser_transforms: Path,
    traj: Path,
    output: Path,
    target_faces: int = 4000,
) -> None:
    m = trimesh.load(mesh, force="mesh", process=False)
    dp = json.loads(dataparser_transforms.read_text())
    D = np.eye(4)
    D[:3, :] = np.array(dp["transform"])
    t = json.loads(traj.read_text())
    s = t["m_per_model_unit"]
    c = np.array(t["objinit_centroid_model"])
    T_obj_cube = np.array(t["T_objinit_cube"])

    v_model = dp["scale"] * (m.vertices @ D[:3, :3].T + D[:3, 3])  # COLMAP -> model units
    v_obj = (v_model - c) * s  # object frame (scan axes), metres
    v_cube = (v_obj - T_obj_cube[:3, 3]) @ T_obj_cube[:3, :3]  # cube frame
    color = None
    if m.visual.kind == "texture":
        color = np.asarray(m.visual.to_color().vertex_colors)[:, :3].mean(0)

    cube = trimesh.Trimesh(v_cube, m.faces, process=True)
    parts = cube.split(only_watertight=False)
    cube = max(parts, key=lambda p: len(p.faces))
    if len(cube.faces) > target_faces:  # open3d: trimesh's decimation needs fast_simplification
        o3m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(cube.vertices), o3d.utility.Vector3iVector(cube.faces))
        o3m = o3m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        cube = trimesh.Trimesh(np.asarray(o3m.vertices), np.asarray(o3m.triangles), process=True)
    cube.fix_normals()
    if color is not None:  # textures don't survive decimation; the cube is uniform, use its mean colour
        cube.visual = trimesh.visual.ColorVisuals(cube, vertex_colors=np.tile([*color, 255], (len(cube.vertices), 1)).astype(np.uint8))
    lo, hi = cube.bounds
    print(f"{len(parts)} components -> kept {len(cube.faces)} faces; extent {np.round(hi - lo, 4)} m, "
          f"centre {np.round((hi + lo) / 2, 4)} m, mean colour {None if color is None else np.round(color).tolist()}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cube.export(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    tyro.cli(main)
