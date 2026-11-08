"""Meshes for the tape task from the measured shapes (tape_dims.json, scene.json):

  data/ur_tape/tape_band.obj   the blue tape band (r_core..R, width w), roll frame (z = axis)
  data/ur_tape/tape_core.obj   the white core ring (r_in..r_core)
  data/ur_tape/mug_tree.obj    post, base disc and branches, robot base frame
  data/ur_nut/stand_*.obj      block, base nut, spacers and rod, robot base frame (3D viewer; Isaac
                               builds the stand from primitives, ur5e_tape_scene_cfg.py)

Then convert each to USD (not instanceable; the tree is static, the roll parts are kinematic bodies):
    for m in tape_band tape_core; do scripts/ur_cube/isaac_run.sh outputs/ur_tape/logs/usd_$m.log "Generated USD" \
        dependencies/IsaacLab/scripts/tools/convert_mesh.py data/ur_tape/$m.obj data/ur_tape/usd/$m.usd \
        --collision-approximation none --mass 0.05 --headless; done
    scripts/ur_cube/isaac_run.sh outputs/ur_tape/logs/usd_mug_tree.log "Generated USD" \
        dependencies/IsaacLab/scripts/tools/convert_mesh.py data/ur_tape/mug_tree.obj data/ur_tape/usd/mug_tree.usd \
        --collision-approximation none --headless
"""

import json
from pathlib import Path

import numpy as np
import trimesh
import tyro


def rod(a, b, r, sections=24) -> trimesh.Trimesh:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return trimesh.creation.cylinder(radius=r, segment=np.stack([a, b]), sections=sections)


def main(out_dir: Path = Path("data/ur_tape"), dims: Path = Path("outputs/ur_tape/tape_dims.json"),
         scene: Path = Path("outputs/ur_tape/scene.json")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    d = json.loads(dims.read_text())
    band = trimesh.creation.annulus(r_min=d["r_core"], r_max=d["R"], height=d["w"], sections=96)
    core = trimesh.creation.annulus(r_min=d["r_in"], r_max=d["r_core"], height=d["w"], sections=96)
    band.export(out_dir / "tape_band.obj")
    core.export(out_dir / "tape_core.obj")

    s = json.loads(scene.read_text())
    t, tz = s["mug_tree"], s["table_z"]
    parts = [rod([*t["post"]["xy"], t["post"]["z0"]], [*t["post"]["xy"], t["post"]["z1"]], t["post"]["radius"]),
             rod([*t["base"]["xy"], tz], [*t["base"]["xy"], tz + t["base"]["thickness"]], t["base"]["radius"], 64)]
    for b in t["branches"].values():
        parts.append(rod(b["root"], b["tip"], b["radius"], 16))
    tree = trimesh.util.concatenate(parts)
    tree.export(out_dir / "mug_tree.obj")

    nut_scene = Path("outputs/ur_nut/scene.json")
    if nut_scene.exists():
        b = json.loads(nut_scene.read_text())["bolt_stand"]
        xy = b["xy"]
        sx, sy, sz = b["block"]["size"]
        block = trimesh.creation.box(extents=(sx, sy, sz))
        block.apply_transform(trimesh.transformations.rotation_matrix(b["block"]["yaw"], [0, 0, 1]))
        block.apply_translation([*xy, (b["block"]["z0"] + b["block"]["z1"]) / 2])
        parts = [block] + [rod([*xy, b[k]["z0"]], [*xy, b[k]["z1"]], b[k]["radius"], 32) for k in ("base", "spacers", "rod")]
        Path("data/ur_nut").mkdir(parents=True, exist_ok=True)
        for name, part in zip(("block", "base", "spacers", "rod"), parts):
            part.export(Path("data/ur_nut") / f"stand_{name}.obj")
        print("bolt stand parts written to data/ur_nut/stand_*.obj")
    print(f"band {len(band.faces)} faces, core {len(core.faces)}, tree {len(tree.faces)} "
          f"(bounds {tree.bounds.round(3).tolist()})")


if __name__ == "__main__":
    tyro.cli(main)
