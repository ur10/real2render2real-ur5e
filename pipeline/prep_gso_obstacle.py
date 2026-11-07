"""Fetch a Google Scanned Objects model and prepare it as an obstacle for the cube scene.

Downloads the model from Gazebo Fuel (CC-BY 4.0, Google LLC), re-centres the mesh on its vertical
axis with the base at z = 0, downsizes the texture, and writes `obstacle.json` with the collision
proxy the planner uses: a vertical cylinder whose radius is the mesh's largest distance from that
axis and whose height is the mesh height (so the proxy contains the whole object). Boxy objects can
use `--proxy box` instead: the mesh is turned so its tightest footprint rectangle is axis-aligned and
the proxy is that box (`size` = x, y, z extents).

Then convert to USD (not instanceable, no physics: the obstacle is visual-only in Isaac):
    python dependencies/IsaacLab/scripts/tools/convert_mesh.py data/obstacles/sauce_bottle/model.obj \
        data/obstacles/sauce_bottle/usd/model.usd --collision-approximation none --headless

Example:
    python scripts/ur_cube/prep_gso_obstacle.py   # default: the 12 fl oz sauce bottle
"""

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as onp
from PIL import Image

FUEL = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/{name}.zip"


def main(
    name: str = "Weston_No_33_Signature_Sausage_Tonic_12_fl_oz",
    output: Path = Path("data/obstacles/sauce_bottle"),
    texture_size: int = 2048,
    proxy: str = "cylinder",
) -> None:
    src = output / "gso"
    if not (src / "meshes" / "model.obj").exists():
        print("downloading", FUEL.format(name=name))
        with urllib.request.urlopen(FUEL.format(name=name), timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(src)

    lines = (src / "meshes" / "model.obj").read_text().splitlines()
    v = onp.array([l.split()[1:4] for l in lines if l.startswith("v ")], dtype=onp.float64)
    # GSO meshes are z-up, metres, base near z = 0. Axis = centre of the xy bounding box.
    c = onp.array([(v[:, 0].min() + v[:, 0].max()) / 2, (v[:, 1].min() + v[:, 1].max()) / 2, v[:, 2].min()])
    v = v - c
    rot = onp.eye(3)
    if proxy == "box":  # align the tightest footprint rectangle with x / y
        import trimesh.bounds
        T2, _ = trimesh.bounds.oriented_bounds_2D(v[:, :2])
        rot[:2, :2] = T2[:2, :2]
        v = v @ rot.T
        v[:, :2] -= (v[:, :2].min(0) + v[:, :2].max(0)) / 2
    radius = float(onp.linalg.norm(v[:, :2], axis=1).max())
    height = float(v[:, 2].max())

    out, i = [], 0
    for l in lines:
        if l.startswith("vn ") and proxy == "box":
            n = rot @ onp.array(l.split()[1:4], dtype=onp.float64)
            out.append("vn " + " ".join(f"{x:.6f}" for x in n))
        elif l.startswith("v "):
            rest = l.split()[4:]  # keep vertex colours if any
            out.append("v " + " ".join(f"{x:.6f}" for x in v[i]) + ("" if not rest else " " + " ".join(rest)))
            i += 1
        elif l.startswith("mtllib"):
            out.append("mtllib model.mtl")
        else:
            out.append(l)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.obj").write_text("\n".join(out) + "\n")
    mtl = (src / "meshes" / "model.mtl").read_text()
    (output / "model.mtl").write_text(mtl)  # map_Kd texture.png, resolved next to the obj
    im = Image.open(src / "materials" / "textures" / "texture.png").convert("RGB")
    if max(im.size) > texture_size:
        im = im.resize((texture_size, texture_size), Image.LANCZOS)
    im.save(output / "texture.png", optimize=True)

    meta = dict(
        source=f"Google Scanned Objects: {name}",
        url=FUEL.format(name=name),
        license="CC-BY 4.0, Copyright 2020 Google LLC",
        usd="usd/model.usd",
        origin="base",  # mesh origin on the vertical axis at the bottom face
        radius=round(radius, 4),
        height=round(height, 4),
        proxy=proxy,
    )
    if proxy == "box":
        meta["size"] = [round(float(x), 4) for x in (v[:, 0].max() - v[:, 0].min(), v[:, 1].max() - v[:, 1].min(), height)]
    (output / "obstacle.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
