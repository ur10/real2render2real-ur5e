"""Franka Panda + Franka Hand URDFs for the cross-embodiment renders (same layout as
build_ur5e_robotiq_urdf.py): a full model with meshes copied next to it, and an IK model with the
fingers welded open and a fixed `tcp` frame (the hand's TCP, 10.34 cm from the hand flange).

Source: example-robot-data's panda_description, fetched by robot_descriptions:
    uv run --no-project --with robot_descriptions==3.3.0 python -c "from robot_descriptions import panda_description"
    python scripts/ur_cube/build_panda_urdf.py --out-dir data/panda
"""

import copy
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import tyro

SRC = Path.home() / ".cache/robot_descriptions/example-robot-data/robots/panda_description"
PKG = "package://example-robot-data/robots/panda_description/"
FINGERS = {"panda_finger_joint1", "panda_finger_joint2"}


def main(out_dir: Path = Path("data/panda")) -> None:
    robot = ET.parse(SRC / "urdf/panda.urdf").getroot()
    for mesh in robot.iter("mesh"):
        rel = mesh.get("filename")[len(PKG):]
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC / rel, out_dir / rel)
        mesh.set("filename", rel)
    j = ET.SubElement(robot, "joint", name="panda_hand_tcp-tcp", type="fixed")
    ET.SubElement(j, "parent", link="panda_hand_tcp")
    ET.SubElement(j, "child", link="tcp")
    ET.SubElement(robot, "link", name="tcp")
    ET.indent(robot)
    ET.ElementTree(robot).write(out_dir / "panda.urdf", xml_declaration=True, encoding="utf-8")

    ik = copy.deepcopy(robot)
    ik.set("name", "panda_ik")
    for jt in ik.findall("joint"):
        if jt.get("name") in FINGERS:
            jt.set("type", "fixed")
            o = jt.find("origin")  # weld open: 4 cm per finger along its axis
            ax = [float(a) for a in jt.find("axis").get("xyz").split()]
            xyz = [float(a) for a in o.get("xyz").split()]
            o.set("xyz", " ".join(str(x + 0.04 * a) for x, a in zip(xyz, ax)))
            for tag in ("mimic", "limit", "axis", "dynamics"):
                for e in jt.findall(tag):
                    jt.remove(e)
    ET.ElementTree(ik).write(out_dir / "panda_ik.urdf", xml_declaration=True, encoding="utf-8")
    n = lambda r: sum(jt.get("type") in ("revolute", "prismatic") for jt in r.findall("joint"))  # noqa: E731
    print(f"wrote {out_dir}/panda.urdf ({n(robot)} moving joints) and panda_ik.urdf ({n(ik)})")


if __name__ == "__main__":
    tyro.cli(main)
