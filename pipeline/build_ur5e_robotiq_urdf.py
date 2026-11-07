"""Build UR5e + Robotiq 2F-85 URDFs for IsaacLab and for jaxmp IK.

Sources (fetched by robot_descriptions): the official UR ROS 2 description (xacro expanded to
`ur5e_raw.urdf`, see below) and the ROS-Industrial Robotiq 2F-85 URDF.

Outputs in <out_dir>:
  ur5e_robotiq85.urdf     full robot for IsaacLab. Root link `ur_base` is the UR *controller*
                          base frame (what RTDE poses and our T_cam2base use), which is ROS
                          `base_link` rotated 180 deg about z.
  ur5e_robotiq85_ik.urdf  arm only (gripper welded open) with a fixed `tcp` frame, for jaxmp
                          IK. jaxmp assumes mimic multiplier 1, which the Robotiq model breaks.
  meshes/                 copied meshes, referenced relative to the URDFs.

Expand the UR xacro first (needs xacrodoc, so use a throwaway env):
    uv run --no-project --with xacrodoc --with robot_descriptions==3.3.0 python -c \
      "from robot_descriptions import ur5e_description as m; \
       from robot_descriptions._xacro import get_urdf_path; import shutil; \
       shutil.copy(get_urdf_path(m), 'data/ur5e_robotiq/ur5e_raw.urdf')"
    python scripts/ur_cube/build_ur5e_robotiq_urdf.py --out-dir data/ur5e_robotiq
    # same mount, bigger arm (cross-embodiment renders): expand ur10e_description the same way, then
    python scripts/ur_cube/build_ur5e_robotiq_urdf.py --ur-type ur10e --out-dir data/ur10e_robotiq \
        --coupling-z 0.04 --tcp-z 0.16
"""

import copy
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import tyro

ROBOTIQ_DIR = Path.home() / ".cache/robot_descriptions/robotiq_arg85_description"
UR_MESHES = Path.home() / ".cache/robot_descriptions/ur_description/meshes"
GRIPPER_JOINTS = {
    "finger_joint", "left_inner_knuckle_joint", "left_inner_finger_joint",
    "right_inner_knuckle_joint", "right_inner_finger_joint", "right_outer_knuckle_joint",
}


def fixed_joint(name: str, parent: str, child: str, xyz=(0, 0, 0), rpy=(0, 0, 0)) -> ET.Element:
    j = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    return j


def relink_meshes(elem: ET.Element, out_dir: Path, ur_type: str = "ur5e") -> None:
    """Copy each referenced mesh into out_dir/meshes and point the URDF at the copy."""
    for mesh in elem.iter("mesh"):
        fname = mesh.get("filename")
        prefix = f"file://{UR_MESHES / ur_type}"
        if fname.startswith(prefix):
            src = Path(fname[len("file://"):])
            rel = Path(f"meshes/{ur_type}") / src.relative_to(UR_MESHES / ur_type)
        elif fname.startswith("package://robotiq_arg85_description/"):
            src = ROBOTIQ_DIR / fname[len("package://robotiq_arg85_description/"):]
            rel = Path("meshes/robotiq_2f85") / src.name
        else:
            raise ValueError(f"unexpected mesh path {fname}")
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, out_dir / rel)
        mesh.set("filename", str(rel))


def main(
    out_dir: Path = Path("data/ur5e_robotiq"),
    coupling_z: float = 0.0,
    gripper_yaw_deg: float = 0.0,
    tcp_z: float = 0.15,
    ur_type: str = "ur5e",
) -> None:
    """
    Args:
        coupling_z: distance from the UR tool0 flange to the Robotiq base (coupling height), m.
        gripper_yaw_deg: rotation of the gripper about the flange z axis.
        tcp_z: TCP distance from the Robotiq base along its approach (z) axis, m.
        ur_type: UR model whose expanded xacro is <out_dir>/<ur_type>_raw.urdf (ur5e, ur10e, ...).
    """
    ur = ET.parse(out_dir / f"{ur_type}_raw.urdf").getroot()
    rq = ET.parse(ROBOTIQ_DIR / "robots/robotiq_arg85_description.URDF").getroot()

    robot = ET.Element("robot", name=f"{ur_type}_robotiq85")
    robot.append(ET.Element("link", name="ur_base"))
    robot.append(fixed_joint("ur_base-base_link", "ur_base", "base_link", rpy=(0, 0, math.pi)))
    # Drop the xacro's world link and ROS's duplicate `base` frame (== ur_base).
    skip_links, skip_joints = {"world", "base"}, {"base_joint", "base_link-base_fixed_joint"}
    for child in ur:
        if child.tag == "link" and child.get("name") in skip_links:
            continue
        if child.tag == "joint" and child.get("name") in skip_joints:
            continue
        if child.tag in ("link", "joint"):
            robot.append(copy.deepcopy(child))

    robot.append(fixed_joint("tool0-robotiq_85_base_link", "tool0", "robotiq_85_base_link",
                             xyz=(0, 0, coupling_z), rpy=(0, 0, math.radians(gripper_yaw_deg))))
    for child in rq:
        if child.tag not in ("link", "joint"):
            continue
        child = copy.deepcopy(child)
        if child.tag == "joint" and child.find("mimic") is not None:
            # Mimic multipliers of -1 drive these joints negative, outside the URDF's [0, upper]
            # limits, and IsaacLab would clamp them. Make the limits symmetric.
            lim = child.find("limit")
            upper = float(lim.get("upper"))
            lim.set("lower", str(-upper))
        robot.append(child)
    robot.append(ET.Element("link", name="tcp"))
    robot.append(fixed_joint("robotiq_85_base_link-tcp", "robotiq_85_base_link", "tcp", xyz=(0, 0, tcp_z)))

    relink_meshes(robot, out_dir, ur_type)
    ET.indent(robot)
    full_path = out_dir / f"{ur_type}_robotiq85.urdf"
    ET.ElementTree(robot).write(full_path, xml_declaration=True, encoding="utf-8")

    # IK variant: same tree with every gripper joint welded at the open pose.
    ik = copy.deepcopy(robot)
    ik.set("name", f"{ur_type}_robotiq85_ik")
    for j in ik.findall("joint"):
        if j.get("name") in GRIPPER_JOINTS:
            j.set("type", "fixed")
            for tag in ("mimic", "limit", "axis", "dynamics"):
                for e in j.findall(tag):
                    j.remove(e)
    ik_path = out_dir / f"{ur_type}_robotiq85_ik.urdf"
    ET.ElementTree(ik).write(ik_path, xml_declaration=True, encoding="utf-8")

    n_rev = lambda r: len([j for j in r.findall("joint") if j.get("type") == "revolute"])
    print(f"wrote {full_path} ({n_rev(robot)} revolute joints) and {ik_path} ({n_rev(ik)} revolute joints)")


if __name__ == "__main__":
    tyro.cli(main)
