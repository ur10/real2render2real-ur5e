"""UR5e bench scene for the tape-hanging task: the same bench, cameras, robot and lights as the cube
scene (ur5e_scene_cfg.py), without the cube fixtures, plus the measured mug tree and tape roll
(scripts/ur_tape/measure_mug_tree.py, fit_tape_dims.py, build_tape_assets.py).

The roll is two kinematic, collision-free bodies that move together: the blue tape band (key
"cube", so the cube generator's replay code drives it) and its white core ("core").
"""

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import UR5eBaseCfg, data_dir, repo_dir

TAPE_SCENE = json.load(open(os.path.join(repo_dir, "outputs/ur_tape/scene.json")))
TAPE_DIMS = json.load(open(os.path.join(repo_dir, "outputs/ur_tape/tape_dims.json")))
_START = TAPE_DIMS["poses"][0]["centre"]


def _roll_part(name: str, usd: str, color) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/ur_tape/usd/{usd}.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(_START)),
    )


@configclass
class UR5eTapeCfg(UR5eBaseCfg):
    """Hang the tape roll on the mug tree's upper branch."""

    fixture0 = None
    fixture1 = None
    mug_tree = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MugTree",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/ur_tape/usd/mug_tree.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.64, 0.42), roughness=0.6),
        ),
    )
    # blue band and white core of the measured roll (mean image colours, roughly linearised)
    cube = _roll_part("TapeBand", "tape_band", (0.02, 0.10, 0.55))
    core = _roll_part("TapeCore", "tape_core", (0.80, 0.82, 0.82))


# ----------------------------------------------------------------------------- nut on the bolt stand
NUT_SCENE = json.load(open(os.path.join(repo_dir, "outputs/ur_nut/scene.json")))
NUT_DIMS = json.load(open(os.path.join(repo_dir, "outputs/ur_nut/nut_dims.json")))


def _stand_part(name: str, part: str, color) -> AssetBaseCfg:
    b = NUT_SCENE["bolt_stand"]
    p = b[part]
    import math
    if part == "block":
        sx, sy, sz = p["size"]
        yaw = p["yaw"]
        spawn = sim_utils.CuboidCfg(size=(sx, sy, sz), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7))
        rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    else:
        spawn = sim_utils.CylinderCfg(radius=p["radius"], height=p["z1"] - p["z0"],
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.5))
        rot = (1.0, 0.0, 0.0, 0.0)
    return AssetBaseCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}", spawn=spawn,
                        init_state=AssetBaseCfg.InitialStateCfg(pos=(*b["xy"], (p["z0"] + p["z1"]) / 2), rot=rot))


@configclass
class UR5eNutCfg(UR5eBaseCfg):
    """Put the green nut on the bolt stand's threaded rod."""

    fixture0 = None
    fixture1 = None
    stand_block = _stand_part("StandBlock", "block", (0.80, 0.64, 0.42))
    stand_base = _stand_part("StandBase", "base", (0.25, 0.27, 0.30))
    stand_spacers = _stand_part("StandSpacers", "spacers", (0.02, 0.35, 0.10))
    stand_rod = _stand_part("StandRod", "rod", (0.12, 0.13, 0.15))
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Nut",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{data_dir}/ur_nut/usd/nut.usd",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.02, 0.40, 0.12), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(NUT_DIMS["poses"][0]["centre"])),
    )
