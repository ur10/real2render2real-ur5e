"""UR5e + Robotiq 2F-85 bench scene, laid out from the RealSense capture.

Everything is in the UR controller base frame (env-local). Table height and fixture positions come
from scripts/ur_cube/measure_scene_from_rgbd.py, camera intrinsics from export_realsense_cameras.py;
the camera poses themselves are loaded by the CameraManager from the matching h5 file.
"""

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import MultiTiledCameraCfg
from isaaclab.utils import configclass

from real2render2real.isaaclab_viser.configs.articulation_configs.ur5e_robotiq_cfg import UR5E_ROBOTIQ_CFG

dir_path = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.join(dir_path, "../../../..")
data_dir = os.path.join(repo_dir, "data")

LAYOUT = json.load(open(os.path.join(repo_dir, "outputs/ur_cube/scene_layout.json")))
INTR = json.load(open(os.path.join(data_dir, "camera_poses/ur_realsense_intrinsics.json")))
UR_CAMERA_POSES_FILE = os.path.join(data_dir, "camera_poses/ur_realsense_front_right.h5")

TABLE_Z = LAYOUT["table_z"]
TABLE_THICKNESS = 0.05
CUBE_SIZE = 0.04
FIXTURE_RADIUS = 0.045
FIXTURE_HEIGHT = sum(f["top_z"] for f in LAYOUT["fixtures"]) / len(LAYOUT["fixtures"]) - TABLE_Z


def _fixture(i: int) -> AssetBaseCfg:
    x, y = LAYOUT["fixtures"][i]["center_xy"]
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Fixture{i}",
        spawn=sim_utils.CylinderCfg(
            radius=FIXTURE_RADIUS,
            height=FIXTURE_HEIGHT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.45, 0.15), roughness=0.6),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, TABLE_Z + FIXTURE_HEIGHT / 2)),
    )


@configclass
class UR5eBaseCfg(InteractiveSceneCfg):
    """UR5e on the bench with the two fixed RealSense views."""

    # Butcher-block bench; covers the robot base and both fixtures. Visual only (kinematic replay).
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.8, 1.6, TABLE_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.62, 0.43), roughness=0.7),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.15, -0.45, TABLE_Z - TABLE_THICKNESS / 2)),
    )
    fixture0 = _fixture(0)
    fixture1 = _fixture(1)

    robot = UR5E_ROBOTIQ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Front + right RealSense, 1280x720. Both share one intrinsic matrix (their average).
    viewport_camera = MultiTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Viewport",
        height=INTR["height"],
        width=INTR["width"],
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=INTR["shared_K"],
            height=INTR["height"],
            width=INTR["width"],
            clipping_range=(0.01, 20),
        ),
        cams_per_env=2,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        # Background/ambient light from a 360 render of the phone scan (make_scan_skybox.py).
        spawn=sim_utils.DomeLightCfg(intensity=900.0, texture_file=f"{data_dir}/ur_cube/skybox_scan.jpg"),
    )
    ceiling_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CeilingLight",
        spawn=sim_utils.CylinderLightCfg(intensity=600.0, radius=1.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, -0.6, 1.8)),
    )


@configclass
class UR5eCubePnPCfg(UR5eBaseCfg):
    """Cube pick-and-place between the two fixtures."""

    # SuGaR mesh of the scanned cube, in the tracked cube frame (scripts/ur_cube/mesh_to_object_frame.py).
    # Kinematic and collision-free: R2R2R replays object poses instead of simulating contact.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.UsdFileCfg(
            # Not instanceable: with the instanceable conversion (data/ur_cube/usd/) the renders of
            # envs >= 1 kept the cube at its spawn pose while physics moved it.
            usd_path=f"{data_dir}/ur_cube/usd_noinst/cube_mesh.usd",
            # Mean colour of the SuGaR texture (vertex colours don't survive the USD import).
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.125, 0.11, 0.082), roughness=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*LAYOUT["fixtures"][1]["center_xy"], TABLE_Z + FIXTURE_HEIGHT + CUBE_SIZE / 2)),
    )


@configclass
class UR5eCubePnPObstacleCfg(UR5eCubePnPCfg):
    """Cube pick-and-place with one inserted obstacle per env: a plain cylinder (below) or, with
    `obstacle_spawn_from_asset`, a scanned object such as data/obstacles/sauce_bottle.

    Visual only (no rigid body, no collider): the simulator moves each env's copy at reset by
    setting its USD transform, and clearance comes from the planner (scripts/ur_cube/demo_repair.py),
    whose collision proxy must match the size set here. A kinematic rigid body was tried first, but
    writing its pose through PhysX left later kinematic writes of envs >= 1 out of the renders.
    """

    obstacle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.26,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.47, 0.66), roughness=0.35),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, -0.76, TABLE_Z + 0.13)),
    )


def obstacle_spawn_from_asset(meta_file: str) -> sim_utils.UsdFileCfg:
    """Spawn config for a prepared obstacle (scripts/ur_cube/prep_gso_obstacle.py writes the json)."""
    meta = json.load(open(meta_file))
    return sim_utils.UsdFileCfg(usd_path=os.path.join(os.path.dirname(os.path.abspath(meta_file)), meta["usd"]))
