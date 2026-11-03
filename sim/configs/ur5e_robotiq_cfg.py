"""UR5e + Robotiq 2F-85 articulation, built by scripts/ur_cube/build_ur5e_robotiq_urdf.py.

The USD root is the UR *controller* base frame (what RTDE poses and the RealSense T_cam2base use),
so the robot is spawned at the env origin with identity rotation.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "../../../../data")

UR5E_ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# 2F-85 linkage: every gripper joint follows finger_joint with these URDF mimic multipliers.
ROBOTIQ_MIMIC = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "right_inner_knuckle_joint": -1.0,
    "right_inner_finger_joint": 1.0,
    "right_outer_knuckle_joint": -1.0,
}
ROBOTIQ_OPEN, ROBOTIQ_CLOSED = 0.0, 0.725  # finger_joint range (rad)

# Arm pose at the start of the human-demo capture (IK of the RTDE-reported flange pose,
# see scripts/ur_cube/check_robot_overlay.py).
UR5E_HOME = [1.5504, -1.7068, 2.0003, -1.8996, -1.4539, 0.0417]

UR5E_ROBOTIQ_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{data_dir}/ur5e_robotiq/usd/ur5e_robotiq85.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={**dict(zip(UR5E_ARM_JOINTS, UR5E_HOME)), **{j: 0.0 for j in ROBOTIQ_MIMIC}},
    ),
    actuators={
        # Stiff, gravity-free PD like FRANKA_PANDA_HIGH_PD_CFG: the pipeline is kinematic replay.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
            effort_limit=330.0, velocity_limit=3.14, stiffness=4e4, damping=80.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=list(ROBOTIQ_MIMIC),
            effort_limit=50.0, velocity_limit=2.0, stiffness=1e3, damping=20.0,
        ),
    },
)
