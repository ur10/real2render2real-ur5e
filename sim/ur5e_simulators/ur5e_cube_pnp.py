"""UR5e + Robotiq 2F-85 cube pick-and-place data generation (R2R2R-style kinematic replay).

Differs from the Franka/YuMi simulators in what it consumes: a metric object trajectory already in
the robot base frame (see scripts/ur_cube/), plus a grasp as a TCP pose in the object frame, instead
of RSRD's scan-unit part deltas. Per reset it:
  * samples a new cube start on the pick fixture and warps the leading 60% of the demo trajectory to
    it (trajgen), keeping the place pose fixed,
  * solves IK (jaxmp) for approach -> grasp -> carry -> release -> retreat and replays it,
  * renders the two RealSense views, logs joints / flange pose / gripper command,
  * keeps an env only if it passes a feasibility filter (IK error, joint limits, finger clearance
    over table and fixtures, cube not sinking into the table).
"""

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as onp
import torch

import real2render2real.utils.transforms as tf
from real2render2real.isaaclab_viser.base import IsaacLabViser
from real2render2real.isaaclab_viser.configs.articulation_configs.ur5e_robotiq_cfg import (
    ROBOTIQ_CLOSED, ROBOTIQ_MIMIC, ROBOTIQ_OPEN, UR5E_ARM_JOINTS, UR5E_HOME,
)
from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import (
    CUBE_SIZE, FIXTURE_HEIGHT, FIXTURE_RADIUS, LAYOUT, TABLE_Z, UR_CAMERA_POSES_FILE,
)
from trajgen.traj_interp import traj_interp_batch

FINGERTIP_BELOW_TCP = 0.03  # fin-ray fingertip sits ~3 cm past the mid-finger TCP


@dataclass
class PnPPhases:
    """Step counts of each phase; carry length comes from the trajectory."""
    setup: int = 6
    approach: int = 30
    close: int = 8
    release: int = 8
    retreat: int = 14
    approach_height: float = 0.10  # pre-grasp TCP height above the grasp
    carry: int = 0
    bounds: dict = field(default_factory=dict)

    def finalize(self, carry: int):
        self.carry = carry
        edges = onp.cumsum([0, self.setup, self.approach, self.close, carry, self.release, self.retreat])
        names = ["setup", "approach", "close", "carry", "release", "retreat"]
        self.bounds = {n: (int(a), int(b)) for n, a, b in zip(names, edges[:-1], edges[1:])}
        self.total = int(edges[-1])

    def phase(self, count: int):
        for name, (a, b) in self.bounds.items():
            if a <= count < b:
                return name, (count - a) / max(b - a - 1, 1)
        return "done", 1.0


def smoothstep(t):
    return t * t * (3 - 2 * t)


def along_tool_z(dz: float, n: int, device) -> tf.SE3:
    """Pure translation of dz along the (right-multiplied) frame's z axis, batched."""
    return tf.SE3(torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, dz]], device=device).repeat(n, 1))


class UR5eCubePnP(IsaacLabViser):
    def __init__(self, *args, traj_path: Path, max_successes: int = 1000, max_episodes: int = None,
                 start_xy_noise: float = 0.03, start_yaw_noise: float = onp.radians(30),
                 wait_for_client: bool = False, randomize_look: bool = True, **kwargs):
        self.randomize_look = randomize_look
        self.camera_poses_file = UR_CAMERA_POSES_FILE
        self.wait_for_client = wait_for_client
        self.max_successes, self.max_episodes, self.episodes = max_successes, max_episodes, 0
        self.start_xy_noise, self.start_yaw_noise = start_xy_noise, start_yaw_noise
        super().__init__(*args, **kwargs)

        data = json.loads(Path(traj_path).read_text())
        self.demo_traj = onp.asarray(data["T_base_obj"], dtype=onp.float64)  # (T, wxyz_xyz), metres
        self.grasp_obj_tcp = torch.tensor(data["grasp_T_obj_tcp"], dtype=torch.float32, device=self.device)
        self.gripper_close = float(data.get("gripper_close", 0.45))
        self.phases = PnPPhases()
        self.phases.finalize(len(self.demo_traj))
        self.run_simulator()

    @property
    def device(self):
        return self.scene.env_origins.device

    # ------------------------------------------------------------------ main loop
    def run_simulator(self):
        from real2render2real.isaaclab_viser.controllers.jaxmp_diff_ik_controller import JaxMPBatchedController

        self.robot = self.scene.articulations["robot"]
        self.cube = self.scene.rigid_objects["cube"]
        # Look indices up by name: SceneEntityCfg.resolve returns them in articulation order,
        # not in the order requested.
        self.arm_ids = [self.robot.joint_names.index(j) for j in UR5E_ARM_JOINTS]
        self.tcp_body = self.robot.body_names.index("tcp")
        self.flange_body = self.robot.body_names.index("tool0")
        self.gripper_ids = [self.robot.joint_names.index(j) for j in ROBOTIQ_MIMIC]
        self.gripper_mult = torch.tensor(list(ROBOTIQ_MIMIC.values()), device=self.device)

        ik_urdf = Path(self.urdf_path["robot"]).with_name("ur5e_robotiq85_ik.urdf")
        self.controller = JaxMPBatchedController(
            urdf_path=ik_urdf, num_envs=self.scene.num_envs, num_ees=1,
            target_names=["robotiq_85_base_link-tcp"], home_pose=onp.array(UR5E_HOME),
        )

        count, self.success_envs = 0, None
        dt = self.sim.get_physics_dt()
        while self.simulation_app.is_running() and self.data_logger_successes() < self.max_successes:
            t0 = time.time()
            if count % self.phases.total == 0:
                if self.max_episodes is not None and self.episodes >= self.max_episodes:
                    self._handle_reset()  # flush the last episode's success/failure
                    break
                self._handle_reset()
                self.episodes += 1
                count = 0
            name, s = self.phases.phase(count)
            self._step_world(count, name, s, dt)
            self._render()
            if name != "setup":
                self._check_feasibility(name)
                self._log_data(count - self.phases.bounds["approach"][0])
            self.sim_step_time_ms.value = (time.time() - t0) * 1e3
            count += 1

    def data_logger_successes(self) -> int:
        # Read the counter directly: get_stats() fails before the first save_data() call.
        return 0 if self.data_logger is None else int(self.data_logger._total_successful_envs)

    # ------------------------------------------------------------------ reset / augmentation
    def _handle_reset(self):
        if self.success_envs is not None:
            print(f"[INFO]: Episode {self.episodes}: success envs {self.success_envs.cpu().numpy().astype(int).tolist()}, "
                  f"failures by reason {self.fail_reasons}", flush=True)
            if self.data_logger is not None:
                # Let queued frames land before env dirs are moved, or the last one can be lost.
                while not self.data_logger.task_queue.empty():
                    time.sleep(0.02)
                time.sleep(0.2)
                if getattr(self, "obj_log", None):
                    poses = onp.stack(self.obj_log, axis=1)  # (N, T, 7) base-frame wxyz_xyz
                    for e in range(self.scene.num_envs):
                        d = Path(self.output_dir) / f"env_{e}" / "robot_data"
                        d.mkdir(parents=True, exist_ok=True)
                        onp.save(d / "object_poses.npy", poses[e])
                self.data_logger.redir_data(self.success_envs)
        self.obj_log = []
        n = self.scene.num_envs

        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)
        self.controller.reset()

        # New cube starts: jitter xy on the pick fixture and yaw about z; keep the demo's height.
        start = tf.SE3(torch.tensor(self.demo_traj[0], dtype=torch.float32))
        yaw = (torch.rand(n) * 2 - 1) * self.start_yaw_noise
        dxy = (torch.rand(n, 2) * 2 - 1) * self.start_xy_noise
        new_starts = tf.SE3.from_rotation(tf.SO3.from_z_radians(yaw)) @ tf.SE3(start.wxyz_xyz.repeat(n, 1))
        ns = new_starts.wxyz_xyz.clone()
        ns[:, 4:] = start.wxyz_xyz[4:] + torch.cat([dxy, torch.zeros(n, 1)], dim=1)
        trajs = traj_interp_batch(traj=self.demo_traj, new_starts=ns.numpy(), proportion=0.6)
        self.trajs = torch.tensor(onp.asarray(trajs), dtype=torch.float32, device=self.device)  # (N, T, 7)

        # Cube is 90-degree symmetric about z: pick the grasp yaw closest to the home wrist.
        self.grasp = self._choose_grasp_yaw(self.trajs[:, 0])
        self.start_tcp = self._tcp_pose_b()
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_appearance()
        self.success_envs = torch.ones(n, dtype=torch.bool, device=self.device)
        self.fail_reasons = {}

    def _choose_grasp_yaw(self, obj0: torch.Tensor) -> tf.SE3:
        n = obj0.shape[0]
        home = self._tcp_pose_b()
        best, best_err = None, None
        for k in range(4):
            g = tf.SE3.from_rotation(tf.SO3.from_z_radians(torch.tensor([k * onp.pi / 2], device=self.device))) @ tf.SE3(self.grasp_obj_tcp[None])
            tcp = tf.SE3(obj0) @ tf.SE3(g.wxyz_xyz.repeat(n, 1))
            err = (tcp.rotation().inverse() @ home.rotation()).log().norm(dim=-1)
            if best is None:
                best, best_err = g.wxyz_xyz.repeat(n, 1), err
            else:
                better = err < best_err
                best[better], best_err = g.wxyz_xyz.repeat(n, 1)[better], torch.minimum(err, best_err)
        return tf.SE3(best)

    # ------------------------------------------------------------------ per-step world update
    def _obj_pose(self, name: str, s: float) -> torch.Tensor:
        T = self.trajs.shape[1]
        if name in ("setup", "approach", "close"):
            idx = 0
        elif name == "carry":
            idx = int(round(s * (T - 1)))
        else:
            idx = T - 1
        return self.trajs[:, idx]

    def _tcp_target(self, name: str, s: float, obj: torch.Tensor) -> tf.SE3:
        grasp = tf.SE3(obj) @ self.grasp
        pre = grasp @ along_tool_z(-self.phases.approach_height, obj.shape[0], self.device)  # back off along tool -z
        if name == "setup":
            return self.start_tcp
        if name == "approach":
            if s < 0.65:  # move from the start pose to the pre-grasp
                return self._interp(self.start_tcp, pre, smoothstep(s / 0.65))
            return self._interp(pre, grasp, smoothstep((s - 0.65) / 0.35))
        if name in ("close", "carry", "release"):
            return grasp
        return self._interp(grasp, pre, smoothstep(s))  # retreat

    @staticmethod
    def _interp(a: tf.SE3, b: tf.SE3, t: float) -> tf.SE3:
        delta = (a.inverse() @ b).log()
        return a @ tf.SE3.exp(delta * t)

    def _gripper_q(self, name: str, s: float) -> float:
        if name in ("setup", "approach"):
            return ROBOTIQ_OPEN
        if name == "close":
            return ROBOTIQ_OPEN + smoothstep(s) * (self.gripper_close - ROBOTIQ_OPEN)
        if name == "carry":
            return self.gripper_close
        if name == "release":
            return self.gripper_close + smoothstep(s) * (ROBOTIQ_OPEN - self.gripper_close)
        return ROBOTIQ_OPEN

    def _step_world(self, count: int, name: str, s: float, dt: float):
        obj = self._obj_pose(name, s)
        self._obj_now = obj
        state = torch.zeros_like(self.cube.data.root_state_w)
        state[:, :3] = obj[:, 4:] + self.scene.env_origins
        state[:, 3:7] = obj[:, :4]
        self.cube.write_root_state_to_sim(state)

        self.tcp_target = self._tcp_target(name, s, obj)
        target = torch.cat([self.tcp_target.wxyz_xyz[:, 4:], self.tcp_target.wxyz_xyz[:, :4]], dim=-1)
        joints = torch.tensor(onp.array(self.controller.compute_ik(target[:, None].cpu().numpy())),
                              dtype=torch.float32, device=self.device)

        q = self.robot.data.joint_pos.clone()
        q[:, self.arm_ids] = joints[:, :6]
        self.gripper_cmd = self._gripper_q(name, s)
        q[:, self.gripper_ids] = self.gripper_cmd * self.gripper_mult
        # Kinematic replay: teleport the joints and hold them with the PD targets.
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt)
        self.phase_name = name

    # ------------------------------------------------------------------ feasibility filter
    def _fail(self, mask: torch.Tensor, reason: str):
        if mask.any():
            self.fail_reasons[reason] = self.fail_reasons.get(reason, 0) + int((mask & self.success_envs).sum())
            self.success_envs &= ~mask

    def _check_feasibility(self, name: str):
        tcp = self._tcp_pose_b()
        target = self.tcp_target
        pos_err = (tcp.translation() - target.translation()).norm(dim=-1)
        rot_err = (tcp.rotation().inverse() @ target.rotation()).log().norm(dim=-1)
        bad = (pos_err > 0.02) | (rot_err > 0.45)  # same thresholds as the R2R2R Franka simulator
        if bad.any() and "ik_error" not in self.fail_reasons:  # first IK failure this episode
            e = int(torch.where(bad)[0][0])
            print(f"[DEBUG] ik_error env{e} phase={name}: pos_err {pos_err[e] * 1000:.1f} mm, rot_err {rot_err[e]:.3f} rad, "
                  f"tcp {tcp.translation()[e].cpu().numpy().round(3)} target {target.translation()[e].cpu().numpy().round(3)}", flush=True)
        self._fail(bad, "ik_error")

        q = self.robot.data.joint_pos[:, self.arm_ids]
        lim = self.robot.data.joint_pos_limits[:, self.arm_ids]
        self._fail(((q < lim[..., 0]) | (q > lim[..., 1])).any(dim=-1), "joint_limit")

        # Fingertips must stay above the table and above a fixture top when over one.
        tip = (tcp @ along_tool_z(FINGERTIP_BELOW_TCP, self.scene.num_envs, self.device)).translation()
        floor = torch.full_like(tip[:, 2], TABLE_Z)
        for f in LAYOUT["fixtures"]:
            over = (tip[:, :2] - torch.tensor(f["center_xy"], device=self.device)).norm(dim=-1) < FIXTURE_RADIUS
            floor = torch.where(over, torch.full_like(floor, TABLE_Z + FIXTURE_HEIGHT), floor)
        # While the cube is held its own faces are between the fingers; only check away from it.
        if name in ("approach", "retreat"):
            self._fail(tip[:, 2] < floor + 0.002, "finger_clearance")

        cube_bottom = self.cube.data.root_state_w[:, 2] - CUBE_SIZE / 2
        self._fail(cube_bottom < TABLE_Z - 0.002, "cube_below_table")

    def _tcp_pose_b(self) -> tf.SE3:
        p = self.robot.data.body_state_w[:, self.tcp_body, :7]
        return tf.SE3(torch.cat([p[:, 3:7], p[:, :3] - self.scene.env_origins], dim=-1))

    # ------------------------------------------------------------------ rendering + logging
    def randomize_lighting(self):
        """Base randomization at ~1/4 intensity: the base values wash out this scene and render
        the black cube brown, far brighter than the RealSense frames."""
        super().randomize_lighting()
        for light in self.lights:
            attr = light.GetAttribute("inputs:intensity")
            attr.Set(float(attr.Get()) * 0.25)

    def randomize_appearance(self):
        """Per-episode table tint and a small skybox yaw (the paper's appearance randomization)."""
        if not self.randomize_look:
            return
        from pxr import Gf, UsdShade
        for prim in self.sim.stage.Traverse():
            path = str(prim.GetPath())
            if "/Table" in path and prim.IsA(UsdShade.Shader):
                inp = UsdShade.Shader(prim).GetInput("diffuseColor")
                if inp:
                    base = onp.array([0.78, 0.62, 0.43])
                    tint = onp.clip(base * onp.random.uniform(0.8, 1.15) + onp.random.normal(0, 0.03, 3), 0, 1)
                    inp.Set(Gf.Vec3f(*tint.tolist()))
        yaw = onp.radians(onp.random.uniform(-25, 25))
        dome = self.scene._extras["dome_light"]._prims[0]
        dome.GetAttribute("xformOp:orient").Set(Gf.Quatd(float(onp.cos(yaw / 2)), Gf.Vec3d(0, 0, float(onp.sin(yaw / 2)))))

    def _setup_viser_gui(self):
        """Headless replacement for the base GUI.

        The base GUI uses viser 1.x widgets (e.g. gui.add_image) but nerfstudio pins viser 0.2.7 in
        this env; generation only needs the per-camera buffers and a few stats.
        """
        for stat in ("render_time_ms", "sim_step_time_ms", "save_time_ms", "images_per_second", "successful_envs"):
            setattr(self, stat, SimpleNamespace(value=0))
        self.env = 0
        self.camera_manager.buffers[self.camera_manager.render_cam] = deque(maxlen=1)
        for f in self.camera_manager.frustums:
            self.camera_manager.buffers[f.name[1:]] = deque(maxlen=1)

    def _handle_client_connection(self):
        if self.wait_for_client:
            super()._handle_client_connection()

    def _render(self):
        t0 = time.time()
        cam = self.isaac_viewport_camera
        frustums = self.camera_manager.frustums
        n = self.scene.num_envs
        xyz = torch.tensor(onp.array([f.position for f in frustums]), dtype=torch.float32).repeat(n, 1)
        wxyz = torch.tensor(onp.array([f.wxyz for f in frustums]), dtype=torch.float32).repeat(n, 1)
        xyz += self.scene.env_origins.cpu().repeat_interleave(len(frustums), dim=0)
        cam.set_world_poses(xyz, wxyz, convention="ros")
        self.sim.render()
        cam.update(0, force_recompute=True)
        rgb = cam.data.output["rgb"]
        for i, f in enumerate(frustums):
            self.camera_manager.buffers[f.name[1:]].append({"rgb": rgb[i::len(frustums)].clone()})
        self.render_time_ms.value = (time.time() - t0) * 1e3

    def _log_data(self, step: int):
        if self.data_logger is None:
            return
        flange = self.robot.data.body_state_w[:, self.flange_body, :7]
        robot_data = {
            "joint_names": [*UR5E_ARM_JOINTS, "finger_joint"],
            "joint_angles": torch.cat([
                self.robot.data.joint_pos[:, self.arm_ids],
                self.robot.data.joint_pos[:, self.gripper_ids[:1]],
            ], dim=1).cpu().numpy(),
            # Flange pose in the base frame, xyz + wxyz: same frame as the real RTDE eef pose.
            "ee_pos": torch.cat([flange[:, :3] - self.scene.env_origins, flange[:, 3:7]], dim=1).cpu().numpy(),
            "gripper_binary_cmd": onp.full((self.scene.num_envs, 1), self.gripper_cmd > 0.5 * (ROBOTIQ_OPEN + self.gripper_close)),
        }
        buffers = {f.name[1:]: self.camera_manager.buffers[f.name[1:]] for f in self.camera_manager.frustums}
        self.data_logger.save_data(buffers, robot_data, step, self.output_dir)
        self.obj_log.append(self._obj_now.cpu().numpy())  # cube pose (wxyz_xyz) at this logged step
        stats = self.data_logger.get_stats()
        self.save_time_ms.value = int(stats["save_time"] * 1e3)
        self.images_per_second.value = stats["images_per_second"]
        self.successful_envs.value = stats["total_successful_envs"]
