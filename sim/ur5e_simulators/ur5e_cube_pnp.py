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

With `obstacle=...` each env also gets an obstacle (a scanned bottle or a cylinder) on the transfer route, and the
motion comes from scripts/ur_cube/demo_repair.py instead of per-step IK: the generated motion is
checked against the obstacle and, with `repair=True`, its free segments (transit and the carry
between lift-off and set-down) are re-optimized while the grasp, lift-off, set-down and retreat stay
as generated. With `repair=False` colliding envs are rejected (the plain generator's behaviour).
Per-env outcomes go to <output_dir>/plans.jsonl.
"""

import json
import sys
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
                 wait_for_client: bool = False, randomize_look: bool = True,
                 obstacle: dict | None = None, repair: bool = True, **kwargs):
        self.randomize_look = randomize_look
        # obstacle: dict(radius, height, origin, fraction=(lo, hi), lateral=(lo, hi)), see
        # _sample_obstacles; radius/height are the planner's collision cylinder. Figure options:
        # xy=(x, y) fixes the placement; compare=True makes env 0 replay the plain generator's motion
        # (through the obstacle, kept unfiltered) next to the repaired motion in the other envs.
        self.obstacle_cfg, self.repair = obstacle, repair
        self.camera_poses_file = UR_CAMERA_POSES_FILE
        self.wait_for_client = wait_for_client
        self.max_successes, self.max_episodes, self.episodes = max_successes, max_episodes, 0
        self.start_xy_noise, self.start_yaw_noise = start_xy_noise, start_yaw_noise
        super().__init__(*args, **kwargs)

        data = json.loads(Path(traj_path).read_text())
        self.demo_traj = onp.asarray(data["T_base_obj"], dtype=onp.float64)  # (T, wxyz_xyz), metres
        self.grasp_obj_tcp = torch.tensor(data["grasp_T_obj_tcp"], dtype=torch.float32, device=self.device)
        self.gripper_close = float(data.get("gripper_close", 0.45))
        self.planner, self.traj_path = None, Path(traj_path)
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

        if self.obstacle_cfg is not None:
            # Built here, next to the IK controller, so JAX initialises at the same point as before.
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ur_cube"))
            from demo_repair import DemoRepair, Phases
            ph = PnPPhases()
            self.planner = DemoRepair(demo_file=self.traj_path, phases=Phases(
                approach=ph.approach, close=ph.close, release=ph.release, retreat=ph.retreat,
                approach_height=ph.approach_height))
            # Warp the cleaned-up track (levelled, minimal yaw; see demo_repair) instead of the raw one.
            self.demo_traj = self.planner.demo_obj

        count, self.success_envs = 0, None
        dt = self.sim.get_physics_dt()
        while self.simulation_app.is_running() and self.data_logger_successes() < self.max_successes:
            t0 = time.time()
            if count % self.phases.total == 0:
                if self.max_episodes is not None and self.episodes >= self.max_episodes:
                    self._handle_reset(final=True)  # flush the last episode's success/failure
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
    def _handle_reset(self, final: bool = False):
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
        # final=True: flushing before exit, no episode follows (skip planning one)
        self.plan = self._plan_with_obstacles() if self.planner is not None and not final else None
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_appearance()
        self.success_envs = torch.ones(n, dtype=torch.bool, device=self.device)
        self.fail_reasons = {}
        if self.plan is not None:
            for e, st in enumerate(self.plan["status"]):
                if not bool(self.plan["ok"][e]):
                    self._fail(torch.arange(n, device=self.device) == e, st)

    # ------------------------------------------------------------------ obstacle + planned motion
    def _sample_obstacles(self) -> onp.ndarray:
        """Per env (x, y): uniform along the env's own cube route and across it, away from fixtures."""
        c = self.obstacle_cfg
        if c.get("xy") is not None:  # fixed placement (figures)
            return onp.tile(onp.asarray(c["xy"], dtype=onp.float64), (self.scene.num_envs, 1))
        fx = onp.array([[*f["center_xy"]] for f in LAYOUT["fixtures"]])
        out = []
        for e in range(self.scene.num_envs):
            t = self.trajs[e].cpu().numpy()
            p0, p1 = t[0, 4:6], t[-1, 4:6]
            d = (p1 - p0) / onp.linalg.norm(p1 - p0)
            for _ in range(50):
                xy = p0 + onp.random.uniform(*c["fraction"]) * (p1 - p0) + onp.random.uniform(*c["lateral"]) * onp.array([-d[1], d[0]])
                if (onp.linalg.norm(fx - xy, axis=1) - FIXTURE_RADIUS - c["radius"] > 0.01).all():
                    break
            out.append(xy)
        return onp.array(out)

    def _place_obstacles(self, xy: onp.ndarray, height: float, origin: str = "center"):
        """Move each env's (visual-only) obstacle prim; positions are env-local. `origin` is where the
        prim's frame sits on the object: "center" (cylinder primitive) or "base" (scanned mesh)."""
        from pxr import Gf
        stage = self.sim.stage
        for e in range(self.scene.num_envs):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/Obstacle")
            attr = prim.GetAttribute("xformOp:translate")
            vec = Gf.Vec3d if "d" in str(attr.GetTypeName()) else Gf.Vec3f
            attr.Set(vec(float(xy[e, 0]), float(xy[e, 1]), float(TABLE_Z + (height / 2 if origin == "center" else 0.0))))

    def _plan_with_obstacles(self) -> dict:
        from demo_repair import World
        c = self.obstacle_cfg
        xy = self._sample_obstacles()
        self._place_obstacles(xy, c["height"], c.get("origin", "center"))

        base = World.from_layout(LAYOUT)
        qs, objs, tcps, ok, status, log = [], [], [], [], [], []
        for e in range(self.scene.num_envs):
            world = base.with_cylinder(float(xy[e, 0]), float(xy[e, 1]), c["radius"], c["height"])
            if c.get("compare") and e > 0:  # same scene in every env: plan once
                pass
            else:
                out = self.planner.run(world, obj_traj=self.trajs[e].cpu().numpy().astype(onp.float64), repair=self.repair)
            good = out["q"] is not None
            q = out["q"] if good else out["q_baseline"]  # failed envs still move (and are dropped)
            nom = out["nominal"]
            if c.get("compare") and e == 0:  # figure mode: env 0 shows the plain generator's motion
                q, nom, good = out["q_baseline"], out["nominal_baseline"], True
            qs.append(q)
            objs.append(self.planner.objects_from_q(q, nom))
            tcps.append(onp.asarray(self.planner.fk_tcp(q).wxyz_xyz))
            ok.append(good)
            status.append(out["status"])
            shown = "generator" if c.get("compare") and e == 0 else "planned"
            chk = out["before"] if shown == "generator" else out.get("after")
            log.append(dict(episode=self.episodes, env=e, obstacle_xy=xy[e].tolist(), status=out["status"], shown=shown,
                            obstacle_clearance_per_step=None if chk is None else onp.round(onp.asarray(chk["obstacle_clearance_per_step"]), 4).tolist(),
                            generator_ok=bool(out["before"]["ok"]), seed=out.get("seed"),
                            grasp_yaw_k=int(nom["grasp_yaw_k"]), time_s=round(out["time_s"], 2),
                            baseline_min_clearance=out["before"]["min_obstacle_clearance"],
                            min_clearance=out["after"]["min_obstacle_clearance"] if out.get("after") else None))
        if self.output_dir is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.output_dir) / "plans.jsonl", "a") as f:
                for r in log:
                    f.write(json.dumps(r) + "\n")
        print("[INFO]: plans:", [r["status"] for r in log], flush=True)
        T = lambda a: torch.tensor(onp.stack(a), dtype=torch.float32, device=self.device)  # noqa: E731
        return dict(q=T(qs), obj=T(objs), tcp=T(tcps), ok=torch.tensor(ok, device=self.device), status=status)

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
        if self.plan is not None:
            return self._step_world_planned(count, name, dt)
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

    def _step_world_planned(self, count: int, name: str, dt: float):
        """Replay the planned joints; the cube follows the gripper while held."""
        i = max(count - self.phases.bounds["approach"][0], 0)
        obj = self.plan["obj"][:, i]
        self._obj_now = obj
        state = torch.zeros_like(self.cube.data.root_state_w)
        state[:, :3] = obj[:, 4:] + self.scene.env_origins
        state[:, 3:7] = obj[:, :4]
        self.cube.write_root_state_to_sim(state)
        self.tcp_target = tf.SE3(self.plan["tcp"][:, i])
        q = self.robot.data.joint_pos.clone()
        q[:, self.arm_ids] = self.plan["q"][:, i]
        self.gripper_cmd = self._gripper_q(name, self.phases.phase(count)[1])
        q[:, self.gripper_ids] = self.gripper_cmd * self.gripper_mult
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt)
        self.phase_name = name

    # ------------------------------------------------------------------ feasibility filter
    def _fail(self, mask: torch.Tensor, reason: str):
        if self.obstacle_cfg is not None and self.obstacle_cfg.get("compare"):
            mask = mask.clone()
            mask[0] = False  # figure mode: keep the plain generator's (colliding) motion for display
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
