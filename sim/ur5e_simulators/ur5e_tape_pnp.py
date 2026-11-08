"""UR5e + Robotiq 2F-85 tape-hanging data generation (R2R2R-style kinematic replay).

Same loop, cameras, randomization and logging as UR5eCubePnP; what changes is the object and how the
motion is made. Per reset it:
  * turns the mug tree about its post by a yaw drawn from `tree_yaw_range` (the measured tree,
    turned so the branch the human used faces the robot's side; scripts/ur_tape/sweep_tree_yaw.py),
    turns the tracked roll trajectory with it, samples a new roll start on the bench (xy jitter +
    spin about the vertical axis) and warps the leading 60% of the trajectory to it (trajgen),
    keeping the hung pose on the branch fixed,
  * plans the arm with scripts/ur_tape/tape_plan.py: the grasp pinches the band from the roll's top
    face at the rim azimuth whose plan is collision-free against the measured mug tree and has the
    shortest joint path; whole-sequence IK; rejected envs (IK error, joint limits, fingertips in the
    bench, arm/gripper within 5 mm of the tree) are dropped like the cube generator's failures,
  * replays the joints kinematically with the roll (band + core) on the tracked/warped trajectory.
Per-env plan outcomes go to <output_dir>/plans.jsonl.

With `obstacle=dict(radius, height, origin, xy=..., compare=...)` a scanned object is placed (visual
only) and the motion comes from scripts/ur_tape/ring_repair.py: checked against it and, with
`repair=True`, its free segments re-optimized; `compare=True` makes env 0 replay the plain
generator's motion next to the repaired one (the before/after figure, make_repair_compare.py).
"""

import json
import sys
import time
from pathlib import Path

import numpy as onp
import torch

import real2render2real.utils.transforms as tf


def jnp_asarray(x):
    import jax.numpy as jnp
    return jnp.asarray(x)
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_cube_pnp import UR5eCubePnP
from trajgen.traj_interp import traj_interp_batch


class UR5eTapePnP(UR5eCubePnP):
    def __init__(self, *args, start_xy_noise: float = 0.04, start_yaw_noise: float = onp.pi,
                 tree_yaw_range: tuple = (250.0, 250.0), planner_kwargs: dict | None = None,
                 fixture_prim: str = "MugTree", start_offset_xy: tuple = (0.0, 0.0), task: str = "tape",
                 obstacle: dict | None = None, **kwargs):
        """planner_kwargs: TapePlanner arguments (scene/dims files, grasp style) for other ring tasks
        such as the nut; fixture_prim: the prim turned with `tree_yaw_range` (none turned if the range
        is (0, 0)); start_offset_xy: shifts every sampled start, e.g. into reach when the human's start
        is outside it."""
        self.tree_yaw_range = tree_yaw_range
        self.planner_kwargs, self.fixture_prim = planner_kwargs or {}, fixture_prim
        self.start_offset_xy = start_offset_xy
        self.task = task
        # kept apart from the cube generator's obstacle_cfg, which would build the cube's planner
        self.ring_obstacle = obstacle
        super().__init__(*args, start_xy_noise=start_xy_noise, start_yaw_noise=start_yaw_noise, **kwargs)

    def _planner(self):
        if getattr(self, "tape_planner", None) is None:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ur_tape"))
            from tape_plan import TapePlanner
            self.tape_planner = TapePlanner(**self.planner_kwargs)
            self.core = self.scene.rigid_objects.get("core")
        return self.tape_planner

    def _handle_reset(self, final: bool = False):
        if self.success_envs is not None:
            print(f"[INFO]: Episode {self.episodes}: success envs {self.success_envs.cpu().numpy().astype(int).tolist()}, "
                  f"failures by reason {self.fail_reasons}", flush=True)
            if self.data_logger is not None:
                while not self.data_logger.task_queue.empty():
                    time.sleep(0.02)
                time.sleep(0.2)
                if getattr(self, "obj_log", None):
                    poses = onp.stack(self.obj_log, axis=1)
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
        if final:
            return
        planner = self._planner()

        # New roll starts: spin about the vertical axis and xy jitter on the bench, same height.
        start = tf.SE3(torch.tensor(self.demo_traj[0], dtype=torch.float32))
        yaw = (torch.rand(n) * 2 - 1) * self.start_yaw_noise
        dxy = (torch.rand(n, 2) * 2 - 1) * self.start_xy_noise
        ns = (tf.SE3.from_rotation(tf.SO3.from_z_radians(yaw)) @ tf.SE3(start.wxyz_xyz.repeat(n, 1))).wxyz_xyz.clone()
        ns[:, 4:] = start.wxyz_xyz[4:] + torch.cat([dxy + torch.tensor(self.start_offset_xy, dtype=torch.float32), torch.zeros(n, 1)], dim=1)
        from tape_plan import rotate_traj_about_post
        self.tree_yaws = onp.random.uniform(*self.tree_yaw_range, size=n)
        trajs = []
        for e in range(n):
            turned = rotate_traj_about_post(self.demo_traj, onp.radians(self.tree_yaws[e]), planner.post_xy)
            trajs.append(onp.asarray(traj_interp_batch(traj=turned, new_starts=ns[e:e + 1].numpy(), proportion=0.6)[0]))
        trajs = onp.stack(trajs).astype(onp.float64)
        if tuple(self.tree_yaw_range) != (0.0, 0.0):
            self._turn_trees(self.tree_yaws, planner.post_xy)
        self.trajs = torch.tensor(trajs, dtype=torch.float32, device=self.device)

        qs, objs, tcps, ok, status, log = [], [], [], [], [], []
        if self.ring_obstacle is not None:
            qs, objs, tcps, ok, status, log = self._plan_with_obstacle(trajs)
        for e in range(n if self.ring_obstacle is None else 0):
            t0 = time.time()
            planner.set_tree_yaw(float(self.tree_yaws[e]))
            p = planner.plan(trajs[e])
            qs.append(p["q"]); objs.append(p["obj"]); tcps.append(p["tcp"])
            ok.append(p["ok"]); status.append(p["status"])
            log.append(dict(episode=self.episodes, env=e, status=p["status"], azimuth_deg=p["azimuth_deg"],
                            tree_yaw_deg=round(float(self.tree_yaws[e]), 2),
                            tree_clearance=round(p["tree_clearance"], 4), table_clearance=round(p["table_clearance"], 4),
                            fingertip_clearance=round(p["fingertip_clearance"], 4),
                            max_pos_err_mm=round(p["max_pos_err_mm"], 2), start_xy=trajs[e][0, 4:6].round(4).tolist(),
                            tries=p["tries"], time_s=round(time.time() - t0, 2)))
        if self.output_dir is not None:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(self.output_dir) / "plans.jsonl", "a") as f:
                for r in log:
                    f.write(json.dumps(r) + "\n")
        print("[INFO]: plans:", status, flush=True)
        T = lambda a: torch.tensor(onp.stack(a), dtype=torch.float32, device=self.device)  # noqa: E731
        self.plan = dict(q=T(qs), obj=T(objs), tcp=T(tcps), ok=torch.tensor(ok, device=self.device), status=status)
        self.start_tcp = self._tcp_pose_b()
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_appearance()
        self.success_envs = torch.ones(n, dtype=torch.bool, device=self.device)
        self.fail_reasons = {}
        for e, st in enumerate(status):
            if not ok[e]:
                self._fail(torch.arange(n, device=self.device) == e, st)

    def _plan_with_obstacle(self, trajs):
        """Plan each env with the obstacle (ring_repair.RingRepair); move the obstacle prims."""
        from ring_repair import RingRepair
        c, n = self.ring_obstacle, self.scene.num_envs
        if getattr(self, "ring_rep", None) is None:
            self.ring_rep = RingRepair(self.task)
        rep = self.ring_rep
        xy = onp.tile(onp.asarray(c["xy"], onp.float64), (n, 1))
        qs, objs, tcps, ok, status, log = [], [], [], [], [], []
        for e in range(n):
            t0 = time.time()
            rep.planner.set_tree_yaw(float(self.tree_yaws[e]))
            world = rep.world().with_cylinder(float(xy[e, 0]), float(xy[e, 1]), c["radius"], c["height"])
            if c.get("compare") and e > 0:
                pass  # same scene in every env: plan once
            else:
                out = rep.run(world, obj_traj=trajs[e], repair=self.repair)
            good = out["q"] is not None
            q, nom = (out["q"], out["nominal"]) if good else (out["q_baseline"], out["nominal_baseline"])
            shown = "planned"
            if c.get("compare") and e == 0:
                q, nom, good, shown = out["q_baseline"], out["nominal_baseline"], True, "generator"
            qs.append(onp.asarray(q))
            objs.append(rep.objects_from_q(q, nom))
            tcps.append(onp.asarray(rep.fk_tcp(jnp_asarray(q)).wxyz_xyz))
            ok.append(good)
            status.append(out["status"])
            chk = out["before"] if shown == "generator" else out.get("after") or out["before"]
            log.append(dict(episode=self.episodes, env=e, obstacle_xy=xy[e].tolist(), status=out["status"], shown=shown,
                            obstacle_clearance_per_step=onp.round(onp.asarray(chk["obstacle_clearance_per_step"]), 4).tolist(),
                            generator_ok=bool(out["before"]["ok"]), baseline_status=out["baseline_status"],
                            baseline_min_clearance=out["before"]["min_obstacle_clearance"],
                            min_clearance=out["after"]["min_obstacle_clearance"] if out.get("after") else None,
                            grasp_azimuth=int(nom["grasp_yaw_k"]), tree_yaw_deg=round(float(self.tree_yaws[e]), 2),
                            time_s=round(time.time() - t0, 2)))
        self._place_obstacles(xy, c["height"], c.get("origin", "base"))
        return qs, objs, tcps, ok, status, log

    def _turn_trees(self, yaws_deg, post_xy):
        """Rotate each env's mug tree prim about the post axis (its mesh is in the base frame)."""
        from pxr import Gf
        stage = self.sim.stage
        c = onp.r_[post_xy, 0.0]
        for e, yaw in enumerate(yaws_deg):
            prim = stage.GetPrimAtPath(f"/World/envs/env_{e}/{self.fixture_prim}")
            psi = onp.radians(yaw)
            Rz = onp.array([[onp.cos(psi), -onp.sin(psi), 0], [onp.sin(psi), onp.cos(psi), 0], [0, 0, 1]])
            t = c - Rz @ c
            tr, orient = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
            vec = Gf.Vec3d if "d" in str(tr.GetTypeName()) else Gf.Vec3f
            tr.Set(vec(*[float(x) for x in t]))
            quat, qv = (Gf.Quatd, Gf.Vec3d) if "d" in str(orient.GetTypeName()) else (Gf.Quatf, Gf.Vec3f)
            orient.Set(quat(float(onp.cos(psi / 2)), qv(0.0, 0.0, float(onp.sin(psi / 2)))))

    def _fail(self, mask: torch.Tensor, reason: str):
        if self.ring_obstacle is not None and self.ring_obstacle.get("compare"):
            mask = mask.clone()
            mask[0] = False  # figure mode: keep the plain generator's (colliding) motion for display
        super()._fail(mask, reason)

    def _write_roll(self, obj: torch.Tensor):
        state = torch.zeros_like(self.cube.data.root_state_w)
        state[:, :3] = obj[:, 4:] + self.scene.env_origins
        state[:, 3:7] = obj[:, :4]
        self.cube.write_root_state_to_sim(state)
        if self.core is not None:
            self.core.write_root_state_to_sim(state.clone())

    def _step_world(self, count: int, name: str, s: float, dt: float):
        i = max(count - self.phases.bounds["approach"][0], 0)
        obj = self.plan["obj"][:, i]
        self._obj_now = obj
        self._write_roll(obj)
        self.tcp_target = tf.SE3(self.plan["tcp"][:, i])
        q = self.robot.data.joint_pos.clone()
        q[:, self.arm_ids] = self.plan["q"][:, i]
        self.gripper_cmd = self._gripper_q(name, s)
        q[:, self.gripper_ids] = self.gripper_cmd * self.gripper_mult
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt)
        self.phase_name = name

    def _check_feasibility(self, name: str):
        """The planner already checked collisions; here only that the replay reproduces the plan."""
        tcp = self._tcp_pose_b()
        target = self.tcp_target
        pos_err = (tcp.translation() - target.translation()).norm(dim=-1)
        rot_err = (tcp.rotation().inverse() @ target.rotation()).log().norm(dim=-1)
        self._fail((pos_err > 0.02) | (rot_err > 0.45), "ik_error")
        q = self.robot.data.joint_pos[:, self.arm_ids]
        lim = self.robot.data.joint_pos_limits[:, self.arm_ids]
        self._fail(((q < lim[..., 0]) | (q > lim[..., 1])).any(dim=-1), "joint_limit")
