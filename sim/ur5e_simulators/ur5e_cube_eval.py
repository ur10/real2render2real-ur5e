"""Closed-loop evaluation of a visuomotor policy in the UR5e cube scene (the same digital twin the
demonstrations are rendered in).

Per batch of envs: the cube starts at a held-out scene's pose (scripts/ur_cube/make_eval_scenes.py),
the optional bottle is placed, lighting / camera / table are drawn from a seed that depends only on
the batch index (so every policy sees the same scenes). The policy sees the two rendered RealSense
views (same downscale, JPEG and crop as its training data) and the joint state, and outputs joint
targets that are applied kinematically, as in data generation.

Contact is not simulated. The cube is grasped when the gripper closes with the cube between the
fingers (cube centre within 1.5 cm of the grasp point the demonstrations use, finger direction within
20 deg of a cube face), then follows the gripper; when the gripper opens it is set down level on what
is under it (a fixture top or the table). An episode succeeds if the cube ends on the goal fixture,
released, and neither the robot nor the held cube ever penetrated the bottle (sphere model from
demo_repair.py, zero margin). Arm contact with the table/fixtures is reported, not scored.
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as onp
import torch

import real2render2real.utils.transforms as tf
from real2render2real.isaaclab_viser.configs.articulation_configs.ur5e_robotiq_cfg import ROBOTIQ_CLOSED, ROBOTIQ_OPEN
from real2render2real.isaaclab_viser.configs.scene_configs.ur5e_scene_cfg import (
    CUBE_SIZE, FIXTURE_RADIUS, LAYOUT, TABLE_Z,
)
from real2render2real.isaaclab_viser.ur5e_simulators.ur5e_cube_pnp import UR5eCubePnP, downscale_rgb
from trajgen.traj_interp import traj_interp_batch

GRASP_TOL, GRASP_YAW_TOL = 0.015, onp.radians(20)


class UR5eCubeEval(UR5eCubePnP):
    def __init__(self, *args, policy_path: Path, scenes: list, eval_seed: int = 777, exec_steps: int = 8,
                 max_steps: int = 180, save_video_envs: int = 0, max_dq: float = 0.1, **kwargs):
        self.policy_path, self.scenes, self.eval_seed = Path(policy_path), scenes, eval_seed
        self.exec_steps, self.max_steps, self.save_video_envs, self.max_dq = exec_steps, max_steps, save_video_envs, max_dq
        self._ep = None
        super().__init__(*args, **kwargs)

    def run_simulator(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/ur_cube"))
        import policy as pol
        self.pol = pol
        self.policy = pol.load_policy(self.policy_path, device=self.device)
        n = self.scene.num_envs
        self.max_episodes = (len(self.scenes) + n - 1) // n
        self.results = []
        self.phases.total = self.phases.setup + self.max_steps  # one eval episode per loop "episode"
        super().run_simulator()
        s = self.summary()
        print("[EVAL]", json.dumps(s), flush=True)
        (Path(self.output_dir) / "summary.json").write_text(json.dumps(s, indent=1))

    # ------------------------------------------------------------------ reset
    def _handle_reset(self, final: bool = False):
        if self._ep is not None:
            self._finish_episode()
        if final:
            return
        n, b = self.scene.num_envs, self.episodes
        idx = [min(b * n + e, len(self.scenes) - 1) for e in range(n)]
        valid = [b * n + e < len(self.scenes) for e in range(n)]
        sc = [self.scenes[i] for i in idx]
        onp.random.seed(self.eval_seed + b)
        random.seed(self.eval_seed + b)
        torch.manual_seed(self.eval_seed + b)

        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, self.robot.data.default_joint_vel)

        start = tf.SE3(torch.tensor(self.demo_traj[0], dtype=torch.float32))
        yaw = torch.tensor([s["yaw"] for s in sc], dtype=torch.float32)
        dxy = torch.tensor([[s["dx"], s["dy"]] for s in sc], dtype=torch.float32)
        ns = (tf.SE3.from_rotation(tf.SO3.from_z_radians(yaw)) @ tf.SE3(start.wxyz_xyz.repeat(n, 1))).wxyz_xyz.clone()
        ns[:, 4:] = start.wxyz_xyz[4:] + torch.cat([dxy, torch.zeros(n, 1)], dim=1)
        trajs = onp.asarray(traj_interp_batch(traj=self.demo_traj, new_starts=ns.numpy(), proportion=0.6))
        self.trajs = torch.tensor(trajs, dtype=torch.float32, device=self.device)

        xy = onp.array([s["bottle_xy"] if s["bottle_xy"] is not None else [0.0, 30.0] for s in sc])
        c = self.obstacle_cfg
        self._place_obstacles(xy, c["height"], c.get("origin", "center"))
        self.randomize_lighting()
        self.randomize_viewaug()
        self.randomize_appearance()

        fx = onp.array([f["center_xy"] for f in LAYOUT["fixtures"]])
        goal = onp.argmin(onp.linalg.norm(fx[None] - trajs[:, -1, None, 4:6], axis=-1), axis=1)
        self._ep = dict(
            idx=idx, valid=valid, scenes=sc, goal=goal, nominal_place=trajs[:, -1, 4:7],
            cube=self.trajs[:, 0].clone(), held=torch.zeros(n, dtype=torch.bool, device=self.device),
            rel=torch.tensor([[1.0, 0, 0, 0, 0, 0, 0]], device=self.device).repeat(n, 1),
            was_closed=torch.zeros(n, dtype=torch.bool, device=self.device),
            chunk=None, q_log=[], held_log=[], cube_log=[], grasp_events=[0] * n, frames=[],
            t0=time.time(),
        )
        self.success_envs = torch.ones(n, dtype=torch.bool, device=self.device)
        self.fail_reasons = {}
        self._write_cube(self._ep["cube"])

    # ------------------------------------------------------------------ stepping
    def _write_cube(self, obj: torch.Tensor):
        self._obj_now = obj
        state = torch.zeros_like(self.cube.data.root_state_w)
        state[:, :3] = obj[:, 4:] + self.scene.env_origins
        state[:, 3:7] = obj[:, :4]
        self.cube.write_root_state_to_sim(state)

    def _obs(self):
        imgs = []
        for cam in self.pol.CAMERAS:
            rgb = downscale_rgb(self.camera_manager.buffers[cam][0]["rgb"], 2).cpu().numpy()
            imgs.append(onp.stack([self.pol.crop_resize(self.pol.jpeg_roundtrip(im), cam) for im in rgb]))
        imgs = torch.tensor(onp.stack(imgs, 1), device=self.device)  # (n, cams, H, W, 3)
        q = torch.cat([self.robot.data.joint_pos[:, self.arm_ids], self.robot.data.joint_pos[:, self.gripper_ids[:1]]], 1)
        return imgs, q

    def _step_world(self, count: int, name: str, s: float, dt: float):
        ep, n = self._ep, self.scene.num_envs
        q = self.robot.data.joint_pos.clone()
        k = count - self.phases.setup
        if k >= 0:
            if k % self.exec_steps == 0:
                imgs, q_obs = self._obs()
                ep["chunk"] = self.policy.act(imgs, q_obs)
            tgt = ep["chunk"][:, k % self.exec_steps]
            cur = q[:, self.arm_ids]
            q[:, self.arm_ids] = cur + (tgt[:, :6] - cur).clamp(-self.max_dq, self.max_dq)
            finger = tgt[:, 6].clamp(ROBOTIQ_OPEN, ROBOTIQ_CLOSED)
            q[:, self.gripper_ids] = finger[:, None] * self.gripper_mult
        else:  # setup: hold home, let the cameras render the new scene
            q = self.robot.data.default_joint_pos.clone()
            finger = torch.full((n,), ROBOTIQ_OPEN, device=self.device)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt)

        # grasp / carry / release (no contact dynamics, see module docstring)
        tcp = self._tcp_pose_b()
        closed = finger > 0.5 * self.gripper_close
        cube = tf.SE3(ep["cube"])
        closing = closed & ~ep["was_closed"] & ~ep["held"]
        if closing.any():
            rel = tcp.inverse() @ cube  # cube in the TCP frame
            g_inv = tf.SE3(self.grasp_obj_tcp[None]).inverse()  # nominal cube-in-TCP
            dp = (rel.translation() - g_inv.translation()).norm(dim=-1)
            # yaw of the cube's x axis vs the TCP's, modulo the cube's 90 deg symmetry
            Rrel = rel.rotation().as_matrix()
            yaw = torch.atan2(Rrel[:, 1, 0], Rrel[:, 0, 0])
            Rg = g_inv.rotation().as_matrix()
            yaw_g = torch.atan2(Rg[:, 1, 0], Rg[:, 0, 0])
            dyaw = torch.remainder(yaw - yaw_g + onp.pi / 4, onp.pi / 2) - onp.pi / 4
            grab = closing & (dp < GRASP_TOL) & (dyaw.abs() < GRASP_YAW_TOL)
            if grab.any():
                ep["held"] = ep["held"] | grab
                ep["rel"][grab] = rel.wxyz_xyz[grab]
                for e in torch.where(grab)[0].tolist():
                    ep["grasp_events"][e] += 1
        release = ep["held"] & ~closed
        if release.any():
            ep["held"] = ep["held"] & ~release
            ep["cube"][release] = self._set_down(ep["cube"][release])
        if ep["held"].any():
            carried = (tcp @ tf.SE3(ep["rel"])).wxyz_xyz
            ep["cube"][ep["held"]] = carried[ep["held"]]
        ep["was_closed"] = closed
        self._write_cube(ep["cube"])
        if k >= 0:
            ep["q_log"].append(q[:, self.arm_ids].cpu().numpy())
            ep["held_log"].append(ep["held"].cpu().numpy())
            ep["cube_log"].append(ep["cube"].cpu().numpy())

    def _set_down(self, obj: torch.Tensor) -> torch.Tensor:
        """Level the cube (keep its yaw) and rest it on the fixture top or table under its centre."""
        out = obj.clone()
        Rm = tf.SE3(obj).rotation().as_matrix()
        # yaw of whichever cube axis is most horizontal-aligned: use the x axis projected on the table
        yaw = torch.atan2(Rm[:, 1, 0], Rm[:, 0, 0])
        out[:, :4] = tf.SO3.from_z_radians(yaw).wxyz
        support = torch.full((obj.shape[0],), TABLE_Z, device=obj.device)
        for f in LAYOUT["fixtures"]:
            on = (obj[:, 4:6] - torch.tensor(f["center_xy"], device=obj.device)).norm(dim=-1) < FIXTURE_RADIUS
            support = torch.where(on, torch.full_like(support, f["top_z"]), support)
        out[:, 6] = support + CUBE_SIZE / 2
        return out

    def _render(self):
        super()._render()
        if self._ep is not None and self.episodes == 1 and self.save_video_envs > 0:
            rgb = downscale_rgb(self.camera_manager.buffers["camera_0"][0]["rgb"][: self.save_video_envs], 2)
            self._ep["frames"].append(rgb.cpu().numpy())

    def _check_feasibility(self, name: str):
        pass

    def _log_data(self, step: int):
        pass

    # ------------------------------------------------------------------ scoring
    def _finish_episode(self):
        import jax.numpy as jnp
        from demo_repair import World, _sdf_cyl
        ep = self._ep
        if not ep["q_log"]:
            self._ep = None
            return
        q = onp.stack(ep["q_log"], 1)  # (n, T, 6)
        held = onp.stack(ep["held_log"], 1)
        cube = onp.stack(ep["cube_log"], 1)  # (n, T, 7)
        base = World.from_layout(LAYOUT)
        fx = onp.array([f["center_xy"] for f in LAYOUT["fixtures"]])
        cl, cr = onp.asarray(self.planner.cube_local), float(self.planner.cube_r)
        c = self.obstacle_cfg
        for e in range(q.shape[0]):
            if not ep["valid"][e]:
                continue
            sc = ep["scenes"][e]
            world = base if sc["bottle_xy"] is None else base.with_cylinder(*sc["bottle_xy"], c["radius"], c["height"])
            arrays = World.arrays(world)
            no_hold = jnp.zeros(q.shape[1], bool)
            ident = __import__("jaxlie").SE3.identity()
            d_obs, d_env = self.planner.distances(jnp.asarray(q[e], jnp.float32), arrays, no_hold, ident)
            d_obs_min = float(onp.asarray(d_obs).min())
            if held[e].any():  # held cube spheres against the bottle
                Tc = __import__("jaxlie").SE3(jnp.asarray(cube[e][held[e]], jnp.float32))
                pts = onp.asarray(jnp.stack([Tc @ jnp.asarray(p, jnp.float32) for p in cl], 1))
                d_cube = onp.asarray(_sdf_cyl(jnp.asarray(pts), jnp.asarray(arrays[2])).min(-1)) - cr
                d_obs_min = min(d_obs_min, float(d_cube.min()))
            final = cube[e, -1]
            g = int(ep["goal"][e])
            on_goal = bool(onp.linalg.norm(final[4:6] - fx[g]) < FIXTURE_RADIUS and not held[e, -1]
                           and abs(final[6] - (LAYOUT["fixtures"][g]["top_z"] + CUBE_SIZE / 2)) < 0.005)
            hit = sc["bottle_xy"] is not None and d_obs_min < 0.0
            r = dict(scene=sc["id"], grasped=bool(held[e].any()), grasp_events=ep["grasp_events"][e],
                     on_goal=on_goal, bottle_collision=bool(hit),
                     min_bottle_clearance=None if sc["bottle_xy"] is None else round(d_obs_min, 4),
                     min_env_clearance=round(float(onp.asarray(d_env).min()), 4),
                     place_err_cm=round(float(onp.linalg.norm(final[4:6] - ep["nominal_place"][e][:2]) * 100), 2),
                     success=bool(on_goal and not hit))
            self.results.append(r)
            with open(Path(self.output_dir) / "results.jsonl", "a") as f:
                f.write(json.dumps(r) + "\n")
        if ep["frames"]:
            fr = onp.stack(ep["frames"], 1)  # (envs, T, H, W, 3)
            import cv2
            for e in range(fr.shape[0]):
                d = Path(self.output_dir) / "video" / ep["scenes"][e]["id"]
                d.mkdir(parents=True, exist_ok=True)
                for t, im in enumerate(fr[e]):
                    cv2.imwrite(str(d / f"{t:04d}.jpg"), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        done = sum(ep["valid"])
        ok = sum(r["success"] for r in self.results[-done:])
        print(f"[EVAL] batch {self.episodes}: {ok}/{done} success, total {sum(r['success'] for r in self.results)}"
              f"/{len(self.results)} ({time.time() - ep['t0']:.0f}s)", flush=True)
        self._ep = None

    def summary(self) -> dict:
        R = self.results
        k = lambda key: sum(r[key] for r in R)  # noqa: E731
        return dict(policy=str(self.policy_path), episodes=len(R), success=k("success"),
                    success_rate=round(k("success") / max(len(R), 1), 3), grasped=k("grasped"),
                    on_goal=k("on_goal"), bottle_collision=k("bottle_collision"))
