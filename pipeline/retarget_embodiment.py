"""Cross-embodiment retargeting: the same tracked cube trajectory, executed by different arms.

R2R2R's robot motion comes from the object's motion, not from a robot demonstration: the generator
follows a grasp on the tracked part with IK. Nothing in that ties it to the arm, so the same cube
trajectory can be rendered for any arm that can reach it. This builds, for each embodiment, the
generator's phases (approach -> close -> carry -> release -> retreat; demo_repair.Phases) from the
cleaned demo track, solves whole-sequence IK (jaxls, as DemoRepair.solve_track) and reports the
TCP tracking error.

  ur5e    UR5e + Robotiq 2F-85 on the bench mount (the robot the demo was recorded for)
  ur10e   UR10e + Robotiq 2F-85 on the same mount
  panda   Franka Panda + Franka Hand standing on the table beside the fixtures

The grasp is the demo's (cube -> TCP) grasp, so all three hold the cube the same way; the Panda's TCP
frame is turned about its approach axis so its fingers close on the same cube faces.

Example (JAX on GPU 1):
    CUDA_VISIBLE_DEVICES=1 python scripts/ur_cube/retarget_embodiment.py --output-dir outputs/ur_cube/embodiments
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
import jaxls  # noqa: E402
import numpy as onp  # noqa: E402
from jaxmp import JaxKinTree  # noqa: E402
from jaxmp.extras.urdf_loader import load_urdf  # noqa: E402
from jaxmp.robot_factors import RobotFactors  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from demo_repair import DEMO_FILE, LAYOUT_FILE, UR5E_HOME, Phases, level_poses, minimal_yaw  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
TABLE_Z = json.loads(LAYOUT_FILE.read_text())["table_z"]


def se3(xyz, yaw=0.0) -> jaxlie.SE3:
    return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_z_radians(yaw), jnp.asarray(xyz, jnp.float32))


ROBOTS = {
    "ur5e": dict(urdf="data/ur5e_robotiq/ur5e_robotiq85.urdf", ik="data/ur5e_robotiq/ur5e_robotiq85_ik.urdf",
                 tcp="robotiq_85_base_link-tcp", home=UR5E_HOME, base=((0.0, 0.0, 0.0), 0.0), tool_yaw=0.0,
                 gripper=dict(finger_joint=(0.0, 0.42)), label="UR5e + Robotiq 2F-85"),
    "ur10e": dict(urdf="data/ur10e_robotiq/ur10e_robotiq85.urdf", ik="data/ur10e_robotiq/ur10e_robotiq85_ik.urdf",
                  tcp="robotiq_85_base_link-tcp", home=UR5E_HOME, base=((0.0, 0.0, 0.0), 0.0), tool_yaw=0.0,
                  gripper=dict(finger_joint=(0.0, 0.42)), label="UR10e + Robotiq 2F-85"),
    # standing on the table top, 0.47-0.49 m from the two fixtures, facing them
    "panda": dict(urdf="data/panda/panda.urdf", ik="data/panda/panda_ik.urdf", tcp="panda_hand_tcp_joint",
                  home=onp.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]),
                  base=((0.10, -0.30, TABLE_Z), -onp.pi / 2), tool_yaw=onp.pi / 2,
                  gripper=dict(panda_finger_joint1=(0.04, 0.02), panda_finger_joint2=(0.04, 0.02)),
                  label="Franka Panda + Franka Hand"),
}


class Arm:
    def __init__(self, spec: dict):
        self.spec = spec
        self.kin = JaxKinTree.from_urdf(load_urdf(None, REPO / spec["ik"]))
        self.dof = self.kin.num_actuated_joints
        self.tcp_idx = list(self.kin.joint_names).index(spec["tcp"])
        self.home = jnp.asarray(spec["home"], jnp.float32)
        self.base = se3(*spec["base"])
        self.JV = RobotFactors.get_var_class(self.kin, default_val=self.home)
        names = onp.array(self.kin.joint_names)
        self.joint_names = [str(n) for n in names[onp.asarray(self.kin.idx_actuated_joint) >= 0]]

    def fk_world(self, q) -> jaxlie.SE3:
        return self.base @ jaxlie.SE3(self.kin.forward_kinematics(q)[..., self.tcp_idx, :])

    def solve(self, targets_world: jaxlie.SE3, smooth: float = 0.3) -> jnp.ndarray:
        tgt = (self.base.inverse() @ targets_world).wxyz_xyz
        n = tgt.shape[0]
        key = (n, smooth)
        if key not in getattr(self, "_jit", {}):  # compile once per sequence length
            self._jit = getattr(self, "_jit", {})
            self._jit[key] = self._make_solver(n, smooth)
        return self._jit[key](tgt, jnp.broadcast_to(self.home, (n, self.dof)))

    def _make_solver(self, n: int, smooth: float):
        JV, kin, home = self.JV, self.kin, self.home

        def pose_res(v, x, t):
            r = (jaxlie.SE3(kin.forward_kinematics(v[x])[..., self.tcp_idx, :]).inverse() @ t).log()
            return jnp.concatenate([r[:3] * 100.0, r[3:] * 20.0])

        @jax.jit
        def run(tgt, q0):
            fs = [jaxls.Factor(pose_res, (JV(jnp.arange(n)), jaxlie.SE3(tgt))),
                  jaxls.Factor(lambda v, a, b: (v[a] - v[b]) * smooth, (JV(jnp.arange(1, n)), JV(jnp.arange(n - 1)))),
                  jaxls.Factor(lambda v, x: (v[x] - home) * 0.001, (JV(jnp.arange(n)),)),
                  RobotFactors.limit_cost_factor(JV, jnp.arange(n), kin, jnp.full(self.dof, 100.0))]
            graph = jaxls.FactorGraph.make(fs, [JV(jnp.arange(n))], use_onp=False)
            sol = graph.solve(jaxls.VarValues.make([JV(jnp.arange(n)).with_value(q0)]),
                              trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                              termination=jaxls.TerminationConfig(max_iterations=300), verbose=False)
            return sol[JV]
        return run


def nominal(obj_traj: onp.ndarray, grasp: jaxlie.SE3, home_tcp: jaxlie.SE3, ph: Phases = Phases()) -> dict:
    """The generator's phases (as DemoRepair.nominal), for a given grasp and start TCP pose."""
    objs = [jaxlie.SE3(jnp.asarray(o, jnp.float32)) for o in obj_traj]
    best = min(range(4), key=lambda k: float(jnp.linalg.norm(
        ((objs[0] @ jaxlie.SE3.from_rotation(jaxlie.SO3.from_z_radians(k * jnp.pi / 2)) @ grasp).rotation().inverse()
         @ home_tcp.rotation()).log())))
    g = jaxlie.SE3.from_rotation(jaxlie.SO3.from_z_radians(best * jnp.pi / 2)) @ grasp
    up = jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, -ph.approach_height]))
    interp = lambda a, b, t: a @ jaxlie.SE3.exp((a.inverse() @ b).log() * t)  # noqa: E731
    ss = lambda t: t * t * (3 - 2 * t)  # noqa: E731
    g0, gT = objs[0] @ g, objs[-1] @ g
    tcp, obj, grip = [], [], []
    a = 1 - ph.descent_frac
    for i in range(ph.approach):
        s = i / max(ph.approach - 1, 1)
        tcp.append(interp(home_tcp, g0 @ up, ss(s / a)) if s < a else interp(g0 @ up, g0, ss((s - a) / ph.descent_frac)))
        obj.append(objs[0]); grip.append(0.0)
    for i in range(ph.close):
        tcp.append(g0); obj.append(objs[0]); grip.append((i + 1) / ph.close)
    for o in objs:
        tcp.append(o @ g); obj.append(o); grip.append(1.0)
    for i in range(ph.release):
        tcp.append(gT); obj.append(objs[-1]); grip.append(1.0 - (i + 1) / ph.release)
    for i in range(ph.retreat):
        s = i / max(ph.retreat - 1, 1)
        tcp.append(interp(gT, gT @ up, ss(s))); obj.append(objs[-1]); grip.append(0.0)
    stack = lambda xs: jaxlie.SE3(jnp.stack([x.wxyz_xyz for x in xs]))  # noqa: E731
    held = onp.array([False] * (ph.approach + ph.close) + [True] * (len(objs) + ph.release) + [False] * ph.retreat)
    return dict(tcp=stack(tcp), obj=stack(obj), grip=onp.array(grip), held=held, grasp=g)


def main(output_dir: Path = Path("outputs/ur_cube/embodiments"), robots: tuple[str, ...] = ("ur5e", "ur10e", "panda")) -> None:
    d = json.loads(DEMO_FILE.read_text())
    demo = minimal_yaw(level_poses(onp.asarray(d["T_base_obj"], onp.float64)))
    grasp_ur = jaxlie.SE3(jnp.asarray(d["grasp_T_obj_tcp"], jnp.float32))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in robots:
        spec = ROBOTS[name]
        arm = Arm(spec)
        grasp = grasp_ur @ jaxlie.SE3.from_rotation(jaxlie.SO3.from_z_radians(spec["tool_yaw"]))
        nom = nominal(demo, grasp, arm.fk_world(arm.home))
        q = arm.solve(nom["tcp"])
        err = (arm.fk_world(q).inverse() @ nom["tcp"]).log()
        pos_mm = onp.asarray(jnp.linalg.norm(err[:, :3], axis=-1)) * 1000
        rot_deg = onp.degrees(onp.asarray(jnp.linalg.norm(err[:, 3:], axis=-1)))
        held_pose = (arm.fk_world(q) @ nom["grasp"].inverse()).wxyz_xyz
        obj = onp.where(nom["held"][:, None], onp.asarray(held_pose), onp.asarray(nom["obj"].wxyz_xyz))
        lo, hi = onp.asarray(arm.kin.limits_lower), onp.asarray(arm.kin.limits_upper)
        qa = onp.asarray(q)
        margin = float(onp.min(onp.minimum(qa - lo, hi - qa)))
        fingers = {j: (o + (c - o) * nom["grip"]).tolist() for j, (o, c) in spec["gripper"].items()}
        onp.savez(output_dir / f"{name}.npz", q=qa, obj=obj, grip=nom["grip"], held=nom["held"],
                  base=onp.asarray(arm.base.wxyz_xyz), joint_names=onp.array(arm.joint_names),
                  finger_names=onp.array(list(fingers)), fingers=onp.array(list(fingers.values())).T,
                  tcp_target=onp.asarray(nom["tcp"].wxyz_xyz))
        report[name] = dict(label=spec["label"], steps=len(qa), max_pos_err_mm=round(float(pos_mm.max()), 2),
                            max_rot_err_deg=round(float(rot_deg.max()), 2), min_joint_limit_margin_rad=round(margin, 3))
        print(name, report[name], flush=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
