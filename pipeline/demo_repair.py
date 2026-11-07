"""Collision check and repair of a generated UR5e demonstration after an obstacle is inserted.

The R2R2R-style generator (ur5e_cube_pnp.py) turns one tracked human demo into robot motion: it
warps the cube trajectory to a new start and follows it with IK through approach -> close -> carry
-> release -> retreat. Nothing in that motion knows about objects that were not in the demo. This
module (1) checks a generated motion against a scene with inserted obstacles and (2) repairs only
the parts that the edit invalidated, keeping the parts that carry the task:

  pinned (task-critical, kept as generated)      free (re-optimized)
  - straight pre-grasp -> grasp descent          - transit from the start pose to the pre-grasp
  - close, lift-off and set-down near fixtures   - the carry between lift-off and set-down
  - release and the straight retreat

Repair is one joint-space trajectory optimization (jaxls Levenberg-Marquardt), roughly

  min_q  sum_t  l1 |q_t - q_t^demo|^2 [free]  +  l2 |q_t+1 - q_t|^2  +  l3 |q_t+1 - 2 q_t + q_t-1|^2
                +  w_c max(0, m - d_t(q))^2   +  w_pose |log(T_tcp(q_t)^-1 T_t^demo)|^2 [pinned]
                +  w_up |rot. of tool vs demo|^2 [free carry]  +  joint / velocity limit hinges

where d_t is the signed distance of a sphere model of the arm, gripper, wrist-camera bracket and
held cube to the world
(table plane, fixture cylinders, obstacle cylinders/boxes). It runs from several initial guesses
(the demo, and the demo bent up / left / right around the obstacle), and a candidate is accepted
only if a denser re-check (4 substeps per step, at least 5 mm from the obstacle) passes. The held cube's poses are recomputed from the
repaired arm motion and the fixed grasp, so rendered observations stay consistent.

Example:
    python scripts/ur_cube/demo_repair.py --obstacle 0.096 -0.763 0.0328 0.1987   # scanned bottle mid-path
    python scripts/ur_cube/demo_repair.py --obstacle-at 0.5 --output outputs/ur_cube/repair/mid
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
import jaxls  # noqa: E402
import numpy as onp  # noqa: E402
import trimesh  # noqa: E402
import trimesh.bounds  # noqa: E402
from jaxmp import JaxKinTree  # noqa: E402
from jaxmp.extras.urdf_loader import load_urdf  # noqa: E402
from jaxmp.robot_factors import RobotFactors  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IK_URDF = REPO / "data/ur5e_robotiq/ur5e_robotiq85_ik.urdf"
LAYOUT_FILE = REPO / "outputs/ur_cube/scene_layout.json"
DEMO_FILE = REPO / "outputs/ur_cube/traj_ep0_f1_to_f0_rgbd.json"

# Mirrors of the simulator constants (configs/*ur5e*_cfg.py import Isaac Lab, so they are restated).
UR5E_HOME = onp.array([1.5504, -1.7068, 2.0003, -1.8996, -1.4539, 0.0417])
CUBE_SIZE = 0.04
FIXTURE_RADIUS = 0.045
FINGERTIP_BELOW_TCP = 0.03
TCP_TARGET = "robotiq_85_base_link-tcp"
GRIPPER_JOINT = "tool0-robotiq_85_base_link"


@dataclass
class Phases:
    """Step counts as in ur5e_cube_pnp.PnPPhases (setup is skipped: it only holds the start pose)."""
    approach: int = 30
    close: int = 8
    release: int = 8
    retreat: int = 14
    approach_height: float = 0.10
    descent_frac: float = 0.35  # last part of the approach: straight pre-grasp -> grasp
    lift_tol: float = 0.02      # cube within this height of its start/end rest pose = near a fixture


# ----------------------------------------------------------------------------- world
@dataclass
class World:
    """Static collision geometry in the robot base frame. Cylinders are vertical: (cx, cy, z0, z1, r)."""
    table_z: float
    fixtures: onp.ndarray                     # (2, 5)
    obs_cyl: onp.ndarray = field(default_factory=lambda: onp.zeros((0, 5)))
    obs_box: onp.ndarray = field(default_factory=lambda: onp.zeros((0, 7)))  # cx cy cz hx hy hz yaw

    @staticmethod
    def from_layout(layout: dict) -> "World":
        tz = layout["table_z"]
        fx = onp.array([[*f["center_xy"], tz, f["top_z"], FIXTURE_RADIUS] for f in layout["fixtures"]])
        return World(table_z=tz, fixtures=fx)

    def with_cylinder(self, x: float, y: float, radius: float, height: float) -> "World":
        c = onp.array([[x, y, self.table_z, self.table_z + height, radius]])
        return World(self.table_z, self.fixtures, onp.vstack([self.obs_cyl, c]), self.obs_box)

    def with_box(self, x: float, y: float, sx: float, sy: float, sz: float, yaw: float = 0.0) -> "World":
        b = onp.array([[x, y, self.table_z + sz / 2, sx / 2, sy / 2, sz / 2, yaw]])
        return World(self.table_z, self.fixtures, self.obs_cyl, onp.vstack([self.obs_box, b]))

    def to_json(self) -> dict:
        return dict(table_z=self.table_z, fixtures=self.fixtures.tolist(), obs_cyl=self.obs_cyl.tolist(),
                    obs_box=self.obs_box.tolist())

    def arrays(self):
        # Pad empty obstacle sets with one far-away dummy so shapes stay static under jit.
        cyl = self.obs_cyl if len(self.obs_cyl) else onp.array([[50.0, 50.0, 0.0, 0.01, 0.01]])
        box = self.obs_box if len(self.obs_box) else onp.array([[50.0, 50.0, 0.0, 0.01, 0.01, 0.01, 0.0]])
        return (jnp.asarray(self.table_z, jnp.float32), jnp.asarray(self.fixtures, jnp.float32),
                jnp.asarray(cyl, jnp.float32), jnp.asarray(box, jnp.float32))


def _sdf_cyl(p, c):
    """Signed distance from points p (..., 3) to vertical cylinders c (K, 5) -> (..., K)."""
    dxy = jnp.sqrt(((p[..., None, :2] - c[:, :2]) ** 2).sum(-1) + 1e-12) - c[:, 4]
    dz = jnp.maximum(c[:, 2] - p[..., None, 2], p[..., None, 2] - c[:, 3])
    q = jnp.stack([dxy, dz], -1)
    return jnp.sqrt((jnp.maximum(q, 0.0) ** 2).sum(-1) + 1e-12) + jnp.minimum(q.max(-1), 0.0)


def _sdf_box(p, b):
    """Signed distance from points p (..., 3) to z-rotated boxes b (K, 7) -> (..., K)."""
    d = p[..., None, :] - b[:, :3]
    cy, sy = jnp.cos(b[:, 6]), jnp.sin(b[:, 6])
    loc = jnp.stack([cy * d[..., 0] + sy * d[..., 1], -sy * d[..., 0] + cy * d[..., 1], d[..., 2]], -1)
    q = jnp.abs(loc) - b[:, 3:6]
    return jnp.sqrt((jnp.maximum(q, 0.0) ** 2).sum(-1) + 1e-12) + jnp.minimum(q.max(-1), 0.0)


# ----------------------------------------------------------------------------- robot sphere model
@dataclass
class SphereModel:
    joint_idx: onp.ndarray  # (S,) kinematic-tree joint whose frame each sphere rides on
    local: onp.ndarray      # (S, 3) centre in that frame
    radius: onp.ndarray     # (S,)
    group: onp.ndarray      # (S,) 0 = base/shoulder (skip vs table), 1 = arm, 2 = gripper
    names: list


ARM_LINKS = {"base_link_inertia": 0, "shoulder_link": 0, "upper_arm_link": 1, "forearm_link": 1,
             "wrist_1_link": 1, "wrist_2_link": 1, "wrist_3_link": 1}


def build_sphere_model(urdf, kin: JaxKinTree, *, inflate: float = 0.005) -> SphereModel:
    """Arm links: a chain of spheres along each collision mesh's minimum bounding cylinder.
    Gripper: hand-placed spheres for the Robotiq body and the fin-ray fingers (the real fingers are
    custom, so the stock finger meshes in the URDF are not trusted)."""
    names = list(kin.joint_names)
    child_joint = {j.child: names.index(n) for n, j in urdf.joint_map.items()}
    J, L, R, G, N = [], [], [], [], []
    for link, grp in ARM_LINKS.items():
        for c in urdf.link_map[link].collisions:
            m = trimesh.load(IK_URDF.parent / c.geometry.mesh.filename, force="mesh")
            if c.origin is not None:
                m.apply_transform(c.origin)
            cyl = trimesh.bounds.minimum_cylinder(m)
            r, h, T = cyl["radius"], cyl["height"], cyl["transform"]
            # Chain along the cylinder axis close to the end caps; sphere radius also covers the
            # cylinder surface between neighbouring centres.
            half = max(h / 2 - 0.3 * r, 0.0)
            n = max(2, int(math.ceil(2 * half / (0.6 * r))) + 1)
            gap = 2 * half / (n - 1)
            rs = math.sqrt(r * r + (gap / 2) ** 2) + inflate
            for z in onp.linspace(-half, half, n):
                J.append(child_joint[link]); L.append((T @ [0, 0, z, 1])[:3]); R.append(rs)
                G.append(grp); N.append(link)
    g = names.index(GRIPPER_JOINT)  # frame of robotiq_85_base_link; fingers open along x, approach is +z
    body = [((0, 0, 0.025), 0.045), ((0, 0, 0.065), 0.045), ((0.03, 0, 0.085), 0.025), ((-0.03, 0, 0.085), 0.025)]
    fingers = [((sx * 0.032, 0, z), 0.017) for sx in (-1, 1) for z in (0.11, 0.135, 0.16, 0.178)]
    # Yellow bracket with the two wrist cameras, along the gripper y axis at the coupling
    # (placed by overlaying candidates on RealSense frame 0).
    bracket = [((0, sy * d, -0.005), 0.03) for sy in (-1, 1) for d in (0.045, 0.075, 0.105, 0.12)]
    for (p, r) in body + fingers + bracket:
        J.append(g); L.append(onp.array(p, float)); R.append(r + inflate); G.append(2); N.append("gripper")
    return SphereModel(onp.array(J), onp.array(L), onp.array(R), onp.array(G), N)


def cube_spheres():
    """Held cube in its own frame: 8 spheres that fill the 4 cm cube to within a few mm."""
    o = CUBE_SIZE / 2 - 0.012
    return onp.array([[sx * o, sy * o, sz * o] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]), 0.014


def level_poses(traj: onp.ndarray) -> onp.ndarray:
    """Snap each cube pose (wxyz_xyz) to its nearest gravity-aligned orientation, keeping yaw and
    position. The RGB-D ICP track of the 4 cm cube drifts up to ~23 deg of tilt while the cube sits
    on a fixture, which no resting cube can do; a level cube also gives a top-down grasp."""
    from scipy.spatial.transform import Rotation
    out = onp.asarray(traj, onp.float64).copy()
    M = Rotation.from_quat(out[:, [1, 2, 3, 0]]).as_matrix()          # (T, 3, 3), columns = cube axes
    k = onp.abs(M[:, 2, :]).argmax(-1)                                # cube axis closest to world z
    a = M[onp.arange(len(M)), :, k] * onp.sign(M[onp.arange(len(M)), 2, k])[:, None]
    z = onp.array([0.0, 0.0, 1.0])
    axis = onp.cross(a, z)
    ang = onp.arcsin(onp.clip(onp.linalg.norm(axis, axis=-1), 0, 1))
    rv = axis / (onp.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12) * ang[:, None]
    lev = Rotation.from_rotvec(rv) * Rotation.from_matrix(M)
    out[:, :4] = lev.as_quat()[:, [3, 0, 1, 2]]
    return out


def minimal_yaw(traj: onp.ndarray, sym_deg: float = 90.0) -> onp.ndarray:
    """For a cube that looks the same every 90 deg about z, a tracked net yaw change of e.g. 80 deg is
    indistinguishable from -10 deg at the end. Rescale the yaw change (same timing) so the net
    rotation is the smallest one consistent with the tracked end pose up to that symmetry."""
    from scipy.spatial.transform import Rotation
    out = onp.asarray(traj, onp.float64).copy()
    rot = Rotation.from_quat(out[:, [1, 2, 3, 0]])
    yaw = onp.unwrap(rot.as_euler("ZYX")[:, 0])
    net = yaw[-1] - yaw[0]
    sym = onp.radians(sym_deg)
    target = net - sym * onp.round(net / sym)
    if abs(net) < 1e-6 or abs(target - net) < 1e-6:
        return out
    new = yaw[0] + (yaw - yaw[0]) * (target / net)
    fix = Rotation.from_euler("z", new - yaw) * rot
    out[:, :4] = fix.as_quat()[:, [3, 0, 1, 2]]
    return out


# ----------------------------------------------------------------------------- the problem
class DemoRepair:
    def __init__(self, demo_file: Path = DEMO_FILE, layout_file: Path = LAYOUT_FILE, phases: Phases = Phases(),
                 level_cube: bool = True):
        self.urdf = load_urdf(None, IK_URDF)
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.tcp_idx = list(self.kin.joint_names).index(TCP_TARGET)
        self.spheres = build_sphere_model(self.urdf, self.kin)
        self.cube_local, self.cube_r = cube_spheres()
        self.layout = json.loads(Path(layout_file).read_text())
        d = json.loads(Path(demo_file).read_text())
        self.demo_obj = onp.asarray(d["T_base_obj"], onp.float64)  # (T, wxyz_xyz)
        if level_cube:  # clean-up of the tracked cube (see level_poses / minimal_yaw)
            self.demo_obj = minimal_yaw(level_poses(self.demo_obj))
        self.grasp_obj_tcp = jaxlie.SE3(jnp.asarray(d["grasp_T_obj_tcp"], jnp.float32))
        self.ph = phases
        self.JointVar = RobotFactors.get_var_class(self.kin, default_val=jnp.asarray(UR5E_HOME, jnp.float32))
        self._jit_cache = {}

    # ------------------------------------------------------------------ nominal (generator) motion
    def fk_tcp(self, q):
        return jaxlie.SE3(self.kin.forward_kinematics(q)[..., self.tcp_idx, :])

    def grasp_candidates(self, obj0: jaxlie.SE3, home: jaxlie.SE3) -> list:
        """The cube is 90-degree symmetric about z, so the demo grasp holds at four yaws. Ordered by
        closeness to the start wrist; the generator always uses the first."""
        cands = []
        for k in range(4):
            g = jaxlie.SE3.from_rotation(jaxlie.SO3.from_z_radians(k * jnp.pi / 2)) @ self.grasp_obj_tcp
            err = float(jnp.linalg.norm(((obj0 @ g).rotation().inverse() @ home.rotation()).log()))
            cands.append((err, k, g))
        return [(k, g) for _, k, g in sorted(cands, key=lambda c: c[0])]

    def nominal(self, obj_traj: onp.ndarray | None = None, grasp_rank: int = 0) -> dict:
        """TCP targets, cube poses and phase labels for every logged step, as the generator builds them."""
        ph = self.ph
        obj_traj = self.demo_obj if obj_traj is None else obj_traj
        T = len(obj_traj)
        home = self.fk_tcp(jnp.asarray(UR5E_HOME, jnp.float32))
        objs = [jaxlie.SE3(jnp.asarray(o, jnp.float32)) for o in obj_traj]
        grasp_k, grasp = self.grasp_candidates(objs[0], home)[grasp_rank]
        up = jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, -ph.approach_height]))
        interp = lambda a, b, t: a @ jaxlie.SE3.exp((a.inverse() @ b).log() * t)  # noqa: E731
        ss = lambda t: t * t * (3 - 2 * t)  # noqa: E731

        tcp, obj, phase, pinned, held = [], [], [], [], []
        g0, gT = objs[0] @ grasp, objs[-1] @ grasp
        a = 1 - ph.descent_frac
        for i in range(ph.approach):
            s = i / max(ph.approach - 1, 1)
            tcp.append(interp(home, g0 @ up, ss(s / a)) if s < a else interp(g0 @ up, g0, ss((s - a) / ph.descent_frac)))
            obj.append(objs[0]); phase.append("approach"); pinned.append(s >= a); held.append(False)
        for i in range(ph.close):
            tcp.append(g0); obj.append(objs[0]); phase.append("close"); pinned.append(True); held.append(False)
        z0, zT = obj_traj[0, 6], obj_traj[-1, 6]
        for i in range(T):
            tcp.append(objs[i] @ grasp); obj.append(objs[i]); phase.append("carry"); held.append(True)
            z = obj_traj[i, 6]
            pinned.append(bool(z < z0 + ph.lift_tol or z < zT + ph.lift_tol))
        # Only the stretch between the first lift-off and the last set-down is free.
        c = onp.arange(ph.approach + ph.close, ph.approach + ph.close + T)
        free_c = c[~onp.array(pinned)[c]]
        if len(free_c):
            for k in range(free_c[0], free_c[-1] + 1):
                pinned[k] = False
        for i in range(ph.release):
            tcp.append(gT); obj.append(objs[-1]); phase.append("release"); pinned.append(True); held.append(True)
        for i in range(ph.retreat):
            s = i / max(ph.retreat - 1, 1)
            tcp.append(interp(gT, gT @ up, ss(s))); obj.append(objs[-1]); phase.append("retreat")
            pinned.append(True); held.append(False)
        stack = lambda xs: jaxlie.SE3(jnp.stack([x.wxyz_xyz for x in xs]))  # noqa: E731
        return dict(tcp=stack(tcp), obj=stack(obj), grasp=grasp, grasp_yaw_k=grasp_k, phase=onp.array(phase),
                    pinned=onp.array(pinned), held=onp.array(held))

    # ------------------------------------------------------------------ geometry
    def world_spheres(self, q, held_obj_from_tcp: jaxlie.SE3 | None = None):
        """Sphere centres (..., S(+8), 3) and radii for joint configs q (..., 6)."""
        Ts = self.kin.forward_kinematics(q)  # (..., J, 7)
        frames = jaxlie.SE3(Ts[..., self.spheres.joint_idx, :])
        c = frames @ jnp.asarray(self.spheres.local, jnp.float32)
        r = jnp.asarray(self.spheres.radius, jnp.float32)
        if held_obj_from_tcp is not None:
            cube = jaxlie.SE3(Ts[..., self.tcp_idx, :]) @ held_obj_from_tcp
            cc = jax.vmap(lambda p: cube @ p, out_axes=-2)(jnp.asarray(self.cube_local, jnp.float32))
            c = jnp.concatenate([c, cc], -2)
            r = jnp.concatenate([r, jnp.full((len(self.cube_local),), self.cube_r, jnp.float32)])
        return c, r

    def distances(self, q, world_arrays, held_mask, inv_grasp: jaxlie.SE3):
        """Per-step clearance: (d_obstacle (...,S'), d_env (...,S')) for spheres incl. the held cube.
        Cube spheres are pushed far away on steps where the cube is not held."""
        table_z, fix, cyl, box = world_arrays
        c, r = self.world_spheres(q, inv_grasp)
        n_cube = len(self.cube_local)
        far = jnp.where(held_mask[..., None, None], 0.0, 100.0)
        c = c.at[..., -n_cube:, :].add(far)
        d_obs = jnp.minimum(_sdf_cyl(c, cyl).min(-1), _sdf_box(c, box).min(-1)) - r
        grp = jnp.concatenate([jnp.asarray(self.spheres.group), jnp.full((n_cube,), 3)])
        d_tab = jnp.where(grp == 0, 10.0, c[..., 2] - table_z - r)
        d_fix = _sdf_cyl(c, fix).min(-1) - r
        d_fix = jnp.where(grp == 0, 10.0, d_fix)
        return d_obs, jnp.minimum(d_tab, d_fix)

    # ------------------------------------------------------------------ solvers
    def _pose_residual(self, q, target: jaxlie.SE3, w_pos, w_rot):
        res = (self.fk_tcp(q).inverse() @ target).log()
        return jnp.concatenate([res[:3] * w_pos, res[3:] * w_rot])

    def solve_track(self, targets: jaxlie.SE3, q_init: jnp.ndarray, smooth: float = 0.3) -> jnp.ndarray:
        """IK for a whole target sequence at once (pose factors + light smoothness + limits)."""
        key = ("track", targets.wxyz_xyz.shape[0])
        if key not in self._jit_cache:
            JV, kin, n = self.JointVar, self.kin, targets.wxyz_xyz.shape[0]

            @jax.jit
            def run(tgt, q0, smooth):
                pose = jaxls.Factor(lambda v, x, t: self._pose_residual(v[x], t, 100.0, 20.0),
                                    (JV(jnp.arange(n)), jaxlie.SE3(tgt)))
                sm = jaxls.Factor(lambda v, a, b: (v[a] - v[b]) * smooth, (JV(jnp.arange(1, n)), JV(jnp.arange(n - 1))))
                rest = jaxls.Factor(lambda v, x: (v[x] - jnp.asarray(UR5E_HOME, jnp.float32)) * 0.001, (JV(jnp.arange(n)),))
                lim = RobotFactors.limit_cost_factor(JV, jnp.arange(n), kin, jnp.full(6, 100.0))
                graph = jaxls.FactorGraph.make([pose, sm, rest, lim], [JV(jnp.arange(n))], use_onp=False)
                sol = graph.solve(jaxls.VarValues.make([JV(jnp.arange(n)).with_value(q0)]),
                                  trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
                                  termination=jaxls.TerminationConfig(max_iterations=200), verbose=False)
                return sol[JV]
            self._jit_cache[key] = run
        return self._jit_cache[key](targets.wxyz_xyz, q_init, smooth)

    def solve_repair(self, nom: dict, q_demo, q_init, world: World, *, margin=0.03, env_margin=0.012,
                     w=dict(track=1.0, vel=4.0, acc=8.0, coll=300.0, pin_pos=3000.0, pin_rot=300.0, up=4.0,
                            vlim=100.0), vmax=None):
        n = q_demo.shape[0]
        key = ("repair", n)
        if key not in self._jit_cache:
            JV, kin = self.JointVar, self.kin

            @jax.jit
            def run(tgt, qd, q0, pinned, held, free_carry, wa, inv_g, w, margin, env_margin, vmax):
                ids = jnp.arange(n)

                def track(v, x, qd_t, pin):
                    return (v[x] - qd_t) * w["track"] * (1.0 - pin)

                def pin_pose(v, x, t, pin):
                    return self._pose_residual(v[x], t, w["pin_pos"], w["pin_rot"]) * pin

                def keep_upright(v, x, t, fc):
                    # Tool orientation during the free carry stays close to the demo's (cube stays level).
                    res = (self.fk_tcp(v[x]).rotation().inverse() @ t.rotation()).log()
                    return res * w["up"] * fc

                def coll(v, x, h, pin):
                    d_obs, d_env = self.distances(v[x], wa, h, inv_g)
                    r_obs = jnp.maximum(0.0, margin - d_obs)
                    r_env = jnp.maximum(0.0, env_margin - d_env) * (1.0 - pin)
                    return jnp.concatenate([r_obs, r_env]) * w["coll"]

                def vel(v, a, b):
                    dq = v[a] - v[b]
                    return jnp.concatenate([dq * w["vel"], jnp.maximum(0.0, jnp.abs(dq) - vmax) * w["vlim"]])

                def acc(v, a, b, c):
                    return (v[a] - 2 * v[b] + v[c]) * w["acc"]

                factors = [
                    jaxls.Factor(track, (JV(ids), qd, pinned)),
                    jaxls.Factor(pin_pose, (JV(ids), jaxlie.SE3(tgt), pinned)),
                    jaxls.Factor(keep_upright, (JV(ids), jaxlie.SE3(tgt), free_carry)),
                    jaxls.Factor(coll, (JV(ids), held, pinned)),
                    jaxls.Factor(vel, (JV(ids[1:]), JV(ids[:-1]))),
                    jaxls.Factor(acc, (JV(ids[2:]), JV(ids[1:-1]), JV(ids[:-2]))),
                    RobotFactors.limit_cost_factor(JV, ids, kin, jnp.full(6, 100.0)),
                ]
                graph = jaxls.FactorGraph.make(factors, [JV(ids)], use_onp=False)
                sol = graph.solve(jaxls.VarValues.make([JV(ids).with_value(q0)]),
                                  trust_region=jaxls.TrustRegionConfig(lambda_initial=10.0),
                                  termination=jaxls.TerminationConfig(max_iterations=300), verbose=False)
                return sol[JV]
            self._jit_cache[key] = run
        free_carry = jnp.asarray((~nom["pinned"]) & (nom["phase"] == "carry"), jnp.float32)
        vmax = self.velocity_limit(q_demo) if vmax is None else vmax
        w = {k: jnp.float32(v) for k, v in w.items()}
        return self._jit_cache[key](nom["tcp"].wxyz_xyz, q_demo, q_init, jnp.asarray(nom["pinned"], jnp.float32),
                                    jnp.asarray(nom["held"]), free_carry, world.arrays(), nom["grasp"].inverse(),
                                    w, jnp.float32(margin), jnp.float32(env_margin), jnp.asarray(vmax, jnp.float32))

    # ------------------------------------------------------------------ checking
    def check(self, q, nom: dict, world: World, *, substeps: int = 4, obs_clear: float = 0.005,
              vmax=None) -> dict:
        """Zero-margin re-check on a denser time grid. Returns per-step clearances and failure reasons."""
        q = jnp.asarray(q)
        s = jnp.linspace(0, 1, substeps, endpoint=False)
        qd = (q[:-1, None] * (1 - s[:, None]) + q[1:, None] * s[:, None]).reshape(-1, 6)
        qd = jnp.concatenate([qd, q[-1:]])
        step = onp.minimum(onp.arange(len(qd)) // substeps, len(q) - 1)
        held = jnp.asarray(nom["held"][step])
        d_obs, d_env = self.distances(qd, World.arrays(world), held, nom["grasp"].inverse())
        d_obs, d_env = onp.asarray(d_obs.min(-1)), onp.asarray(d_env.min(-1))
        free = ~nom["pinned"][step]
        tcp = self.fk_tcp(q)
        perr = onp.asarray(jnp.linalg.norm(tcp.translation() - nom["tcp"].translation(), axis=-1))
        rerr = onp.asarray(jnp.linalg.norm((tcp.rotation().inverse() @ nom["tcp"].rotation()).log(), axis=-1))
        vmax = onp.asarray(self.velocity_limit(self._q_demo) if vmax is None else vmax)
        dq_j = onp.abs(onp.diff(onp.asarray(q), axis=0))
        dq = dq_j.max(-1)
        lo, hi = onp.asarray(self.kin.limits_lower), onp.asarray(self.kin.limits_upper)
        reasons = {}
        if (d_obs < obs_clear).any():
            reasons["obstacle_collision"] = int((d_obs < obs_clear).sum())
        if (d_env[free] < 0).any():
            reasons["env_collision"] = int((d_env[free] < 0).sum())
        pin = nom["pinned"]
        pin_bad = pin & ((perr > 0.003) | (rerr > 0.03))
        if pin_bad.any():
            reasons["pinned_pose_error"] = dict(steps=onp.where(pin_bad)[0].tolist(),
                                                max_mm=round(float(perr[pin].max() * 1e3), 2),
                                                max_rad=round(float(rerr[pin].max()), 4))
        v_bad = (dq_j > vmax + 1e-4).any(-1)
        if v_bad.any():
            reasons["velocity_limit"] = dict(steps=onp.where(v_bad)[0].tolist())
        qa = onp.asarray(q)
        if ((qa < lo) | (qa > hi)).any():
            reasons["joint_limit"] = 1
        # Colliding spans in step indices, for the report.
        bad = onp.zeros(len(q), bool)
        bad[step[d_obs < obs_clear]] = True
        spans, i = [], 0
        while i < len(bad):
            if bad[i]:
                j = i
                while j + 1 < len(bad) and bad[j + 1]:
                    j += 1
                spans.append(dict(start=int(i), end=int(j), phase=str(nom["phase"][i]), pinned=bool(nom["pinned"][i:j + 1].any())))
                i = j + 1
            else:
                i += 1
        per_step = onp.full(len(q), onp.inf)
        onp.minimum.at(per_step, step, d_obs)
        return dict(ok=not reasons, reasons=reasons, min_obstacle_clearance=float(d_obs.min()),
                    min_env_clearance_free=float(d_env[free].min()) if free.any() else None,
                    obstacle_clearance_per_step=per_step, collision_spans=spans,
                    max_pinned_pos_err_mm=float(perr[pin].max() * 1e3), max_step_rad=float(dq.max()))

    @staticmethod
    def velocity_limit(q_demo, slack: float = 1.25, floor: float = 0.05):
        """Per-joint cap on the change per logged step: the demo's own peak for that joint (the
        replay keeps the demo's timing), with some slack, never below `floor` rad/step."""
        return onp.maximum(onp.abs(onp.diff(onp.asarray(q_demo), axis=0)).max(0) * slack, floor)

    # ------------------------------------------------------------------ seeds and the full pipeline
    def bend(self, nom: dict, offset: onp.ndarray) -> jaxlie.SE3:
        """Demo TCP targets with each free stretch pushed by `offset` (m) under a sin^2 bump."""
        tcp = onp.asarray(nom["tcp"].wxyz_xyz).copy()
        free = ~nom["pinned"]
        i = 0
        while i < len(free):
            if free[i]:
                j = i
                while j + 1 < len(free) and free[j + 1]:
                    j += 1
                k = onp.arange(i, j + 1)
                # The transit stretch starts at the start pose, which must stay put: half bump there.
                s = (k - i + 1) / (j - i + 2)
                tcp[k, 4:] += onp.sin(onp.pi * s)[:, None] ** 2 * offset
                i = j + 1
            else:
                i += 1
        return jaxlie.SE3(jnp.asarray(tcp, jnp.float32))

    def run(self, world: World, obj_traj: onp.ndarray | None = None, *, seeds_m: float = 0.14,
            try_grasp_symmetries: bool = True, repair: bool = True) -> dict:
        """Check the generator's motion; if an edit invalidates it, try the cube's other grasp yaws
        (first as generated, then repaired) before rejecting. `before` is always the generator's
        own motion (first grasp candidate)."""
        t0 = time.time()
        attempts = []
        for rank in range(4 if try_grasp_symmetries else 1):
            r = self._run_one(world, obj_traj, rank, seeds_m, repair)
            attempts.append(r)
            if r["status"] in ("valid_as_generated", "repaired"):
                break
        ok = [r for r in attempts if r["status"] in ("valid_as_generated", "repaired")]
        out = dict(ok[0] if ok else attempts[0])
        base = attempts[0]
        out.update(q_baseline=base["q_demo"], nominal_baseline=base["nominal"], before=base["before"],
                   baseline_status=base["status"],
                   grasp_rank=next(i for i, r in enumerate(attempts) if r is ok[0]) if ok else 0,
                   attempts=[dict(rank=i, yaw_k=r["nominal"]["grasp_yaw_k"], status=r["status"])
                             for i, r in enumerate(attempts)])
        if ok and ok[0] is not base:
            out["status"] = ok[0]["status"] + "_other_grasp_yaw"
        out["time_s"] = time.time() - t0
        return out

    def _run_one(self, world: World, obj_traj, grasp_rank: int, seeds_m: float, repair: bool = True) -> dict:
        t0 = time.time()
        nom = self.nominal(obj_traj, grasp_rank)
        n = nom["tcp"].wxyz_xyz.shape[0]
        q_home = jnp.broadcast_to(jnp.asarray(UR5E_HOME, jnp.float32), (n, 6))
        q_demo = self.solve_track(nom["tcp"], q_home)
        self._q_demo = q_demo
        before = self.check(q_demo, nom, world)
        out = dict(nominal=nom, q_demo=onp.asarray(q_demo), before=before, world=world)
        if before["ok"]:
            out.update(status="valid_as_generated", q=onp.asarray(q_demo), after=before, seeds=[])
            out["time_s"] = time.time() - t0
            return out
        if not repair:
            out.update(status="reject_collision", q=None, after=None, seeds=[])
            out["time_s"] = time.time() - t0
            return out
        if any(s["pinned"] for s in before["collision_spans"]):
            out.update(status="reject_obstacle_in_task_segment", q=None, after=None, seeds=[])
            out["time_s"] = time.time() - t0
            return out
        p0, p1 = onp.asarray(nom["obj"].translation())[[0, -1]]
        along = (p1 - p0)[:2] / (onp.linalg.norm((p1 - p0)[:2]) + 1e-9)
        side = onp.array([-along[1], along[0], 0.0])
        offsets = {"demo": onp.zeros(3), "over": onp.array([0, 0, seeds_m]), "left": side * seeds_m,
                   "right": -side * seeds_m, "over_left": (side + [0, 0, 1]) * seeds_m * 0.7,
                   "over_right": (-side + [0, 0, 1]) * seeds_m * 0.7}
        seeds = []
        for name, off in offsets.items():
            q_init = q_demo if name == "demo" else self.solve_track(self.bend(nom, off), q_demo)
            q = self.solve_repair(nom, q_demo, q_init, world)
            chk = self.check(q, nom, world)
            qa = onp.asarray(q)
            dev = float(onp.sqrt(((qa - onp.asarray(q_demo)) ** 2).sum(-1)).mean())
            acc = float((onp.diff(qa, 2, axis=0) ** 2).sum())
            seeds.append(dict(name=name, q=qa, check=chk, mean_joint_dev=dev, sum_sq_acc=acc,
                              score=dev + 5.0 * acc))
        ok = [s for s in seeds if s["check"]["ok"]]
        best = min(ok or seeds, key=lambda s: (not s["check"]["ok"], s["score"]))
        out.update(status="repaired" if ok else "repair_failed", q=best["q"] if ok else None,
                   after=best["check"], seed=best["name"], seeds=seeds, q_best=best["q"])
        out["time_s"] = time.time() - t0
        return out

    def objects_from_q(self, q, nom: dict) -> onp.ndarray:
        """Cube poses (T, wxyz_xyz): follow the gripper while held, else the nominal pose."""
        held_pose = (self.fk_tcp(jnp.asarray(q)) @ nom["grasp"].inverse()).wxyz_xyz
        return onp.where(nom["held"][:, None], onp.asarray(held_pose), onp.asarray(nom["obj"].wxyz_xyz))


def path_metrics(rep: DemoRepair, q_demo, q, nom) -> dict:
    a, b = onp.asarray(rep.fk_tcp(jnp.asarray(q_demo)).translation()), onp.asarray(rep.fk_tcp(jnp.asarray(q)).translation())
    L = lambda p: float(onp.linalg.norm(onp.diff(p, axis=0), axis=-1).sum())  # noqa: E731
    return dict(max_tcp_dev_cm=float(onp.linalg.norm(a - b, axis=-1).max() * 100),
                tcp_path_len_demo_m=L(a), tcp_path_len_repair_m=L(b),
                max_joint_dev_rad=float(onp.abs(onp.asarray(q) - onp.asarray(q_demo)).max()))


def save(out: dict, rep: DemoRepair, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    nom = out["nominal"]
    arrays = dict(q_baseline=out["q_baseline"], q_demo=out["q_demo"], tcp_target=onp.asarray(nom["tcp"].wxyz_xyz), phase=nom["phase"],
                  pinned=nom["pinned"], held=nom["held"], obj_demo=onp.asarray(nom["obj"].wxyz_xyz),
                  grasp_T_obj_tcp=onp.asarray(nom["grasp"].wxyz_xyz),
                  clear_before=out["before"]["obstacle_clearance_per_step"])
    if out["q"] is not None:
        arrays.update(q_repair=out["q"], obj_repair=rep.objects_from_q(out["q"], nom),
                      clear_after=out["after"]["obstacle_clearance_per_step"])
    if out.get("q_best") is not None and out["q"] is None:
        arrays.update(q_best_failed=out["q_best"])  # closest failed candidate, for debugging
    onp.savez(path / "repair.npz", **arrays)
    strip = lambda c: {k: v for k, v in c.items() if k != "obstacle_clearance_per_step"} if c else None  # noqa: E731
    report = dict(status=out["status"], baseline_status=out["baseline_status"], seed=out.get("seed"),
                  grasp_yaw_k=int(nom["grasp_yaw_k"]), attempts=out["attempts"], time_s=round(out["time_s"], 2),
                  world=out["world"].to_json(), before=strip(out["before"]), after=strip(out["after"]),
                  seeds=[dict(name=s["name"], ok=s["check"]["ok"], reasons=s["check"]["reasons"],
                              min_obstacle_clearance=s["check"]["min_obstacle_clearance"],
                              mean_joint_dev=s["mean_joint_dev"], sum_sq_acc=s["sum_sq_acc"]) for s in out["seeds"]])
    if out["q"] is not None:
        report["path"] = path_metrics(rep, out["q_baseline"], out["q"], nom)
    (path / "report.json").write_text(json.dumps(report, indent=2))
    return report


def plot(out: dict, rep: DemoRepair, path: Path):
    """Top and side views of the TCP path before (red where it collides) and after repair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nom, world = out["nominal"], out["world"]
    a = onp.asarray(rep.fk_tcp(jnp.asarray(out["q_baseline"])).translation())
    hit = out["before"]["obstacle_clearance_per_step"] < 0.005
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    for k, (i, j, lab) in enumerate([(0, 1, "top view (x, y)"), (0, 2, "side view (x, z)")]):
        A = ax[k]
        A.plot(a[:, i], a[:, j], color="#999", lw=1.5, label="generated (demo-warped)")
        A.scatter(a[hit, i], a[hit, j], color="#d62728", s=10, zorder=3, label="in collision")
        if out["q"] is not None:
            b = onp.asarray(rep.fk_tcp(jnp.asarray(out["q"])).translation())
            A.plot(b[:, i], b[:, j], color="#2ca02c", lw=2, label="repaired")
        for f in world.fixtures:
            if j == 1:
                A.add_patch(plt.Circle(f[:2], f[4], color="#f28e2b", alpha=0.4))
            else:
                A.add_patch(plt.Rectangle((f[0] - f[4], f[2]), 2 * f[4], f[3] - f[2], color="#f28e2b", alpha=0.4))
        for c in world.obs_cyl:
            if j == 1:
                A.add_patch(plt.Circle(c[:2], c[4], color="#4e79a7", alpha=0.6))
            else:
                A.add_patch(plt.Rectangle((c[0] - c[4], c[2]), 2 * c[4], c[3] - c[2], color="#4e79a7", alpha=0.6))
        A.set_title(lab); A.set_aspect("equal"); A.grid(alpha=0.3)
    ax[0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{out['status']}" + (f" (seed: {out.get('seed')})" if out.get("seed") else ""))
    fig.tight_layout()
    fig.savefig(path / "paths.png", dpi=130)
    plt.close(fig)


def main(
    output: Path = Path("outputs/ur_cube/repair/default"),
    obstacle: tuple[float, float, float, float] | None = None,
    obstacle_at: float = 0.5,
    obstacle_radius: float = 0.035,
    obstacle_height: float = 0.26,
    lateral: float = 0.0,
    level_cube: bool = True,
) -> None:
    """Insert a vertical cylinder obstacle (x y radius height, or a fraction along the demo's cube
    path plus a lateral offset), check the generated demo against it and repair it."""
    rep = DemoRepair(level_cube=level_cube)
    world = World.from_layout(rep.layout)
    if obstacle is not None:
        world = world.with_cylinder(*obstacle)
    else:
        p = rep.demo_obj[:, 4:6]
        p0, p1 = p[0], p[-1]
        d = (p1 - p0) / onp.linalg.norm(p1 - p0)
        xy = p0 + obstacle_at * (p1 - p0) + lateral * onp.array([-d[1], d[0]])
        world = world.with_cylinder(float(xy[0]), float(xy[1]), obstacle_radius, obstacle_height)
    out = rep.run(world)
    report = save(out, rep, output)
    plot(out, rep, output)
    print(json.dumps({k: report[k] for k in ("status", "baseline_status", "seed", "grasp_yaw_k", "attempts", "time_s")}))
    print("before:", report["before"]["reasons"], "spans", report["before"]["collision_spans"])
    if report.get("after"):
        print("after:", report["after"]["reasons"], "min clearance", round(report["after"]["min_obstacle_clearance"], 4))
    for s in report["seeds"]:
        print(f"  seed {s['name']:10s} ok={s['ok']} clear={s['min_obstacle_clearance']:.3f} dev={s['mean_joint_dev']:.3f} "
              f"acc={s['sum_sq_acc']:.4f} {s['reasons']}")
    if "path" in report:
        print("path:", report["path"])
    print("wrote", output)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
