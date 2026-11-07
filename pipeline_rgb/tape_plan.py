"""Robot motion for the tape-hanging task from an object trajectory (the R2R2R generator step).

Grasp: the 2F-85 pinches the roll's band (tape + core, r_in..R) from its top face. TCP z (approach)
runs down the roll's axis, TCP x (the fingers' closing direction) is radial, and the TCP sits
`tcp_above` over the face so the fingertips stop just above the bench. The roll is symmetric about
its axis, so the grasp can sit at any azimuth on the rim: candidates every `azimuth_step` degrees
are planned and checked, and the cheapest collision-free one is kept.

Phases (as the cube generator): approach from the home pose to a pre-grasp `approach_height` back
along the tool axis, straight descent, close, carry (TCP = object pose x grasp), release, straight
retreat back along the tool axis (on the branch this slides the inner finger out of the core).
Joints come from one whole-sequence IK solve (jaxls, as retarget_embodiment.Arm). A plan is
rejected for IK error, joint limits, fingertips below the bench, or arm/gripper spheres
(demo_repair's sphere model) within `margin` of the mug tree.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jaxlie
import numpy as onp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ur_cube"))
from demo_repair import FINGERTIP_BELOW_TCP, IK_URDF, UR5E_HOME, build_sphere_model  # noqa: E402
from jaxmp.extras.urdf_loader import load_urdf  # noqa: E402
from retarget_embodiment import ROBOTS, Arm  # noqa: E402

REPO = HERE.parents[1]


@dataclass
class TapePhases:
    approach: int = 30
    close: int = 8
    release: int = 8
    retreat: int = 14
    approach_height: float = 0.10
    descent_frac: float = 0.35


def capsules_from_scene(scene: dict) -> tuple[onp.ndarray, onp.ndarray]:
    """Fixture as capsules (K, 7: a xyz, b xyz, r) and vertical cylinders (M, 5: x y z0 z1 r).
    Mug tree: post + branches as capsules, base disc as a cylinder. Bolt stand: rod as a capsule;
    spacers, base nut and block as cylinders (the block's circumscribed one)."""
    if "bolt_stand" in scene:
        b = scene["bolt_stand"]
        xy = b["xy"]
        # above the placed nut's bottom face the rod is inside the nut's hole, between the fingers
        caps = [[*xy, b["rod"]["z0"], *xy, b["rod"].get("collision_z1", b["rod"]["z1"]), b["rod"]["radius"]]]
        cyl = [[*xy, b[k]["z0"], b[k]["z1"], b[k]["radius"]] for k in ("spacers", "base")]
        cyl.append([*xy, b["block"]["z0"], b["block"]["z1"], b["block"]["size"][0] / 2 * 2 ** 0.5])
        return onp.array(caps), onp.array(cyl)
    t = scene["mug_tree"]
    p = t["post"]
    caps = [[*p["xy"], p["z0"], *p["xy"], p["z1"], p["radius"]]]
    for b in t["branches"].values():
        caps.append([*b["root"], *b["tip"], b["radius"]])
    base = t["base"]
    cyl = [[*base["xy"], scene["table_z"], scene["table_z"] + base["thickness"], base["radius"]]]
    return onp.array(caps), onp.array(cyl)


def sdf_capsules(p, caps):
    a, b, r = caps[:, :3], caps[:, 3:6], caps[:, 6]
    ab = b - a
    t = onp.clip(((p[..., None, :] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)
    q = a + t[..., None] * ab
    return onp.linalg.norm(p[..., None, :] - q, axis=-1) - r


def sdf_vcyl(p, c):
    dxy = onp.linalg.norm(p[..., None, :2] - c[:, :2], axis=-1) - c[:, 4]
    dz = onp.maximum(c[:, 2] - p[..., None, 2], p[..., None, 2] - c[:, 3])
    q = onp.stack([dxy, dz], -1)
    return onp.linalg.norm(onp.maximum(q, 0), axis=-1) + onp.minimum(q.max(-1), 0)


def rotate_about_post(scene: dict, psi: float) -> dict:
    """The mug tree turned by psi (rad) about its post axis (the tree's own symmetry axis)."""
    import copy
    s = copy.deepcopy(scene)
    if "mug_tree" not in s:
        return s
    t = s["mug_tree"]
    c = onp.array(t["post"]["xy"])
    R = onp.array([[onp.cos(psi), -onp.sin(psi)], [onp.sin(psi), onp.cos(psi)]])
    for b in t["branches"].values():
        for k in ("root", "tip"):
            p = onp.array(b[k])
            p[:2] = c + R @ (p[:2] - c)
            b[k] = p.tolist()
        b["azimuth_deg"] = float((b["azimuth_deg"] + onp.degrees(psi) + 180) % 360 - 180)
    t["yaw_deg"] = float(onp.degrees(psi))
    return s


def rotate_traj_about_post(traj: onp.ndarray, psi: float, post_xy) -> onp.ndarray:
    """Object poses (T, wxyz_xyz) rotated with the tree by psi about the vertical post axis."""
    from scipy.spatial.transform import Rotation
    out = onp.asarray(traj, onp.float64).copy()
    Rz = Rotation.from_euler("z", psi)
    out[:, :4] = (Rz * Rotation.from_quat(out[:, [1, 2, 3, 0]])).as_quat()[:, [3, 0, 1, 2]]
    c = onp.r_[post_xy, 0.0]
    out[:, 4:] = Rz.apply(out[:, 4:] - c) + c
    return out


class TapePlanner:
    def __init__(self, scene_file: Path = REPO / "outputs/ur_tape/scene.json",
                 dims_file: Path = REPO / "outputs/ur_tape/tape_dims.json", phases: TapePhases = TapePhases(),
                 tcp_above: float = 0.010, azimuth_step: float = 30.0, margin: float = 0.005,
                 tree_yaw_deg: float = 0.0, grasp_style: str = "band"):
        """grasp_style: "band" pinches the roll's band from its top face (tape); "outer" pinches a
        small ring across its outside from above (nut)."""
        self.grasp_style = grasp_style
        self.scene_measured = json.loads(Path(scene_file).read_text())
        self.scene = rotate_about_post(self.scene_measured, onp.radians(tree_yaw_deg))
        self.dims = json.loads(Path(dims_file).read_text())
        self.ph, self.tcp_above, self.margin = phases, tcp_above, margin
        self.arm = Arm(ROBOTS["ur5e"])
        self.spheres = build_sphere_model(load_urdf(None, IK_URDF), self.arm.kin)
        self.caps, self.base_cyl = capsules_from_scene(self.scene)
        self.table_z = self.scene["table_z"]
        self.azimuths = onp.radians(onp.arange(0, 360, azimuth_step))
        self.home_tcp = self.arm.fk_world(jnp.asarray(UR5E_HOME, jnp.float32))

    def set_tree_yaw(self, deg: float):
        """Turn the mug tree about its post (collision model); see rotate_about_post."""
        self.scene = rotate_about_post(self.scene_measured, onp.radians(deg))
        self.caps, self.base_cyl = capsules_from_scene(self.scene)

    @property
    def post_xy(self):
        s = self.scene_measured
        return onp.asarray(s["mug_tree"]["post"]["xy"] if "mug_tree" in s else s["bolt_stand"]["xy"])

    def grasp(self, az: float) -> jaxlie.SE3:
        """TCP pose in the roll frame: on the band's mid-radius at azimuth az, above the +z face,
        approach (TCP z) along -axis, fingers closing radially (TCP x)."""
        d = self.dims
        r_mid = (d["R"] + d["r_in"]) / 2 if self.grasp_style == "band" else 0.0
        radial = onp.array([onp.cos(az), onp.sin(az), 0.0])
        z = onp.array([0.0, 0.0, -1.0])
        y = onp.cross(z, radial)
        Rm = onp.stack([radial, y, z], 1)
        p = r_mid * radial + onp.array([0, 0, d["w"] / 2 + self.tcp_above])
        return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_matrix(jnp.asarray(Rm, jnp.float32)),
                                                         jnp.asarray(p, jnp.float32))

    def nominal(self, obj_traj: onp.ndarray, g: jaxlie.SE3) -> dict:
        ph = self.ph
        objs = [jaxlie.SE3(jnp.asarray(o, jnp.float32)) for o in obj_traj]
        up = jaxlie.SE3.from_translation(jnp.array([0.0, 0.0, -ph.approach_height]))
        interp = lambda a, b, t: a @ jaxlie.SE3.exp((a.inverse() @ b).log() * t)  # noqa: E731
        ss = lambda t: t * t * (3 - 2 * t)  # noqa: E731
        g0, gT = objs[0] @ g, objs[-1] @ g
        tcp, obj, phase = [], [], []
        a = 1 - ph.descent_frac
        for i in range(ph.approach):
            s = i / max(ph.approach - 1, 1)
            tcp.append(interp(self.home_tcp, g0 @ up, ss(s / a)) if s < a else interp(g0 @ up, g0, ss((s - a) / ph.descent_frac)))
            obj.append(objs[0]); phase.append("approach")
        for _ in range(ph.close):
            tcp.append(g0); obj.append(objs[0]); phase.append("close")
        for o in objs:
            tcp.append(o @ g); obj.append(o); phase.append("carry")
        for _ in range(ph.release):
            tcp.append(gT); obj.append(objs[-1]); phase.append("release")
        for i in range(ph.retreat):
            s = i / max(ph.retreat - 1, 1)
            tcp.append(interp(gT, gT @ up, ss(s))); obj.append(objs[-1]); phase.append("retreat")
        st = lambda xs: onp.stack([onp.asarray(x.wxyz_xyz) for x in xs])  # noqa: E731
        return dict(tcp=st(tcp), obj=st(obj), phase=onp.array(phase))

    def sphere_centres(self, q):
        Ts = self.arm.kin.forward_kinematics(jnp.asarray(q))
        fr = jaxlie.SE3(Ts[..., self.spheres.joint_idx, :])
        return onp.asarray(self.arm.base @ (fr @ jnp.asarray(self.spheres.local, jnp.float32)))

    def check(self, q, nom) -> dict:
        tcp = self.arm.fk_world(jnp.asarray(q))
        tgt = jaxlie.SE3(jnp.asarray(nom["tcp"], jnp.float32))
        err = onp.asarray((tcp.inverse() @ tgt).log())
        pos_err, rot_err = onp.linalg.norm(err[:, :3], axis=1), onp.linalg.norm(err[:, 3:], axis=1)
        lo, hi = onp.asarray(self.arm.kin.limits_lower), onp.asarray(self.arm.kin.limits_upper)
        limits_ok = bool(((q >= lo) & (q <= hi)).all())
        tips = onp.asarray((tcp @ jaxlie.SE3.from_translation(jnp.array([0, 0, FINGERTIP_BELOW_TCP]))).translation())
        tip_clear = float((tips[:, 2] - self.table_z).min())
        c = self.sphere_centres(q)
        r = self.spheres.radius
        d_tree = onp.minimum(sdf_capsules(c, self.caps).min(-1), sdf_vcyl(c, self.base_cyl).min(-1)) - r
        grp = onp.asarray(self.spheres.group)
        d_tab = onp.where(grp[None] == 1, c[..., 2] - self.table_z - r, 10.0)  # arm links vs bench
        tree_clear, table_clear = float(d_tree.min()), float(d_tab.min())
        status = "ok"
        bad = onp.where((pos_err > 0.005) | (rot_err > 0.05))[0]
        worst_phase = str(nom["phase"][bad[onp.argmax(pos_err[bad])]]) if len(bad) else None
        if pos_err.max() > 0.005 or rot_err.max() > 0.05:
            status = "ik_error"
        elif not limits_ok:
            status = "joint_limit"
        elif tip_clear < 0.002:
            status = "finger_clearance"
        elif tree_clear < self.margin:
            status = "fixture_collision"
        elif table_clear < 0.0:
            status = "table_collision"
        return dict(status=status, ok=status == "ok", max_pos_err_mm=float(pos_err.max() * 1000),
                    max_rot_err_deg=float(onp.degrees(rot_err.max())), tree_clearance=tree_clear,
                    table_clearance=table_clear, fingertip_clearance=tip_clear, ik_worst_phase=worst_phase,
                    tree_worst_phase=str(nom["phase"][onp.argmin(d_tree.min(-1))]),
                    tree_clearance_per_step=d_tree.min(-1))

    def plan(self, obj_traj: onp.ndarray) -> dict:
        """Best grasp azimuth for this object trajectory: collision-free first, then shortest joint path."""
        best = None
        tries = []
        for az in self.azimuths:
            g = self.grasp(float(az))
            nom = self.nominal(obj_traj, g)
            q = onp.asarray(self.arm.solve(jaxlie.SE3(jnp.asarray(nom["tcp"], jnp.float32))))
            chk = self.check(q, nom)
            path = float(onp.abs(onp.diff(q, axis=0)).sum())
            tries.append(dict(azimuth_deg=round(float(onp.degrees(az)), 1), status=chk["status"],
                              ik_err_mm=round(chk["max_pos_err_mm"], 1), ik_phase=chk["ik_worst_phase"],
                              tree_phase=chk["tree_worst_phase"],
                              tree_clearance=round(chk["tree_clearance"], 4), joint_path=round(path, 3)))
            key = (not chk["ok"], path)
            if best is None or key < best[0]:
                best = (key, dict(q=q, obj=nom["obj"], tcp=nom["tcp"], phase=nom["phase"], grasp=onp.asarray(g.wxyz_xyz),
                                  azimuth_deg=float(onp.degrees(az)), **{k: v for k, v in chk.items()}))
        out = best[1]
        out["tries"] = tries
        return out
