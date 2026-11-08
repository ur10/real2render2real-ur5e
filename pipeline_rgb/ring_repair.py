"""Scene edits + demo repair for the RGB-only tasks (tape roll, nut): the cube's DemoRepair
(scripts/ur_cube/demo_repair.py) with the task-specific parts swapped in.

  * Grasp candidates: the ring planner's grasp positions around the rim (tape_plan.py), best first
    (valid in the unedited scene, then shortest joint path); up to 4 are tried before rejecting.
  * Pinned segments: the straight descent, close, release and retreat as for the cube, plus the
    carry while the object is near its start (within `lift_tol` of its rest height: the roll tipping
    up on its edge, the nut lifting off) and near its goal (within `place_pin_r` of the final pose:
    threading onto the branch, the descent onto the rod). Only the transit and the middle of the
    carry are re-optimized.
  * Held object: a ring of spheres around the roll's band / the nut.
  * Environment: the fixture's vertical parts as cylinders in the optimization (mug-tree post and
    base; bolt-stand rod, spacers, base nut, block). Every result is re-checked with the ring
    planner's full check, which includes the mug tree's angled branches, the fingertips over the
    bench and joint limits.

    python scripts/ur_tape/ring_repair.py --task tape --xy 0.03 -0.80     # one bottle placement
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax.numpy as jnp  # noqa: E402
import jaxlie  # noqa: E402
import numpy as onp  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ur_cube"))
from demo_repair import IK_URDF, TCP_TARGET, UR5E_HOME, DemoRepair, Phases, World, build_sphere_model  # noqa: E402
from jaxmp import JaxKinTree  # noqa: E402
from jaxmp.extras.urdf_loader import load_urdf  # noqa: E402
from jaxmp.robot_factors import RobotFactors  # noqa: E402
from tape_plan import TapePlanner  # noqa: E402

REPO = HERE.parents[1]
TASKS = {
    "tape": dict(traj="outputs/ur_tape/traj_ep1_hang_clean.json", scene="outputs/ur_tape/scene.json",
                 dims="outputs/ur_tape/tape_dims.json", grasp_style="band", tcp_above=0.010, place_pin_r=0.08,
                 tree_yaw=250.0, start_offset=(0.0, 0.0)),
    "nut": dict(traj="outputs/ur_nut/traj_ep4_place_clean.json", scene="outputs/ur_nut/scene.json",
                dims="outputs/ur_nut/nut_dims.json", grasp_style="outer", tcp_above=0.017, place_pin_r=0.05,
                tree_yaw=0.0, start_offset=(0.08, 0.12)),
}


def ring_spheres(dims: dict):
    """Held ring as spheres on its mid-radius (object frame), radius covering band and width."""
    r_mid = (dims["R"] + dims["r_in"]) / 2
    r = max((dims["R"] - dims["r_in"]) / 2, dims["w"] / 2) + 0.002
    n = max(int(onp.ceil(2 * onp.pi * r_mid / (1.2 * r))), 6)
    t = onp.linspace(0, 2 * onp.pi, n, endpoint=False)
    return onp.stack([r_mid * onp.cos(t), r_mid * onp.sin(t), onp.zeros(n)], 1), float(r)


def fixture_cylinders(scene: dict) -> onp.ndarray:
    """Vertical parts of the fixture as (K, 5) cylinders x y z0 z1 r (World.fixtures format)."""
    if "bolt_stand" in scene:
        b = scene["bolt_stand"]
        xy = b["xy"]
        rows = [[*xy, b[k]["z0"], b[k]["z1"], b[k]["radius"]] for k in ("spacers", "base")]
        rows.append([*xy, b["rod"]["z0"], b["rod"].get("collision_z1", b["rod"]["z1"]), b["rod"]["radius"]])
        rows.append([*xy, b["block"]["z0"], b["block"]["z1"], b["block"]["size"][0] / 2 * 2 ** 0.5])
        return onp.array(rows)
    t = scene["mug_tree"]
    p, base = t["post"], t["base"]
    return onp.array([[*p["xy"], p["z0"], p["z1"], p["radius"]],
                      [*base["xy"], scene["table_z"], scene["table_z"] + base["thickness"], base["radius"]]])


class RingRepair(DemoRepair):
    def __init__(self, task: str = "tape", phases: Phases = Phases(), tree_yaw: float | None = None):
        cfg = TASKS[task]
        self.task, self.cfg = task, cfg
        self.planner = TapePlanner(scene_file=REPO / cfg["scene"], dims_file=REPO / cfg["dims"],
                                   grasp_style=cfg["grasp_style"], tcp_above=cfg["tcp_above"],
                                   tree_yaw_deg=cfg["tree_yaw"] if tree_yaw is None else tree_yaw)
        self.urdf = load_urdf(None, IK_URDF)
        self.kin = JaxKinTree.from_urdf(self.urdf)
        self.tcp_idx = list(self.kin.joint_names).index(TCP_TARGET)
        self.spheres = build_sphere_model(self.urdf, self.kin)
        self.cube_local, self.cube_r = ring_spheres(self.planner.dims)  # "cube" = the held object here
        d = json.loads((REPO / cfg["traj"]).read_text())
        self.demo_obj = onp.asarray(d["T_base_obj"], onp.float64)
        self.ph = phases
        self.JointVar = RobotFactors.get_var_class(self.kin, default_val=jnp.asarray(UR5E_HOME, jnp.float32))
        self._jit_cache = {}
        self._cands = None

    def world(self) -> World:
        """The unedited scene: bench + the fixture's vertical parts (with the tree's turn applied)."""
        return World(table_z=self.planner.table_z, fixtures=fixture_cylinders(self.planner.scene))

    # ------------------------------------------------------------------ grasp candidates
    def ring_candidates(self, obj_traj: onp.ndarray, k: int = 4) -> list:
        p = self.planner.plan(obj_traj)
        tries = sorted(p["tries"], key=lambda t: (t["status"] != "ok", t["joint_path"]))[:k]
        return [(int(round(t["azimuth_deg"])), self.planner.grasp(onp.radians(t["azimuth_deg"])), t["status"]) for t in tries]

    def nominal(self, obj_traj: onp.ndarray | None = None, grasp_rank: int = 0) -> dict:
        ph = self.ph
        obj_traj = self.demo_obj if obj_traj is None else obj_traj
        if self._cands is None:
            self._cands = self.ring_candidates(obj_traj)
        az, grasp, _ = self._cands[grasp_rank]
        T = len(obj_traj)
        home = self.fk_tcp(jnp.asarray(UR5E_HOME, jnp.float32))
        objs = [jaxlie.SE3(jnp.asarray(o, jnp.float32)) for o in obj_traj]
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
        for _ in range(ph.close):
            tcp.append(g0); obj.append(objs[0]); phase.append("close"); pinned.append(True); held.append(False)
        z0 = obj_traj[0, 6]
        for i in range(T):
            tcp.append(objs[i] @ grasp); obj.append(objs[i]); phase.append("carry"); held.append(True)
            near_goal = onp.linalg.norm(obj_traj[i, 4:] - obj_traj[-1, 4:]) < self.cfg["place_pin_r"]
            pinned.append(bool(obj_traj[i, 6] < z0 + ph.lift_tol or near_goal))
        c = onp.arange(ph.approach + ph.close, ph.approach + ph.close + T)
        free_c = c[~onp.array(pinned)[c]]
        if len(free_c):
            for k in range(free_c[0], free_c[-1] + 1):
                pinned[k] = False
        for _ in range(ph.release):
            tcp.append(gT); obj.append(objs[-1]); phase.append("release"); pinned.append(True); held.append(True)
        for i in range(ph.retreat):
            s = i / max(ph.retreat - 1, 1)
            tcp.append(interp(gT, gT @ up, ss(s))); obj.append(objs[-1]); phase.append("retreat")
            pinned.append(True); held.append(False)
        stack = lambda xs: jaxlie.SE3(jnp.stack([x.wxyz_xyz for x in xs]))  # noqa: E731
        return dict(tcp=stack(tcp), obj=stack(obj), grasp=grasp, grasp_yaw_k=az, phase=onp.array(phase),
                    pinned=onp.array(pinned), held=onp.array(held))

    # ------------------------------------------------------------------ full pipeline
    def _run_one(self, world: World, obj_traj, grasp_rank: int, seeds_m: float, repair: bool = True) -> dict:
        out = super()._run_one(world, obj_traj, grasp_rank, seeds_m, repair)
        if out.get("q") is not None:  # the ring planner's full check: branches, fingertips, limits
            nom = out["nominal"]
            # pinned steps keep their targets (DemoRepair.check verifies them); free steps are free by design
            fk = onp.asarray(self.fk_tcp(jnp.asarray(out["q"])).wxyz_xyz)
            tcp = onp.where(nom["pinned"][:, None], onp.asarray(nom["tcp"].wxyz_xyz), fk)
            chk = self.planner.check(onp.asarray(out["q"]), dict(tcp=tcp, phase=nom["phase"]))
            out["fixture_check"] = chk["status"]
            if not chk["ok"]:
                out.update(status="repair_failed" if out["status"] == "repaired" else "reject_" + chk["status"], q=None)
        return out

    def run(self, world: World, obj_traj: onp.ndarray | None = None, *, seeds_m: float = 0.14,
            try_grasp_symmetries: bool = True, repair: bool = True) -> dict:
        t0 = time.time()
        obj_traj = self.demo_obj if obj_traj is None else obj_traj
        self._cands = self.ring_candidates(obj_traj)
        attempts = []
        for rank in range(len(self._cands) if try_grasp_symmetries else 1):
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
                   attempts=[dict(rank=i, azimuth=r["nominal"]["grasp_yaw_k"], status=r["status"]) for i, r in enumerate(attempts)])
        if ok and ok[0] is not base:
            out["status"] = ok[0]["status"] + "_other_grasp_yaw"
        out["time_s"] = time.time() - t0
        self._cands = None
        return out


def main(task: str = "tape", xy: tuple[float, float] = (0.03, -0.80), obstacle: Path = REPO / "data/obstacles/sauce_bottle/obstacle.json") -> None:
    rep = RingRepair(task)
    meta = json.loads(obstacle.read_text())
    from tape_plan import rotate_traj_about_post
    from trajgen.traj_interp import traj_interp_batch
    T = rep.demo_obj
    if rep.cfg["tree_yaw"]:
        T = rotate_traj_about_post(T, onp.radians(rep.cfg["tree_yaw"]), rep.planner.post_xy)
    ns = rep.demo_obj[:1].copy()
    ns[0, 4:6] += rep.cfg["start_offset"]
    T = onp.asarray(traj_interp_batch(traj=T, new_starts=ns, proportion=0.6)[0], onp.float64)
    w = rep.world().with_cylinder(xy[0], xy[1], meta["radius"], meta["height"])
    out = rep.run(w, T)
    print(task, "bottle at", xy, "->", out["status"], "| plain:", out["baseline_status"],
          "| min clearance before", round(out["before"]["min_obstacle_clearance"], 4),
          "after", None if out.get("after") is None else round(out["after"]["min_obstacle_clearance"], 4),
          "| attempts", out["attempts"], f"| {out['time_s']:.1f} s")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
