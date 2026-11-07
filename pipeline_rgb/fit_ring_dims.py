"""Measure a ring-shaped object (outer radius, hole radius, thickness) from its coloured silhouette in
the calibrated views, on rest frames in different places. The generic version of fit_tape_dims.py,
used for the green nut:

    python scripts/ur_tape/fit_ring_dims.py --colour green --no-core --frames 4:0 4:465 \
        --inits "-0.136,-0.92,-0.012,0,0,1" "0.086,-0.638,0.172,0,0,1" --output outputs/ur_nut/nut_dims.json

Same-coloured scenery (the bolt stand's green spacers) is ignored: it is whatever has the colour in
each episode's first frame away from the object. A frame where only one view sees the object only
contributes that view.
"""

import json
import sys
from pathlib import Path

import numpy as np
import tyro
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from cams import load_cams  # noqa: E402
from fit_tape_dims import read_frame  # noqa: E402
from tape_model import TapeDims, angles_from_axis, axis_from_angles  # noqa: E402
from track_tape_rgb import Fitter, static_masks  # noqa: E402


def main(frames: tuple[str, ...], inits: tuple[str, ...], output: Path, colour: str = "green", core: bool = False,
         R: float = 0.025, r_in: float = 0.011, w: float = 0.015, roi_r: float = 0.07) -> None:
    cams = load_cams()
    poses, obs_list, statics = [], [], {}
    d0 = TapeDims(R=R, r_core=r_in, r_in=r_in, w=w, core=core)
    for key, ini in zip(frames, inits):
        ep, f = map(int, key.split(":"))
        v = np.array([float(x) for x in ini.split(",")])
        p = np.r_[v[:3], angles_from_axis(v[3:])]
        if ep not in statics:  # first frame of the episode, object at the first listed pose of that episode
            statics[ep] = static_masks(cams, [read_frame(ep, i, 0) for i in range(2)], d0, colour, p)
        ims = [read_frame(ep, i, f) for i in range(2)]
        fit = Fitter(cams, d0, colour=colour, static=statics[ep], roi_r=roi_r)
        obs = fit.observe(ims, p[:3])
        q, c, _ = fit.fit(ims, p)
        print(f"{key}: pose-only 1-IoU {c:.3f}, views {sum(x is not None for x in fit.view_costs(obs, q))}", flush=True)
        poses.append(q)
        obs_list.append((ep, obs, sum(x is not None for x in fit.view_costs(obs, q))))

    def dims_of(x):
        return TapeDims(R=x[0], r_core=x[1], r_in=x[1], w=x[2], core=core)

    def total(x):
        if not (0.003 < x[1] < x[0] - 0.003 and 0.004 < x[2] < 0.06):
            return 10.0
        fit = Fitter(cams, dims_of(x), colour=colour, roi_r=roi_r)
        tot = 0.0
        for k, (ep, obs, nv) in enumerate(obs_list):
            pk = x[3 + 5 * k: 8 + 5 * k]
            if nv < 2:  # one view: keep the resting height/axis, fit xy
                pk = np.r_[pk[:2], poses[k][2:]]
            tot += fit.cost(obs, pk)[0]
        return tot

    x0 = np.r_[R, r_in, w, np.concatenate(poses)]
    r = minimize(total, x0, method="Powell", options=dict(xtol=1e-5, ftol=1e-5, maxfev=6000))
    dims = dims_of(r.x)
    fit = Fitter(cams, dims, colour=colour, roi_r=roi_r)
    per = [fit.cost(obs, r.x[3 + 5 * k: 8 + 5 * k])[0] for k, (_, obs, _) in enumerate(obs_list)]
    print("dims", dims, "1-IoU per frame", np.round(per, 3))
    out = dict(R=dims.R, r_core=dims.r_core, r_in=dims.r_in, w=dims.w, core=core, colour=colour,
               fit_frames=list(frames[:len(poses)]), mean_iou_per_frame=[round(1 - p, 3) for p in per],
               poses=[dict(frame=frames[k], centre=r.x[3 + 5 * k: 6 + 5 * k].tolist(),
                           axis=axis_from_angles(*r.x[6 + 5 * k: 8 + 5 * k]).tolist()) for k in range(len(poses))])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    tyro.cli(main)
