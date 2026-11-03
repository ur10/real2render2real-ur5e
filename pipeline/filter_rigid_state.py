"""Cut a DiG rigid-body state down to the part above a height in the robot base frame.

GARField can't separate the cube from the fixture it sits on at any group level, so the saved
state also contains the fixture top and some table. Gaussians above the fixture top are the cube
(dark, ~4 cm footprint); below are fixture/table. We keep the cube and re-export with DiG's own
writers (state .pt, point_cloud.ply, object renders/masks, cameras.json) so RSRD and SuGaR see the
format they expect. The original state is moved out of the glob path RSRD loads from.

Run in the r2r2r_rsrd env:
    python scripts/ur_cube/filter_rigid_state.py \
        --dig-config outputs/table_new/dig/2026-11-02_184246/config.yml \
        --state outputs/table_new/state_rigid_20261102_185432.pt \
        --scan-to-base outputs/table_new/scan_to_base.json --min-z 0.0095
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tyro
from nerfstudio.utils.eval_utils import eval_setup


def main(
    dig_config: Path,
    state: Path,
    scan_to_base: Path,
    min_z: float = 0.0095,
    max_radius_xy: float = 0.035,
) -> None:
    params = torch.load(state, map_location="cpu")
    T = np.array(json.loads(scan_to_base.read_text())["T_base_model"])
    means_b = params["means"].detach().numpy() @ T[:3, :3].T + T[:3, 3]

    high = means_b[:, 2] > min_z
    center = np.median(means_b[high, :2], axis=0)
    keep = high & (np.linalg.norm(means_b[:, :2] - center, axis=1) < max_radius_xy)
    kept = means_b[keep]
    print(f"keeping {keep.sum()}/{len(keep)} gaussians; base-frame center {np.median(kept, 0).round(4)}, "
          f"extent {(np.percentile(kept, 98, 0) - np.percentile(kept, 2, 0)).round(4)} m")

    def images_on_cpu(cfg):
        # The DiG datamanager stacks every training image at once; on GPU that needs ~13 GB.
        cfg.pipeline.datamanager.cache_images = "cpu"
        return cfg

    _, pipeline, _, _ = eval_setup(dig_config, test_mode="inference", update_config_callback=images_on_cpu)
    keep_t = torch.from_numpy(keep)
    for name in pipeline.model.gauss_params.keys():
        pipeline.model.gauss_params[name] = torch.nn.Parameter(params[name].detach()[keep_t].clone().to(pipeline.device))

    stem = "state_rigid" + datetime.now().strftime("_%Y%m%d_%H%M%S")
    out_pt = pipeline.state_dir / f"{stem}.pt"
    torch.save({k: v for k, v in pipeline.model.gauss_params.items()}, out_pt)
    (pipeline.state_dir / stem).mkdir(exist_ok=True)
    with torch.no_grad():
        pipeline.write_state_to_ply(dir_name=stem)
        pipeline.save_rendered_images(dir_name=stem)
        pipeline.convert_cameras(dir_name=stem)
    print(f"wrote {out_pt} and {pipeline.state_dir / stem}/")

    # Move the unfiltered state out of the state_rigid_*.pt glob RSRD loads from.
    backup = pipeline.state_dir / "_unfiltered_states"
    backup.mkdir(exist_ok=True)
    for p in (state, state.with_suffix("")):
        if p.exists():
            shutil.move(str(p), backup / p.name)
    print(f"moved original state to {backup}/")


if __name__ == "__main__":
    tyro.cli(main)
