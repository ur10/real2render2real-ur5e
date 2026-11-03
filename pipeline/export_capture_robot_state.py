"""Export the per-episode robot state from the capture's replay_buffer.zarr to JSON.

The pipeline envs don't have zarr, so run this once with any env that does (e.g. data_collect):
    ~/miniconda3/envs/data_collect/bin/python scripts/ur_cube/export_capture_robot_state.py \
        ~/ur_human_capture outputs/ur_cube/capture_robot_state.json

robot0_eef_quat in this buffer is a 3-vector, i.e. a rotation vector (UR RTDE convention).
"""

import json
import sys
from pathlib import Path

import zarr

capture_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
z = zarr.open(str(capture_dir / "replay_buffer.zarr"), "r")
ends = [int(e) for e in z["meta/episode_ends"][:]]
episodes = []
for i, end in enumerate(ends):
    start = 0 if i == 0 else ends[i - 1]
    episodes.append({
        "episode": i,
        "steps": end - start,
        "tcp_pos": z["data/robot0_eef_pos"][start].tolist(),
        "tcp_rotvec": z["data/robot0_eef_quat"][start].tolist(),
        "gripper_qpos": z["data/robot0_gripper_qpos"][start].tolist(),
        "tcp_pos_std": z["data/robot0_eef_pos"][start:end].std(0).tolist(),
    })
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"episodes": episodes}, indent=2))
print(f"wrote {output}")
