"""Pack generated rollouts into a training set for train_policy.py.

Each source is a `successes` folder of the generator (run_ur5e_cube.py) and how many rollouts to take
from it (sorted by name, i.e. generation order). Images are cropped to the workspace and resized with
the same function the closed-loop evaluator uses (policy.crop_resize).

Output (<output>/): images.npy (N, 2, 128, 224, 3) uint8, memory-mapped; qpos.npy (N, 7) float32
(6 arm joints + finger_joint, the logged joint_angles); episode_ends.npy; meta.json.

Example:
    python scripts/ur_cube/build_policy_dataset.py --output outputs/ur_cube/policy_data/r2r2r_100 \
        --sources output_data/ur5e_pol_r2r2r/successes:100
"""

import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

from policy import CAMERAS, IMG_H, IMG_W, crop_resize


def load_rollout(d: str):
    d = Path(d)
    q = h5py.File(d / "robot_data" / "robot_data.h5", "r")["joint_angles"][()].astype(np.float32)
    imgs = []
    for cam in CAMERAS:
        files = sorted((d / cam / "rgb").glob("*.jpg"))
        imgs.append(np.stack([crop_resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB), cam) for f in files]))
    n = min(len(q), *(len(i) for i in imgs))
    return q[:n], np.stack([i[:n] for i in imgs], 1)


def main(output: Path, sources: tuple[str, ...], workers: int = 8) -> None:
    dirs, used = [], []
    for s in sources:
        path, count = s.rsplit(":", 1)
        found = sorted(p for p in Path(path).iterdir() if (p / "robot_data" / "robot_data.h5").exists())
        if len(found) < int(count):
            raise SystemExit(f"{path}: only {len(found)} rollouts, asked for {count}")
        dirs += [str(p) for p in found[: int(count)]]
        used.append(dict(source=path, count=int(count)))
    lens = []
    output.mkdir(parents=True, exist_ok=True)
    for d in dirs:  # sizes first (cheap: h5 shape + file counts)
        n_img = min(len(list((Path(d) / c / "rgb").glob("*.jpg"))) for c in CAMERAS)
        n_q = h5py.File(Path(d) / "robot_data" / "robot_data.h5", "r")["joint_angles"].shape[0]
        lens.append(min(n_img, n_q))
    N = int(sum(lens))
    images = np.lib.format.open_memmap(output / "images.npy", mode="w+", dtype=np.uint8, shape=(N, len(CAMERAS), IMG_H, IMG_W, 3))
    qpos = np.zeros((N, 7), np.float32)
    i = 0
    with Pool(workers) as pool:
        for k, (q, im) in enumerate(pool.imap(load_rollout, dirs)):
            n = lens[k]
            images[i:i + n], qpos[i:i + n] = im[:n], q[:n]
            i += n
            if k % 50 == 0:
                print(f"{k + 1}/{len(dirs)} rollouts", flush=True)
    images.flush()
    np.save(output / "qpos.npy", qpos)
    np.save(output / "episode_ends.npy", np.cumsum(lens))
    (output / "meta.json").write_text(json.dumps(dict(sources=used, rollouts=len(dirs), frames=N, rollout_dirs=dirs), indent=1))
    print(f"{output}: {len(dirs)} rollouts, {N} frames, {images.nbytes / 1e9:.2f} GB")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
