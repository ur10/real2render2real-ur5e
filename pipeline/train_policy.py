"""Train the visuomotor policy (policy.py) on a dataset from build_policy_dataset.py.

Behaviour cloning with action chunks: from (images_t, q_t) predict q_{t+1..t+16}, L1 on normalized
joints. Same model, augmentation and number of gradient steps for every dataset, so differences in
closed-loop success come from the data.

Example:
    python scripts/ur_cube/train_policy.py --data outputs/ur_cube/policy_data/r2r2r_400 \
        --output outputs/ur_cube/policies/r2r2r_400
"""

import json
import math
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import torch
import torch.nn.functional as F

from policy import HORIZON, VisuomotorPolicy


def augment(imgs: torch.Tensor, shift: int = 6) -> torch.Tensor:
    """Random translation (+-shift px) and mild brightness/contrast jitter. imgs: (n, c, H, W, 3) uint8."""
    n, c, H, W, _ = imgs.shape
    x = imgs.flatten(0, 1).permute(0, 3, 1, 2).float()
    t = (torch.rand(n * c, 2, device=x.device) * 2 - 1) * shift
    theta = torch.zeros(n * c, 2, 3, device=x.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1
    theta[:, 0, 2], theta[:, 1, 2] = t[:, 0] * 2 / W, t[:, 1] * 2 / H
    x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), padding_mode="border", align_corners=False)
    b = 1 + (torch.rand(n * c, 1, 1, 1, device=x.device) * 2 - 1) * 0.1
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * (1 + (torch.rand_like(b) * 2 - 1) * 0.1) + m) * b
    return x.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).view(n, c, H, W, 3)


def main(data: Path, output: Path, steps: int = 20000, batch: int = 64, lr: float = 3e-4, trunk_lr: float = 1e-4,
         val_frac: float = 0.05, seed: int = 0, device: str = "cuda") -> None:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    images = np.load(data / "images.npy", mmap_mode="r")
    qpos = np.load(data / "qpos.npy")
    ends = np.load(data / "episode_ends.npy")
    starts = np.concatenate([[0], ends[:-1]])
    n_ep = len(ends)
    ep_of = np.repeat(np.arange(n_ep), ends - starts)
    # Validation episodes (only if there are enough episodes to spare some).
    n_val = int(round(n_ep * val_frac)) if n_ep >= 20 else 0
    val_eps = set(rng.choice(n_ep, n_val, replace=False).tolist()) if n_val else set()
    idx_train = np.array([i for i in range(len(qpos)) if ep_of[i] not in val_eps])
    idx_val = np.array([i for i in range(len(qpos)) if ep_of[i] in val_eps])

    # action chunk index table: q[min(i + 1 + h, end - 1)]
    end_of = ends[ep_of] - 1
    act_idx = np.minimum(np.arange(len(qpos))[:, None] + 1 + np.arange(HORIZON)[None], end_of[:, None])

    model = VisuomotorPolicy().to(device)
    model.q_mean.copy_(torch.tensor(qpos.mean(0)))
    model.q_std.copy_(torch.tensor(np.maximum(qpos.std(0), 1e-2)))
    trunk = list(model.trunk.parameters())
    rest = [p for n, p in model.named_parameters() if not n.startswith("trunk.")]
    opt = torch.optim.AdamW([dict(params=trunk, lr=trunk_lr), dict(params=rest, lr=lr)], weight_decay=1e-4)
    base_lrs = [trunk_lr, lr]
    scaler = torch.amp.GradScaler()
    q_all = torch.tensor(qpos, device=device)
    act_all = torch.tensor(act_idx, device=device)

    q_in: Queue = Queue(maxsize=6)

    def loader(worker_seed):  # one RNG per thread (numpy Generators are not thread-safe)
        r = np.random.default_rng(worker_seed)
        while True:
            b = np.sort(r.choice(idx_train, batch, replace=len(idx_train) < batch))
            q_in.put((b, torch.from_numpy(np.ascontiguousarray(images[b]))))
    for w in range(3):
        threading.Thread(target=loader, args=(seed * 100 + w + 1,), daemon=True).start()

    def loss_fn(b, imgs, aug):
        imgs = imgs.to(device, non_blocking=True)
        if aug:
            imgs = augment(imgs)
        bt = torch.as_tensor(b, device=device)
        q = q_all[bt]
        target = (q_all[act_all[bt]] - model.q_mean) / model.q_std
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(imgs, q)
        return (pred.float() - target).abs().mean()

    output.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    warm = 500
    for step in range(1, steps + 1):
        f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm)))
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * f
        b, imgs = q_in.get()
        model.train()
        loss = loss_fn(b, imgs, aug=True)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step % 500 == 0 or step == steps:
            rec = dict(step=step, loss=round(loss.item(), 5), time_min=round((time.time() - t0) / 60, 1))
            if len(idx_val):
                model.eval()
                with torch.no_grad():
                    vb = np.sort(rng.choice(idx_val, min(256, len(idx_val)), replace=False))
                    rec["val_loss"] = round(loss_fn(vb, torch.from_numpy(np.ascontiguousarray(images[vb])), aug=False).item(), 5)
            log.append(rec)
            print(rec, flush=True)
    meta = json.loads((data / "meta.json").read_text())
    torch.save(dict(model=model.state_dict(), model_kwargs={}, data=str(data), rollouts=meta["rollouts"],
                    frames=meta["frames"], steps=steps, batch=batch, log=log), output / "policy.pt")
    (output / "train_log.json").write_text(json.dumps(dict(data=str(data), rollouts=meta["rollouts"], log=log), indent=1))
    print("saved", output / "policy.pt")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
