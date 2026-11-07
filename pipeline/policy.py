"""Visuomotor policy for the UR5e cube task, shared by training (train_policy.py) and closed-loop
evaluation (ur5e_cube_eval.py).

Observation: the two static RealSense views (front, right), each cropped to the workspace and resized
to 224x128, plus the 7 joint positions (6 arm + finger_joint). Action: the next H = 16 joint
positions (absolute, 7-D), executed open-loop for `exec_steps` before re-planning (receding horizon).

Network: an ImageNet ResNet-18 trunk (frozen batch-norm statistics) per camera view, a spatial-softmax
keypoint head per camera, the joint state through a small MLP, and an MLP to the action chunk.
L1 loss on normalized actions. Deliberately plain so dataset differences, not model tricks, drive
the comparison.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_W, IMG_H = 224, 128
# Workspace crops in full-resolution (1280x720) pixel coordinates: x0, y0, x1, y1 (aspect 1.75).
CROPS = {"camera_0": (320, 150, 1040, 561), "camera_1": (160, 171, 1120, 720)}
CAMERAS = ("camera_0", "camera_1")
HORIZON = 16


def crop_resize(img: np.ndarray, camera: str) -> np.ndarray:
    """img: (H, W, 3) uint8 at any scale of 1280x720 -> (IMG_H, IMG_W, 3) uint8."""
    s = img.shape[1] / 1280.0
    x0, y0, x1, y1 = (int(round(v * s)) for v in CROPS[camera])
    return cv2.resize(img[y0:y1, x0:x1], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def jpeg_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Same lossy step the dataset images went through (cv2.imwrite default quality, BGR)."""
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class SpatialSoftmax(nn.Module):
    def __init__(self, c_in: int, k: int = 32):
        super().__init__()
        self.proj = nn.Conv2d(c_in, k, 1)

    def forward(self, x):
        x = self.proj(x)
        n, k, h, w = x.shape
        p = F.softmax(x.flatten(2), -1).view(n, k, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)  # (n, 2k)


class VisuomotorPolicy(nn.Module):
    def __init__(self, horizon: int = HORIZON, n_cams: int = 2, kp: int = 32, pretrained: bool = True):
        super().__init__()
        import torchvision
        from torchvision.ops.misc import FrozenBatchNorm2d

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        trunk = torchvision.models.resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.trunk = nn.Sequential(*list(trunk.children())[:-2])  # (n, 512, H/32, W/32)
        self.kp = nn.ModuleList([SpatialSoftmax(512, kp) for _ in range(n_cams)])
        self.state = nn.Sequential(nn.Linear(7, 64), nn.ReLU(), nn.Linear(64, 64))
        d = n_cams * 2 * kp + 64
        self.head = nn.Sequential(nn.Linear(d, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
                                  nn.Linear(512, horizon * 7))
        self.horizon = horizon
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # Joint/action normalization (set from the training set).
        self.register_buffer("q_mean", torch.zeros(7))
        self.register_buffer("q_std", torch.ones(7))

    def forward(self, imgs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """imgs: (n, cams, H, W, 3) uint8; q: (n, 7) -> normalized action chunk (n, horizon, 7)."""
        n, c = imgs.shape[:2]
        x = imgs.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.img_mean) / self.img_std
        f = self.trunk(x)
        f = f.view(n, c, *f.shape[1:])
        feats = [self.kp[i](f[:, i]) for i in range(c)]
        s = self.state((q - self.q_mean) / self.q_std)
        out = self.head(torch.cat(feats + [s], -1))
        return out.view(n, self.horizon, 7)

    @torch.no_grad()
    def act(self, imgs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Absolute joint targets (n, horizon, 7)."""
        return self(imgs, q) * self.q_std + self.q_mean


def load_policy(path, device="cuda") -> VisuomotorPolicy:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = VisuomotorPolicy(**ck.get("model_kwargs", {}), pretrained=False).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m
