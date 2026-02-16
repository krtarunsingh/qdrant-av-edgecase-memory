import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ..config import SETTINGS


POINTNET_CHECKPOINT_REPO = "nanopiero/pointnet_igloos"
POINTNET_CHECKPOINT_FILE = "pointnet_500_ep.pth"
POINTNET_SAMPLE_POINTS = 1024
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class _TNet(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2, keepdim=False).values
        x = F.relu(self.bn4(self.linear1(x)))
        x = F.relu(self.bn5(self.linear2(x)))
        x = self.linear3(x)

        ident = torch.eye(self.k, dtype=x.dtype, device=x.device).reshape(1, self.k * self.k)
        x = x + ident.repeat(b, 1)
        return x.view(-1, self.k, self.k)


class _PointNetBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tnet1 = _TNet(k=3)
        self.tnet2 = _TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trans1 = self.tnet1(x)
        x = torch.bmm(trans1, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        trans2 = self.tnet2(x)
        x = torch.bmm(trans2, x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = torch.max(x, dim=2, keepdim=False).values
        return x


def _prepare_points(points_xyz: np.ndarray, sample_points: int = POINTNET_SAMPLE_POINTS) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N,3)")
    if pts.shape[0] == 0:
        raise ValueError("points_xyz is empty")

    pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
    center = pts.mean(axis=0, keepdims=True)
    pts = pts - center
    scale = float(np.max(np.linalg.norm(pts, axis=1)))
    if scale > 1e-6:
        pts = pts / scale

    n = pts.shape[0]
    if n >= sample_points:
        idx = np.linspace(0, n - 1, sample_points, dtype=np.int64)
        pts = pts[idx]
    else:
        reps = int(np.ceil(sample_points / n))
        idx = np.tile(np.arange(n, dtype=np.int64), reps)[:sample_points]
        pts = pts[idx]
    return pts.astype(np.float32)


@lru_cache(maxsize=1)
def _load_pointnet_backbone() -> tuple[_PointNetBackbone, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = os.getenv("AV_LIDAR_POINTNET_REPO", POINTNET_CHECKPOINT_REPO)
    filename = os.getenv("AV_LIDAR_POINTNET_FILE", POINTNET_CHECKPOINT_FILE)

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError(f"Unexpected PointNet checkpoint format at {ckpt_path}")

    backbone_state = {
        k[len("backbone.") :]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith("backbone.")
    }
    if not backbone_state:
        raise ValueError("PointNet checkpoint is missing backbone.* weights")

    model = _PointNetBackbone()
    model.load_state_dict(backbone_state, strict=True)
    model.eval()
    model.to(device)
    return model, device


def lidar_embed(points_xyz: np.ndarray) -> list[float]:
    model, device = _load_pointnet_backbone()
    pts = _prepare_points(points_xyz)
    x = torch.from_numpy(pts).transpose(0, 1).unsqueeze(0).to(device)

    with torch.inference_mode():
        feat = model(x)
        feat = F.normalize(feat, p=2, dim=-1)

    v = feat.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
    if v.shape[0] != SETTINGS.lidar_dim:
        raise ValueError(
            f"PointNet embedding dim={v.shape[0]} does not match SETTINGS.lidar_dim={SETTINGS.lidar_dim}. "
            "Update the configured dimension and recreate the collection."
        )
    return v.tolist()

