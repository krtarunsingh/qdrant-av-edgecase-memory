import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from ..config import SETTINGS


RADAR_MODEL_REPO = "VyDat/Radar_Signal-Classification"
RADAR_MODEL_FILE = "22139012_22139015.pt"
RADAR_IMAGE_SIZE = 64
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def _signal_to_model_input(signal: np.ndarray) -> torch.Tensor:
    s = np.asarray(signal, dtype=np.float32).reshape(-1)
    if s.size == 0:
        raise ValueError("signal is empty")

    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    s = s - float(s.mean())
    scale = float(s.std()) + 1e-6
    s = s / scale

    x = torch.from_numpy(s)
    spec = torch.stft(
        x,
        n_fft=64,
        hop_length=8,
        win_length=64,
        return_complex=True,
    )
    mag = torch.log1p(torch.abs(spec))
    mag = mag.unsqueeze(0).unsqueeze(0)
    mag = F.interpolate(mag, size=(RADAR_IMAGE_SIZE, RADAR_IMAGE_SIZE), mode="bilinear", align_corners=False)
    mag = mag.squeeze(0).squeeze(0)

    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
    dt = torch.diff(mag, dim=1, prepend=mag[:, :1])
    df = torch.diff(mag, dim=0, prepend=mag[:1, :])

    img = torch.stack([mag, dt, df], dim=0).unsqueeze(0)
    return img.to(dtype=torch.float32)


def _extract_penultimate_features(model: torch.jit.RecursiveScriptModule, x: torch.Tensor) -> torch.Tensor:
    # Follow the scripted network forward path and stop before final classification layer.
    x = model.relu(model.bn1(model.conv1(x)))
    x = model.msm1(model.Block1(x))
    x = model.Block2(model.transition1(x))
    x = model.transition2(model.msm2(x))
    x = model.msm3(model.Block3(x))
    x = model.global_pool(x)
    x = x.view(x.size(0), -1)
    x = torch.relu(model.fc1(x))
    return x


@lru_cache(maxsize=1)
def _load_radar_model() -> tuple[torch.jit.RecursiveScriptModule, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = os.getenv("AV_RADAR_MODEL_REPO", RADAR_MODEL_REPO)
    filename = os.getenv("AV_RADAR_MODEL_FILE", RADAR_MODEL_FILE)

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model, device


def radar_embed(signal: np.ndarray) -> list[float]:
    model, device = _load_radar_model()
    x = _signal_to_model_input(signal).to(device)

    with torch.inference_mode():
        feat = _extract_penultimate_features(model, x)
        feat = F.normalize(feat, p=2, dim=-1)

    v = feat.squeeze(0).detach().cpu().to(dtype=torch.float32).numpy()
    if v.shape[0] != SETTINGS.radar_dim:
        raise ValueError(
            f"Radar embedding dim={v.shape[0]} does not match SETTINGS.radar_dim={SETTINGS.radar_dim}. "
            "Update the configured dimension and recreate the collection."
        )
    return v.tolist()
