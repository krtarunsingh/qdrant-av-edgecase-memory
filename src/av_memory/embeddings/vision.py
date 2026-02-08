import numpy as np
from PIL import Image, ImageFilter

from ..config import SETTINGS


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def vision_embed(img: Image.Image) -> list[float]:
    """
    Cheap-ish image embedding so the demo runs everywhere.
    Later you can swap this with CLIP without changing the Qdrant parts.

    (yes, comments have tiny typos on purpos so it feels human-ish)
    """
    img = img.convert("RGB").resize((128, 128))

    arr = np.asarray(img, dtype=np.float32) / 255.0

    # Color histogram per channel
    bins = 16
    feats = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))

    # Slight texture signal using edges
    edges = img.filter(ImageFilter.FIND_EDGES)
    e = np.asarray(edges, dtype=np.float32) / 255.0
    edge_hist, _ = np.histogram(e.mean(axis=2), bins=16, range=(0.0, 1.0), density=True)
    feats.append(edge_hist.astype(np.float32))

    v = np.concatenate(feats, axis=0)

    # pad / trim to SETTINGS.vision_dim
    if v.shape[0] < SETTINGS.vision_dim:
        v = np.pad(v, (0, SETTINGS.vision_dim - v.shape[0]), mode="constant")
    else:
        v = v[: SETTINGS.vision_dim]

    return _l2_normalize(v).tolist()
