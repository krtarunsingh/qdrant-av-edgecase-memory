import numpy as np
from ..config import SETTINGS


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def lidar_embed(points_xyz: np.ndarray) -> list[float]:
    """
    points_xyz: (N,3) float32
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N,3)")

    # Cast once up front so downstream math is predictable and fast.
    pts = points_xyz.astype(np.float32)

    # Global stats summarize scene scale and spread.
    mean = pts.mean(axis=0)
    std = pts.std(axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    # Add a radial histogram to preserve rough spatial density around ego.
    r = np.linalg.norm(pts, axis=1)
    hist, _ = np.histogram(r, bins=64, range=(0.0, max(1.0, float(r.max()))), density=True)

    v = np.concatenate([mean, std, mins, maxs, hist.astype(np.float32)], axis=0)

    # Keep output size fixed to match the Qdrant named-vector schema.
    if v.shape[0] < SETTINGS.lidar_dim:
        v = np.pad(v, (0, SETTINGS.lidar_dim - v.shape[0]), mode="constant")
    else:
        v = v[: SETTINGS.lidar_dim]

    return _l2_normalize(v).tolist()

