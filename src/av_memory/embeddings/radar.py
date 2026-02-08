import numpy as np
from ..config import SETTINGS


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def radar_embed(signal: np.ndarray) -> list[float]:
    """
    signal: (T,) float32
    """
    s = np.asarray(signal, dtype=np.float32).reshape(-1)

    # FFT magnitude
    fft = np.fft.rfft(s)
    mag = np.abs(fft).astype(np.float32)

    # downsample into bins
    bins = 128
    if mag.shape[0] < bins:
        mag = np.pad(mag, (0, bins - mag.shape[0]), mode="constant")
    else:
        idx = np.linspace(0, mag.shape[0] - 1, bins).astype(int)
        mag = mag[idx]

    v = mag[: SETTINGS.radar_dim]
    if v.shape[0] < SETTINGS.radar_dim:
        v = np.pad(v, (0, SETTINGS.radar_dim - v.shape[0]), mode="constant")

    return _l2_normalize(v).tolist()
