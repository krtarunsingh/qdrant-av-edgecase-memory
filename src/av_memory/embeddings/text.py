import os
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..config import SETTINGS


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_text_model() -> SentenceTransformer:
    model_id = os.getenv("AV_TEXT_MODEL_ID", DEFAULT_TEXT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_id, device=device)


def text_embed(text: str) -> list[float]:
    model = _load_text_model()
    v = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    if v.shape[0] != SETTINGS.text_dim:
        raise ValueError(
            f"Text embedding dim={v.shape[0]} does not match SETTINGS.text_dim={SETTINGS.text_dim}. "
            "Update the configured dimension and recreate the collection."
        )
    return v.tolist()

