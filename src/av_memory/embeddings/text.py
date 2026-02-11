import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from ..config import SETTINGS


_VEC = HashingVectorizer(
    # I use HashingVectorizer so I do not need a fit/persist step in this demo pipeline.
    # I keep scripts stateless this way, which makes reruns easy while I iterate.
    n_features=SETTINGS.text_dim,
    alternate_sign=False,
    norm=None,
    lowercase=True,
    stop_words="english",
)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v) + 1e-12)
    return (v / n).astype(np.float32)


def text_embed(text: str) -> list[float]:
    # I convert sparse output to dense because Qdrant expects dense vectors here.
    # I normalize to keep cosine behavior stable across short and long query strings.
    x = _VEC.transform([text])
    v = x.toarray().astype(np.float32).reshape(-1)
    return _l2_normalize(v).tolist()
