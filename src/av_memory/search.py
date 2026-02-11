from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import SETTINGS


@dataclass
class SearchWeights:
    vision: float = 0.40
    lidar: float = 0.30
    radar: float = 0.15
    text: float = 0.15


def _make_filter(
    weather: str | None = None,
    time_of_day: str | None = None,
    road_type: str | None = None,
    location_bucket: str | None = None,
    ts_min: int | None = None,
    ts_max: int | None = None,
) -> qm.Filter | None:
    # Build clauses only for explicitly provided arguments.
    # This avoids accidental over-constraint of retrieval behavior.
    must: list[qm.FieldCondition] = []

    if weather is not None:
        must.append(qm.FieldCondition(key="weather", match=qm.MatchValue(value=weather)))
    if time_of_day is not None:
        must.append(qm.FieldCondition(key="time_of_day", match=qm.MatchValue(value=time_of_day)))
    if road_type is not None:
        must.append(qm.FieldCondition(key="road_type", match=qm.MatchValue(value=road_type)))
    if location_bucket is not None:
        must.append(qm.FieldCondition(key="location_bucket", match=qm.MatchValue(value=location_bucket)))

    if ts_min is not None or ts_max is not None:
        must.append(
            qm.FieldCondition(
                key="ts",
                range=qm.Range(
                    gte=ts_min,
                    lte=ts_max,
                ),
            )
        )

    if not must:
        return None
    return qm.Filter(must=must)


def search_modality(
    client: QdrantClient,
    vector_name: str,
    query_vector: list[float],
    limit: int = 20,
    filt: qm.Filter | None = None,
    collection_name: str | None = None,
) -> list[qm.ScoredPoint]:
    name = collection_name or SETTINGS.collection

    # Query a single named vector space so each modality can be tuned independently.
    res = client.query_points(
        collection_name=name,
        query=query_vector,          # Use the query arg style supported by this client version.
        using=vector_name,           # Explicitly select the named vector modality.
        limit=limit,
        query_filter=filt,
        with_payload=True,
        with_vectors=False,
    )
    return list(res.points)


def fuse_rankings(
    lists_by_modality: dict[str, Iterable[qm.ScoredPoint]],
    weights: SearchWeights,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """
    Use weighted score fusion here.
    Qdrant scores are strong within a modality, but they are not perfectly calibrated
    across different modalities, so normalize each modality before mixing weights.
    """
    wmap = {
        "vision": weights.vision,
        "lidar": weights.lidar,
        "radar": weights.radar,
        "text": weights.text,
    }

    # Normalize each modality by its top score before weighted aggregation.
    max_score: dict[str, float] = {}
    for mod, items in lists_by_modality.items():
        ms = 0.0
        for it in items:
            if it.score is not None:
                ms = max(ms, float(it.score))
        max_score[mod] = ms if ms > 1e-9 else 1.0

    fused: dict[str, dict[str, Any]] = {}

    for mod, items in lists_by_modality.items():
        w = float(wmap.get(mod, 0.0))
        denom = float(max_score.get(mod, 1.0))

        for sp in items:
            if sp.score is None:
                continue

            sid = str(sp.id)
            # Soft normalization lets weaker modalities still contribute.
            norm_score = float(sp.score) / denom

            if sid not in fused:
                fused[sid] = {
                    "id": sid,
                    "fused_score": 0.0,
                    "per_modality": {},
                    "payload": sp.payload or {},
                }

            fused[sid]["fused_score"] += w * norm_score
            fused[sid]["per_modality"][mod] = float(sp.score)

    out = list(fused.values())
    out.sort(key=lambda x: x["fused_score"], reverse=True)
    return out[:top_k]


def search_fused(
    client: QdrantClient,
    query_vectors: dict[str, list[float]],
    weights: SearchWeights | None = None,
    limit_per_modality: int = 30,
    top_k: int = 20,
    weather: str | None = None,
    time_of_day: str | None = None,
    road_type: str | None = None,
    location_bucket: str | None = None,
    ts_min: int | None = None,
    ts_max: int | None = None,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    if weights is None:
        weights = SearchWeights()

    # Apply the same payload/time filter to each modality query for consistency.
    filt = _make_filter(
        weather=weather,
        time_of_day=time_of_day,
        road_type=road_type,
        location_bucket=location_bucket,
        ts_min=ts_min,
        ts_max=ts_max,
    )

    lists_by_modality: dict[str, list[qm.ScoredPoint]] = {}

    # Retrieve candidates per modality first, then fuse into one ranking.
    for mod in ["vision", "lidar", "radar", "text"]:
        qv = query_vectors.get(mod)
        if not qv:
            continue
        lists_by_modality[mod] = search_modality(
            client=client,
            vector_name=mod,
            query_vector=qv,
            limit=limit_per_modality,
            filt=filt,
            collection_name=collection_name,
        )

    return fuse_rankings(lists_by_modality, weights=weights, top_k=top_k)


def is_novel_scene(
    fused_results: list[dict[str, Any]],
    threshold: float = 0.78,
    min_results: int = 3,
) -> bool:
    """
    Treat the scene as novel when retrieval confidence is weak.
    The threshold is empirical and should be tuned for the active embedder stack.
    """
    # Mark as novel by default when too few candidates are retrieved.
    if len(fused_results) < min_results:
        return True
    best = float(fused_results[0]["fused_score"])
    return best < threshold

