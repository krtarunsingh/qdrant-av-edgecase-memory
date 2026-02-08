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

    res = client.query_points(
        collection_name=name,
        query=query_vector,          # ✅ correct for 1.16.2
        using=vector_name,           # ✅ named vector selector
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
    Weighted score fusion:
    - Qdrant scores are comparable within a modality, but not always across them.
    - We'll do a soft normalization per modality based on max score in that list.
    """
    wmap = {
        "vision": weights.vision,
        "lidar": weights.lidar,
        "radar": weights.radar,
        "text": weights.text,
    }

    # Collect max scores for normalization
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
            sid = str(sp.id)
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
    weights: SearchWeights = SearchWeights(),
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
    filt = _make_filter(
        weather=weather,
        time_of_day=time_of_day,
        road_type=road_type,
        location_bucket=location_bucket,
        ts_min=ts_min,
        ts_max=ts_max,
    )

    lists_by_modality: dict[str, list[qm.ScoredPoint]] = {}

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
