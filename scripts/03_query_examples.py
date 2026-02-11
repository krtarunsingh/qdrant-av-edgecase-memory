import time
import random

from av_memory.schema import get_client
from av_memory.config import SETTINGS
from av_memory.search import search_fused, SearchWeights

SECONDS_PER_DAY = 60 * 60 * 24

# Query-by-example flow: reuse one stored point's vectors as the query.
# This mirrors replay-style retrieval used in AV debugging workflows.


def main() -> None:
    client = get_client()

    # Sample existing IDs instead of assuming a fixed ingest size.
    sample = client.scroll(
        collection_name=SETTINGS.collection,
        limit=256,
        with_payload=False,
        with_vectors=False,
    )
    points, _ = sample
    if not points:
        raise RuntimeError("Collection is empty. Please ingest data first.")

    sid = random.choice(points).id

    pt = client.retrieve(
        collection_name=SETTINGS.collection,
        ids=[sid],
        with_vectors=True,
        with_payload=True,
    )
    if not pt:
        raise RuntimeError("Could not retrieve example point. Did you ingest data?")

    base = pt[0]
    qvecs = base.vector  # Pass all named vectors forward for fused retrieval.
    payload = base.payload or {}

    print("🔎 Query-by-example scenario:")
    print(f"   id: {payload.get('sid')} (point_id={sid})")
    print(f"   label: {payload.get('label')}")
    print(f"   weather: {payload.get('weather')}, time: {payload.get('time_of_day')}, road: {payload.get('road_type')}")
    print(f"   notes: {payload.get('notes')}")
    print("")

    # Apply an explicit time window to simulate "recent memory" retrieval.
    now = int(time.time())
    last_12_months = SECONDS_PER_DAY * 365

    results = search_fused(
        client=client,
        query_vectors=qvecs,
        weights=SearchWeights(vision=0.45, lidar=0.30, radar=0.15, text=0.10),
        limit_per_modality=40,
        top_k=12,
        # Keep time_of_day fixed and limit to recent data for a realistic retrieval slice.
        time_of_day=payload.get("time_of_day"),
        ts_min=now - last_12_months,
        ts_max=now,
        collection_name=SETTINGS.collection,
    )

    print("🏁 Top fused results:")
    for i, r in enumerate(results, start=1):
        p = r["payload"]
        print(
            f"{i:02d}. score={r['fused_score']:.4f} "
            f"id={r['id']} label={p.get('label')} weather={p.get('weather')} time={p.get('time_of_day')} "
            f"near_miss={p.get('near_miss')}"
        )

    print("\nDone ✅")


if __name__ == "__main__":
    main()

