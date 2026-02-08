import time
import random

from av_memory.schema import get_client
from av_memory.config import SETTINGS
from av_memory.search import search_fused, SearchWeights

# We'll "query by example": pick a random stored point, use its vectors as query vectors.
# This is super practical in real AV stacks too (replay / retrieval), and it's easy to demo.


def main() -> None:
    client = get_client()

    # Pick a random scenario id from range used in ingestion
    pick = random.randint(0, 1999)
    sid = f"scn_{pick:07d}"

    pt = client.retrieve(collection_name=SETTINGS.collection, ids=[sid], with_vectors=True, with_payload=True)
    if not pt:
        raise RuntimeError("Could not retrieve example point. Did you ingest data?")

    base = pt[0]
    qvecs = base.vector  # named vectors dict
    payload = base.payload or {}

    print("🔎 Query-by-example scenario:")
    print(f"   id: {sid}")
    print(f"   label: {payload.get('label')}")
    print(f"   weather: {payload.get('weather')}, time: {payload.get('time_of_day')}, road: {payload.get('road_type')}")
    print(f"   notes: {payload.get('notes')}")
    print("")

    now = int(time.time())
    last_12_months = 60 * 60 * 24 * 365

    results = search_fused(
        client=client,
        query_vectors=qvecs,
        weights=SearchWeights(vision=0.45, lidar=0.30, radar=0.15, text=0.10),
        limit_per_modality=40,
        top_k=12,
        # Example: filter for same time_of_day, and restrict to last 12 months
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
