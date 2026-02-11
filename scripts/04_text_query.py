import argparse
import time

from av_memory.schema import get_client
from av_memory.config import SETTINGS
from av_memory.search import search_fused, SearchWeights, is_novel_scene
from av_memory.embeddings import text_embed


def main() -> None:
    # I expose common filter knobs in CLI so I can quickly test retrieval behavior.
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=str, required=True, help="Query text, e.g. 'pedestrian low light' or 'slippery road rain'")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--last_months", type=int, default=12)
    ap.add_argument("--time_of_day", type=str, default=None)
    ap.add_argument("--weather", type=str, default=None)
    args = ap.parse_args()
    if args.topk <= 0:
        raise ValueError("--topk must be > 0")
    if args.last_months <= 0:
        raise ValueError("--last_months must be > 0")

    client = get_client()

    # I compute a rolling time window because stale incidents are often less relevant.
    now = int(time.time())
    ts_min = now - (60 * 60 * 24 * 30 * args.last_months)

    # I only embed the query text here; search_fused still handles ranking/filtering.
    q_text = text_embed(args.q)

    results = search_fused(
        client=client,
        query_vectors={"text": q_text},
        weights=SearchWeights(vision=0.0, lidar=0.0, radar=0.0, text=1.0),
        limit_per_modality=50,
        top_k=args.topk,
        ts_min=ts_min,
        ts_max=now,
        time_of_day=args.time_of_day,
        weather=args.weather,
        collection_name=SETTINGS.collection,
    )

    print(f"\nQuery: {args.q}\n")
    for i, r in enumerate(results, start=1):
        p = r["payload"]
        print(
            f"{i:02d}. score={r['fused_score']:.4f} id={r['id']} "
            f"label={p.get('label')} weather={p.get('weather')} time={p.get('time_of_day')} road={p.get('road_type')}"
        )

    # I run a simple novelty check to decide whether this feels like known memory or not.
    novel = is_novel_scene(results, threshold=0.72)
    print("\n🚦 Decision:")
    print("   NEW SCENE / EDGE-CASE (trigger mapping)" if novel else "   KNOWN SCENE (reuse memory)")

    print("")


if __name__ == "__main__":
    main()
