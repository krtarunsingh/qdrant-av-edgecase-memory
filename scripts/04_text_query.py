import argparse
import time

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from av_memory.schema import get_client
from av_memory.config import SETTINGS
from av_memory.search import search_fused, SearchWeights, is_novel_scene
from av_memory.embeddings import text_embed

SECONDS_PER_DAY = 60 * 60 * 24
SECONDS_PER_MONTH = SECONDS_PER_DAY * 30


def _ensure_qdrant_available(client) -> None:
    try:
        client.get_collections()
    except Exception as exc:
        raise SystemExit(
            "Qdrant is not reachable at "
            f"{SETTINGS.qdrant_url}. Start Docker Desktop, run `docker compose up -d`, "
            "then run `python scripts/01_create_collection.py` and ingest data."
        ) from exc


def main() -> None:
    # Expose common filter knobs in the CLI for quick retrieval tests.
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
    _ensure_qdrant_available(client)

    # Compute a rolling time window because stale incidents are often less relevant.
    now = int(time.time())
    ts_min = now - (SECONDS_PER_MONTH * args.last_months)

    # Embed only query text here; search_fused handles ranking and filtering.
    q_text = text_embed(args.q)

    try:
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
    except UnexpectedResponse as exc:
        body = (exc.content or b"").decode("utf-8", errors="ignore")
        if "Vector dimension error" in body:
            raise SystemExit(
                "Collection schema does not match current embedding dimensions. "
                "Run `python scripts/01_create_collection.py` and then re-ingest with "
                "`python scripts/02_ingest_synthetic.py --count 2000`."
            ) from exc
        raise SystemExit(
            "Qdrant returned an error response. "
            "Recreate collection if model dimensions changed: `python scripts/01_create_collection.py`."
        ) from exc
    except ResponseHandlingException as exc:
        raise SystemExit(
            "Qdrant query failed. Ensure Qdrant is running and the collection is created. "
            "Try: `docker compose up -d` then `python scripts/01_create_collection.py`."
        ) from exc

    print(f"\nQuery: {args.q}\n")
    for i, r in enumerate(results, start=1):
        p = r["payload"]
        raw_text_score = float((r.get("per_modality") or {}).get("text", r["fused_score"]))
        note_preview = str(p.get("notes", "")).strip().replace("\n", " ")
        if len(note_preview) > 96:
            note_preview = note_preview[:93] + "..."
        print(
            f"{i:02d}. score={raw_text_score:.4f} id={r['id']} "
            f"label={p.get('label')} weather={p.get('weather')} time={p.get('time_of_day')} road={p.get('road_type')} "
            f"notes=\"{note_preview}\""
        )

    # Run a simple novelty check to classify known memory vs new edge case.
    raw_score_results = [
        {
            **r,
            "fused_score": float((r.get("per_modality") or {}).get("text", r["fused_score"])),
        }
        for r in results
    ]
    novel = is_novel_scene(raw_score_results, threshold=0.55)
    print("\nDecision:")
    print("   NEW SCENE / EDGE-CASE (trigger mapping)" if novel else "   KNOWN SCENE (reuse memory)")

    print("")


if __name__ == "__main__":
    main()

