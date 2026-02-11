import argparse

from av_memory.schema import get_client
from av_memory.ingest import ingest_scenarios
from av_memory.config import SETTINGS


def main() -> None:
    # Expose CLI flags to control dataset size and reproducibility.
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Reuse the shared ingestion path so script and library behavior match.
    client = get_client()
    ingest_scenarios(client, count=args.count, batch_size=args.batch, seed=args.seed, collection_name=SETTINGS.collection)

    # Print final collection size as a post-ingest sanity check.
    info = client.get_collection(SETTINGS.collection)
    print(f"✅ Ingested: {args.count} scenarios")
    print(f"   Points now: {info.points_count}")


if __name__ == "__main__":
    main()

