from av_memory.schema import get_client, recreate_collection, ensure_payload_indexes
from av_memory.config import SETTINGS


def main() -> None:
    # Recreate the collection to keep demo runs deterministic.
    # This avoids schema drift during rapid iteration.
    client = get_client()
    recreate_collection(client, SETTINGS.collection)
    # Create payload indexes immediately to keep filtered queries fast.
    ensure_payload_indexes(client, SETTINGS.collection)
    print(f"✅ Collection ready: {SETTINGS.collection}")


if __name__ == "__main__":
    main()

