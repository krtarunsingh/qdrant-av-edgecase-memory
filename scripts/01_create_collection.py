from av_memory.schema import get_client, recreate_collection, ensure_payload_indexes
from av_memory.config import SETTINGS


def main() -> None:
    client = get_client()
    recreate_collection(client, SETTINGS.collection)
    ensure_payload_indexes(client, SETTINGS.collection)
    print(f"✅ Collection ready: {SETTINGS.collection}")


if __name__ == "__main__":
    main()
