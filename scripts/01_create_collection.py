from av_memory.schema import get_client, recreate_collection, ensure_payload_indexes
from av_memory.config import SETTINGS


def main() -> None:
    # I recreate the collection when I want a clean demo state.
    # I do this to avoid schema drift while I keep iterating.
    client = get_client()
    recreate_collection(client, SETTINGS.collection)
    # I add payload indexes right away so filtered search is fast from the start.
    ensure_payload_indexes(client, SETTINGS.collection)
    print(f"✅ Collection ready: {SETTINGS.collection}")


if __name__ == "__main__":
    main()
