from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import SETTINGS


def get_client() -> QdrantClient:
    # Centralize client creation so all scripts use the same endpoint/config.
    return QdrantClient(url=SETTINGS.qdrant_url)


def recreate_collection(client: QdrantClient, collection_name: str | None = None) -> None:
    name = collection_name or SETTINGS.collection

    # Prefer clean reset over migration logic for this demo.
    # Deterministic schema recreation is simpler during rapid iteration.
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        client.delete_collection(collection_name=name)

    vectors_config = {
        "vision": qm.VectorParams(size=SETTINGS.vision_dim, distance=qm.Distance.COSINE),
        "lidar": qm.VectorParams(size=SETTINGS.lidar_dim, distance=qm.Distance.COSINE),
        "radar": qm.VectorParams(size=SETTINGS.radar_dim, distance=qm.Distance.COSINE),
        "text": qm.VectorParams(size=SETTINGS.text_dim, distance=qm.Distance.COSINE),
    }

    # Keep default HNSW settings until query scale justifies tuning.
    client.create_collection(
        collection_name=name,
        vectors_config=vectors_config,
        optimizers_config=qm.OptimizersConfigDiff(
            indexing_threshold=2000,
        ),
    )


def ensure_payload_indexes(client: QdrantClient, collection_name: str | None = None) -> None:
    """
    Index payload fields we filter on often.
    (Not mandatory, but speedups are real when data grows)
    """
    name = collection_name or SETTINGS.collection

    # Index frequently filtered fields to keep filtered retrieval fast.
    for field in ["weather", "time_of_day", "road_type", "location_bucket"]:
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )

    # Index timestamp separately because most demos slice by "last N months".
    client.create_payload_index(
        collection_name=name,
        field_name="ts",
        field_schema=qm.PayloadSchemaType.INTEGER,
    )

