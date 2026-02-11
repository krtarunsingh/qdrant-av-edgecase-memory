from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # I keep these defaults local-first so I can run the demo with just docker compose.
    qdrant_url: str = "http://localhost:6333"
    collection: str = "av_edgecase_memory"

    # I treat vector dimensions as schema contracts; once the collection exists,
    # I should avoid changing these unless I recreate the collection.
    vision_dim: int = 256
    lidar_dim: int = 128
    radar_dim: int = 128
    text_dim: int = 256


# I import one shared settings object everywhere so scripts and library code stay aligned.
SETTINGS = Settings()
