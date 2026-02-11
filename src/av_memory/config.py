from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Local-first defaults keep the demo runnable with only Docker Compose.
    qdrant_url: str = "http://localhost:6333"
    collection: str = "av_edgecase_memory"

    # Vector dimensions are schema contracts.
    # After collection creation, change these only when recreating the collection.
    vision_dim: int = 256
    lidar_dim: int = 128
    radar_dim: int = 128
    text_dim: int = 256


# Shared settings instance keeps scripts and library code aligned.
SETTINGS = Settings()

