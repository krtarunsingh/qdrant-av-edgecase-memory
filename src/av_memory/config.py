from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    collection: str = "av_edgecase_memory"

    # Named vector sizes (keep stable once created)
    vision_dim: int = 256
    lidar_dim: int = 128
    radar_dim: int = 128
    text_dim: int = 256


SETTINGS = Settings()
