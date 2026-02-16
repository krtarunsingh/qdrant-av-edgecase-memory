from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Local-first defaults keep the demo runnable with only Docker Compose.
    qdrant_url: str = "http://localhost:6333"
    collection: str = "av_edgecase_memory"

    # Vector dimensions are schema contracts.
    # After collection creation, change these only when recreating the collection.
    # Defaults correspond to:
    # - vision: openai/clip-vit-base-patch32 (512)
    # - lidar: PointNet backbone global feature (1024)
    # - radar: VyDat radar model penultimate feature (64)
    # - text: sentence-transformers/all-MiniLM-L6-v2 (384)
    vision_dim: int = 512
    lidar_dim: int = 1024
    radar_dim: int = 64
    text_dim: int = 384


# Shared settings instance keeps scripts and library code aligned.
SETTINGS = Settings()

