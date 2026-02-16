# qdrant-av-edgecase-memory

Multi-modal edge-case memory for autonomous vehicles using Qdrant named vectors.

Each scenario stores one vector per modality:
- `vision`: CLIP image embedding (`openai/clip-vit-base-patch32`)
- `lidar`: PointNet backbone embedding (pretrained checkpoint from `nanopiero/pointnet_igloos`)
- `radar`: trained radar TorchScript embedding (`VyDat/Radar_Signal-Classification`)
- `text`: SentenceTransformer embedding (`sentence-transformers/all-MiniLM-L6-v2`)

## Vector dimensions

The collection schema is configured in `src/av_memory/config.py`:
- `vision_dim=512`
- `lidar_dim=1024`
- `radar_dim=64`
- `text_dim=384`

If you change models, update dimensions and recreate the collection.

## Quick start

1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

2. Run Qdrant

```bash
docker compose up -d
```

3. Recreate collection (required after model/dimension changes)

```bash
python scripts/01_create_collection.py
```

4. Ingest synthetic scenarios

```bash
python scripts/02_ingest_synthetic.py --count 2000 --batch 128 --seed 42
```

5. Query examples

```bash
python scripts/03_query_examples.py
python scripts/04_text_query.py --q "pedestrian crossing low light" --time_of_day night
python scripts/04_text_query.py --q "slippery road rain dusk"
```

## Optional model overrides

Use environment variables to switch model IDs/checkpoints:
- `AV_VISION_MODEL_ID` (default: `openai/clip-vit-base-patch32`)
- `AV_TEXT_MODEL_ID` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `AV_LIDAR_POINTNET_REPO` (default: `nanopiero/pointnet_igloos`)
- `AV_LIDAR_POINTNET_FILE` (default: `pointnet_500_ep.pth`)
- `AV_RADAR_MODEL_REPO` (default: `VyDat/Radar_Signal-Classification`)
- `AV_RADAR_MODEL_FILE` (default: `22139012_22139015.pt`)
