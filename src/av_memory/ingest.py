import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import SETTINGS
from .embeddings import vision_embed, lidar_embed, radar_embed, text_embed


WEATHERS = ["clear", "rain", "fog", "snow", "overcast"]
TIMES = ["day", "dusk", "night"]
ROAD_TYPES = ["city", "highway", "residential", "intersection"]
SECONDS_PER_MONTH = 60 * 60 * 24 * 30
LOOKBACK_MONTHS = 14


@dataclass
class Scenario:
    sid: int
    ts: int
    lat: float
    lon: float
    location_bucket: str
    weather: str
    time_of_day: str
    road_type: str
    near_miss: bool
    label: str
    notes: str

    # I keep raw modality payloads here so I can derive vectors at ingest time.
    image: Image.Image
    lidar_points: np.ndarray
    radar_signal: np.ndarray


def _bucket_location(lat: float, lon: float, step: float = 0.01) -> str:
    # I keep geobucketing coarse on purpose because I only need
    # location-level filtering here, not precise geo-search.
    bl = math.floor(lat / step) * step
    bo = math.floor(lon / step) * step
    return f"{bl:.2f},{bo:.2f}"


def _make_synthetic_frame(label: str, time_of_day: str, weather: str, seed: int) -> Image.Image:
    """
    I generate a simple synthetic camera frame with rough visual cues.
    I do not aim for realism here; I only need consistent patterns that
    the lightweight embedder can learn and retrieve.
    """
    rng = random.Random(seed)

    w, h = 256, 256
    bg = (30, 30, 30) if time_of_day == "night" else (190, 210, 220)
    if weather == "fog":
        bg = (150, 150, 150)
    if weather == "rain":
        bg = (120, 140, 160)

    img = Image.new("RGB", (w, h), bg)
    dr = ImageDraw.Draw(img)

    # I draw a road strip first so every frame has a shared baseline structure.
    dr.rectangle([0, int(h * 0.65), w, h], fill=(70, 70, 70))

    # I add lane markers so the frame has some repeated geometry.
    for x in range(0, w, 40):
        dr.rectangle([x, int(h * 0.8), x + 20, int(h * 0.82)], fill=(230, 230, 60))

    # I overlay event-specific shapes to create class-dependent signals.
    if "pedestrian" in label:
        x = rng.randint(40, 200)
        dr.ellipse([x, 120, x + 25, 145], fill=(250, 220, 200))  # I use a tiny blob as a head proxy.
        dr.rectangle([x + 10, 145, x + 15, 185], fill=(250, 220, 200))  # I keep body shape minimal.

    if "near_miss" in label:
        # I use a bright red blob to represent sudden-risk context.
        dr.ellipse([170, 40, 240, 110], fill=(220, 40, 40))

    if "slippery" in label:
        # I paint a blue puddle-like region for slippery-road scenes.
        dr.ellipse([60, 185, 160, 235], fill=(60, 120, 220))

    if time_of_day == "night":
        # I add a simple headlight glow for nighttime bias.
        dr.ellipse([10, 160, 90, 240], fill=(240, 240, 180))

    # I sprinkle rain streaks so weather changes are not only metadata labels.
    if weather == "rain":
        for _ in range(25):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            dr.line([x, y, x + 6, y + 12], fill=(200, 210, 230), width=1)

    return img


def _make_synthetic_lidar(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # I start from a noisy road-plane cloud to represent the common background.
    n = 800
    x = rng.normal(0, 10, size=n)
    y = rng.normal(0, 2, size=n)
    z = rng.normal(0, 0.2, size=n)  # I keep z near 0 so most points sit on ground plane.
    pts = np.stack([x, y, z], axis=1)

    # I then inject event-specific point clusters so each class gets distinct geometry.
    if "pedestrian" in label:
        c = rng.normal([5, 0, 1.2], [0.4, 0.4, 0.3], size=(180, 3))
        pts = np.concatenate([pts, c], axis=0)

    if "near_miss" in label:
        c = rng.normal([2, -1.5, 1.5], [0.6, 0.3, 0.5], size=(220, 3))
        pts = np.concatenate([pts, c], axis=0)

    if "slippery" in label:
        # I mimic a flatter reflective patch with tighter z spread.
        c = rng.normal([0, 2, 0.05], [3.0, 0.6, 0.05], size=(200, 3))
        pts = np.concatenate([pts, c], axis=0)

    return pts.astype(np.float32)


def _make_synthetic_radar(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, 512, dtype=np.float32)
    base = 0.05 * rng.normal(size=t.shape).astype(np.float32)

    # I add class-specific frequency components so radar embeddings separate better.
    s = base
    if "pedestrian" in label:
        s = s + 0.3 * np.sin(2 * np.pi * 18 * t).astype(np.float32)
    if "near_miss" in label:
        s = s + 0.4 * np.sin(2 * np.pi * 40 * t).astype(np.float32)
    if "slippery" in label:
        s = s + 0.25 * np.sin(2 * np.pi * 8 * t).astype(np.float32)

    return s.astype(np.float32)


def _build_notes(edge_type: str, weather: str, time_of_day: str, road_type: str) -> str:
    notes = f"{edge_type} | weather={weather} | time={time_of_day} | road={road_type}"
    if "pedestrian" in edge_type:
        notes += " - pedestrian detected crossing"
    if "slippery" in edge_type:
        notes += " - low traction, possible hydroplaning"
    if "near_miss" in edge_type:
        notes += " - close call with cut-in vehicle"
    return notes


def make_scenario(i: int, now_ts: int, seed: int) -> Scenario:
    rng = random.Random(seed + i)

    # I sample around one city center so location filters remain meaningful.
    lat = 12.9716 + rng.uniform(-0.08, 0.08)
    lon = 77.5946 + rng.uniform(-0.08, 0.08)
    location_bucket = _bucket_location(lat, lon)

    weather = rng.choice(WEATHERS)
    time_of_day = rng.choice(TIMES)
    road_type = rng.choice(ROAD_TYPES)

    # I keep normal driving more frequent, but I still sample enough edge-cases.
    edge_types = ["pedestrian_low_light", "slippery_road", "near_miss_cut_in", "normal_drive"]
    et = rng.choices(edge_types, weights=[0.25, 0.20, 0.20, 0.35], k=1)[0]

    near_miss = et.startswith("near_miss") or (et == "pedestrian_low_light" and time_of_day == "night")

    label = et
    notes = _build_notes(et, weather, time_of_day, road_type)

    # I spread timestamps across ~14 months so time-window retrieval is testable.
    ts = now_ts - rng.randint(0, SECONDS_PER_MONTH * LOOKBACK_MONTHS)

    img = _make_synthetic_frame(label, time_of_day, weather, seed=seed + i * 7)
    lidar = _make_synthetic_lidar(label, seed=seed + i * 11)
    radar = _make_synthetic_radar(label, seed=seed + i * 13)

    return Scenario(
        sid=i,
        ts=ts,
        lat=lat,
        lon=lon,
        location_bucket=location_bucket,
        weather=weather,
        time_of_day=time_of_day,
        road_type=road_type,
        near_miss=near_miss,
        label=label,
        notes=notes,
        image=img,
        lidar_points=lidar,
        radar_signal=radar,
    )


def scenario_to_point(s: Scenario) -> qm.PointStruct:
    # I build one vector per modality and store everything as named vectors.
    # I rely on this so I can query each modality independently and fuse later.
    v_vision = vision_embed(s.image)
    v_lidar = lidar_embed(s.lidar_points)
    v_radar = radar_embed(s.radar_signal)
    v_text = text_embed(s.notes)

    payload: dict[str, Any] = {
        "sid": f"scn_{s.sid:07d}",
        "ts": s.ts,
        "lat": s.lat,
        "lon": s.lon,
        "location_bucket": s.location_bucket,
        "weather": s.weather,
        "time_of_day": s.time_of_day,
        "road_type": s.road_type,
        "near_miss": s.near_miss,
        "label": s.label,
        "notes": s.notes,
    }

    return qm.PointStruct(
        id=s.sid,
        vector={
            "vision": v_vision,
            "lidar": v_lidar,
            "radar": v_radar,
            "text": v_text,
        },
        payload=payload,
    )


def ingest_scenarios(
    client: QdrantClient,
    count: int,
    batch_size: int = 128,
    seed: int = 42,
    collection_name: str | None = None,
) -> None:
    if count <= 0:
        return
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    name = collection_name or SETTINGS.collection
    now_ts = int(time.time())

    # I buffer points and upsert in batches because single-point writes are slower.
    buf: list[qm.PointStruct] = []
    for i in range(count):
        s = make_scenario(i=i, now_ts=now_ts, seed=seed)
        buf.append(scenario_to_point(s))

        if len(buf) >= batch_size:
            client.upsert(collection_name=name, points=buf)
            buf.clear()

    if buf:
        client.upsert(collection_name=name, points=buf)
