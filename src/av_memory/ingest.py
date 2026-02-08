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


@dataclass
class Scenario:
    sid: str
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

    # raw modality data
    image: Image.Image
    lidar_points: np.ndarray
    radar_signal: np.ndarray


def _bucket_location(lat: float, lon: float, step: float = 0.01) -> str:
    # rough geobucket, good enough for payload filtering
    bl = math.floor(lat / step) * step
    bo = math.floor(lon / step) * step
    return f"{bl:.2f},{bo:.2f}"


def _make_synthetic_frame(label: str, time_of_day: str, weather: str, seed: int) -> Image.Image:
    """
    Creates a simple synthetic "camera frame".
    We draw icons: pedestrian, car, slippery, etc.
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

    # road
    dr.rectangle([0, int(h * 0.65), w, h], fill=(70, 70, 70))

    # lane lines
    for x in range(0, w, 40):
        dr.rectangle([x, int(h * 0.8), x + 20, int(h * 0.82)], fill=(230, 230, 60))

    # draw event-specific objects
    if "pedestrian" in label:
        x = rng.randint(40, 200)
        dr.ellipse([x, 120, x + 25, 145], fill=(250, 220, 200))  # head
        dr.rectangle([x + 10, 145, x + 15, 185], fill=(250, 220, 200))  # body-ish

    if "near_miss" in label:
        # red warning blob
        dr.ellipse([170, 40, 240, 110], fill=(220, 40, 40))

    if "slippery" in label:
        # blue puddle
        dr.ellipse([60, 185, 160, 235], fill=(60, 120, 220))

    if time_of_day == "night":
        # headlight glow
        dr.ellipse([10, 160, 90, 240], fill=(240, 240, 180))

    # add a few rain streaks
    if weather == "rain":
        for _ in range(25):
            x = rng.randint(0, w)
            y = rng.randint(0, h)
            dr.line([x, y, x + 6, y + 12], fill=(200, 210, 230), width=1)

    return img


def _make_synthetic_lidar(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # base "road plane"
    n = 800
    x = rng.normal(0, 10, size=n)
    y = rng.normal(0, 2, size=n)
    z = rng.normal(0, 0.2, size=n)  # near ground
    pts = np.stack([x, y, z], axis=1)

    # add cluster for pedestrian / obstacle
    if "pedestrian" in label:
        c = rng.normal([5, 0, 1.2], [0.4, 0.4, 0.3], size=(180, 3))
        pts = np.concatenate([pts, c], axis=0)

    if "near_miss" in label:
        c = rng.normal([2, -1.5, 1.5], [0.6, 0.3, 0.5], size=(220, 3))
        pts = np.concatenate([pts, c], axis=0)

    if "slippery" in label:
        # "flat reflective area" -> slightly different z distribution
        c = rng.normal([0, 2, 0.05], [3.0, 0.6, 0.05], size=(200, 3))
        pts = np.concatenate([pts, c], axis=0)

    return pts.astype(np.float32)


def _make_synthetic_radar(label: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, 512, dtype=np.float32)
    base = 0.05 * rng.normal(size=t.shape).astype(np.float32)

    # add frequency patterns per label
    s = base
    if "pedestrian" in label:
        s = s + 0.3 * np.sin(2 * np.pi * 18 * t).astype(np.float32)
    if "near_miss" in label:
        s = s + 0.4 * np.sin(2 * np.pi * 40 * t).astype(np.float32)
    if "slippery" in label:
        s = s + 0.25 * np.sin(2 * np.pi * 8 * t).astype(np.float32)

    return s.astype(np.float32)


def make_scenario(i: int, now_ts: int, seed: int) -> Scenario:
    rng = random.Random(seed + i)

    # random-ish coords around some city-ish center
    lat = 12.9716 + rng.uniform(-0.08, 0.08)
    lon = 77.5946 + rng.uniform(-0.08, 0.08)
    location_bucket = _bucket_location(lat, lon)

    weather = rng.choice(WEATHERS)
    time_of_day = rng.choice(TIMES)
    road_type = rng.choice(ROAD_TYPES)

    # Create labels with a few edge-case types
    edge_types = ["pedestrian_low_light", "slippery_road", "near_miss_cut_in", "normal_drive"]
    et = rng.choices(edge_types, weights=[0.25, 0.20, 0.20, 0.35], k=1)[0]

    near_miss = et.startswith("near_miss") or (et == "pedestrian_low_light" and time_of_day == "night")

    label = et
    notes = f"{et} | weather={weather} | time={time_of_day} | road={road_type}"
    if "pedestrian" in et:
        notes += " - pedestrian detected crossing"
    if "slippery" in et:
        notes += " - low traction, possible hydroplaning"
    if "near_miss" in et:
        notes += " - close call with cut-in vehicle"

    # Timestamp spread into last 14 months
    ts = now_ts - rng.randint(0, 60 * 60 * 24 * 30 * 14)

    img = _make_synthetic_frame(label, time_of_day, weather, seed=seed + i * 7)
    lidar = _make_synthetic_lidar(label, seed=seed + i * 11)
    radar = _make_synthetic_radar(label, seed=seed + i * 13)

    return Scenario(
        sid=f"scn_{i:07d}",
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
    v_vision = vision_embed(s.image)
    v_lidar = lidar_embed(s.lidar_points)
    v_radar = radar_embed(s.radar_signal)
    v_text = text_embed(s.notes)

    payload: dict[str, Any] = {
        "sid": s.sid,
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
    name = collection_name or SETTINGS.collection
    now_ts = int(time.time())

    buf: list[qm.PointStruct] = []
    for i in range(count):
        s = make_scenario(i=i, now_ts=now_ts, seed=seed)
        buf.append(scenario_to_point(s))

        if len(buf) >= batch_size:
            client.upsert(collection_name=name, points=buf)
            buf.clear()

    if buf:
        client.upsert(collection_name=name, points=buf)
