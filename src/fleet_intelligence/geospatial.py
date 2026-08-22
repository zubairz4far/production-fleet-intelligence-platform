from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088
ETA_TARGET = "actual_travel_minutes"
ETA_REQUIRED_COLUMNS = (
    "trip_id",
    "timestamp",
    "origin_lat",
    "origin_lon",
    "destination_lat",
    "destination_lon",
    "planned_distance_km",
    "traffic_index",
    "weather_severity",
    "vehicle_load_kg",
    "stops_remaining",
    ETA_TARGET,
)
ETA_FEATURE_COLUMNS = (
    "planned_distance_km",
    "great_circle_km",
    "detour_ratio",
    "traffic_index",
    "weather_severity",
    "vehicle_load_kg",
    "stops_remaining",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "bearing_sin",
    "bearing_cos",
)


@dataclass(frozen=True)
class EtaDatasetSummary:
    rows: int
    first_timestamp: str
    last_timestamp: str
    mean_distance_km: float
    mean_travel_minutes: float


def haversine_km(
    lat1: np.ndarray | pd.Series | float,
    lon1: np.ndarray | pd.Series | float,
    lat2: np.ndarray | pd.Series | float,
    lon2: np.ndarray | pd.Series | float,
) -> np.ndarray:
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lon2_rad = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_radians(
    lat1: np.ndarray | pd.Series,
    lon1: np.ndarray | pd.Series,
    lat2: np.ndarray | pd.Series,
    lon2: np.ndarray | pd.Series,
) -> np.ndarray:
    lat1_rad = np.radians(np.asarray(lat1, dtype=float))
    lon1_rad = np.radians(np.asarray(lon1, dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lon2_rad = np.radians(np.asarray(lon2, dtype=float))
    dlon = lon2_rad - lon1_rad

    y = np.sin(dlon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(dlon)
    return np.arctan2(y, x)


def validate_eta_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ETA_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing ETA-dataset columns: {missing}")

    data = frame.copy()
    data["trip_id"] = data["trip_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")

    numeric = [column for column in ETA_REQUIRED_COLUMNS if column not in {"trip_id", "timestamp"}]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="raise")

    for column in ("origin_lat", "destination_lat"):
        if not data[column].between(-90.0, 90.0).all():
            raise ValueError(f"{column} must be between -90 and 90")
    for column in ("origin_lon", "destination_lon"):
        if not data[column].between(-180.0, 180.0).all():
            raise ValueError(f"{column} must be between -180 and 180")

    if (data["planned_distance_km"] <= 0).any():
        raise ValueError("planned_distance_km must be positive")
    if (data[ETA_TARGET] <= 0).any():
        raise ValueError(f"{ETA_TARGET} must be positive")
    if (data["traffic_index"] <= 0).any():
        raise ValueError("traffic_index must be positive")
    if not data["weather_severity"].between(0.0, 1.0).all():
        raise ValueError("weather_severity must be in [0, 1]")
    if (data["vehicle_load_kg"] < 0).any() or (data["stops_remaining"] < 0).any():
        raise ValueError("vehicle_load_kg and stops_remaining must be non-negative")

    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


def build_eta_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = validate_eta_dataset(frame)
    great_circle = haversine_km(
        data["origin_lat"],
        data["origin_lon"],
        data["destination_lat"],
        data["destination_lon"],
    )
    great_circle = np.maximum(great_circle, 0.05)
    bearing = initial_bearing_radians(
        data["origin_lat"],
        data["origin_lon"],
        data["destination_lat"],
        data["destination_lon"],
    )
    hour = data["timestamp"].dt.hour + data["timestamp"].dt.minute / 60.0
    dow = data["timestamp"].dt.dayofweek

    features = pd.DataFrame(
        {
            "planned_distance_km": data["planned_distance_km"],
            "great_circle_km": great_circle,
            "detour_ratio": data["planned_distance_km"] / great_circle,
            "traffic_index": data["traffic_index"],
            "weather_severity": data["weather_severity"],
            "vehicle_load_kg": data["vehicle_load_kg"],
            "stops_remaining": data["stops_remaining"],
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "dow_sin": np.sin(2.0 * np.pi * dow / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * dow / 7.0),
            "bearing_sin": np.sin(bearing),
            "bearing_cos": np.cos(bearing),
        }
    )
    output = pd.concat(
        [data[["trip_id", "timestamp", ETA_TARGET]], features],
        axis=1,
    )
    return output.sort_values("timestamp", kind="stable").reset_index(drop=True)


def summarize_eta_dataset(frame: pd.DataFrame) -> EtaDatasetSummary:
    data = validate_eta_dataset(frame)
    return EtaDatasetSummary(
        rows=len(data),
        first_timestamp=data["timestamp"].min().isoformat(),
        last_timestamp=data["timestamp"].max().isoformat(),
        mean_distance_km=float(data["planned_distance_km"].mean()),
        mean_travel_minutes=float(data[ETA_TARGET].mean()),
    )
