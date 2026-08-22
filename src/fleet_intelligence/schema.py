from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET = "failure_within_horizon"
ID_COLUMNS = ("vehicle_id", "timestamp")
FEATURE_COLUMNS = (
    "mileage_km",
    "engine_temp_c",
    "oil_pressure_kpa",
    "battery_voltage",
    "vibration_rms",
    "fuel_rate_lph",
    "hours_since_service",
)
REQUIRED_COLUMNS = (*ID_COLUMNS, *FEATURE_COLUMNS, TARGET)


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    vehicles: int
    positive_rate: float
    first_timestamp: str
    last_timestamp: str


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = frame.copy()
    data["vehicle_id"] = data["vehicle_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must contain only 0/1 labels")

    if data[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Feature columns must not contain missing values in v0.1")

    if data["vehicle_id"].str.len().eq(0).any():
        raise ValueError("vehicle_id must not be empty")

    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


def summarize_dataset(data: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        rows=len(data),
        vehicles=data["vehicle_id"].nunique(),
        positive_rate=float(data[TARGET].mean()),
        first_timestamp=data["timestamp"].min().isoformat(),
        last_timestamp=data["timestamp"].max().isoformat(),
    )
