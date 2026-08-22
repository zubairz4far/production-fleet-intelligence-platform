from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS

ANOMALY_TARGET = "telemetry_anomaly"
TELEMETRY_SIGNALS = (
    "engine_temp_c",
    "oil_pressure_kpa",
    "battery_voltage",
    "vibration_rms",
    "fuel_rate_lph",
)
CONTEXT_FEATURES = ("mileage_km", "hours_since_service")
ANOMALY_REQUIRED_COLUMNS = ("vehicle_id", "timestamp", *FEATURE_COLUMNS, ANOMALY_TARGET)


@dataclass(frozen=True)
class AnomalyDatasetSummary:
    rows: int
    vehicles: int
    anomaly_rate: float
    first_timestamp: str
    last_timestamp: str


def validate_anomaly_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ANOMALY_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing anomaly-dataset columns: {missing}")

    data = frame.copy()
    data["vehicle_id"] = data["vehicle_id"].astype(str)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="raise")

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[ANOMALY_TARGET] = pd.to_numeric(data[ANOMALY_TARGET], errors="raise").astype(int)
    if not set(data[ANOMALY_TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{ANOMALY_TARGET} must contain only 0/1 labels")
    if data["vehicle_id"].str.len().eq(0).any():
        raise ValueError("vehicle_id must not be empty")

    return data.sort_values("timestamp", kind="stable").reset_index(drop=True)


def anomaly_feature_columns() -> tuple[str, ...]:
    columns: list[str] = [*CONTEXT_FEATURES]
    for signal in TELEMETRY_SIGNALS:
        columns.extend(
            [
                signal,
                f"{signal}_lag1",
                f"{signal}_past_mean_3",
                f"{signal}_past_std_3",
                f"{signal}_delta_from_past3",
                f"{signal}_past_mean_6",
                f"{signal}_past_std_6",
            ]
        )
    return tuple(columns)


def build_past_only_telemetry_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build sequential telemetry features without peeking at future observations.

    Rolling statistics are computed after ``shift(1)`` within each vehicle, so the
    historical summaries for an observation contain only earlier observations.
    The current sensor value is still available because real-time anomaly detection
    must score the observation that just arrived.
    """

    data = validate_anomaly_dataset(frame)
    ordered = data.sort_values(["vehicle_id", "timestamp"], kind="stable").copy()
    features = pd.DataFrame(index=ordered.index)

    for column in CONTEXT_FEATURES:
        features[column] = ordered[column]

    grouped = ordered.groupby("vehicle_id", sort=False)
    for signal in TELEMETRY_SIGNALS:
        current = ordered[signal]
        previous = grouped[signal].shift(1)
        past_mean_3 = grouped[signal].transform(
            lambda values: values.shift(1).rolling(window=3, min_periods=1).mean()
        )
        past_std_3 = grouped[signal].transform(
            lambda values: values.shift(1).rolling(window=3, min_periods=2).std(ddof=0)
        )
        past_mean_6 = grouped[signal].transform(
            lambda values: values.shift(1).rolling(window=6, min_periods=1).mean()
        )
        past_std_6 = grouped[signal].transform(
            lambda values: values.shift(1).rolling(window=6, min_periods=2).std(ddof=0)
        )

        features[signal] = current
        features[f"{signal}_lag1"] = previous
        features[f"{signal}_past_mean_3"] = past_mean_3
        features[f"{signal}_past_std_3"] = past_std_3
        features[f"{signal}_delta_from_past3"] = current - past_mean_3
        features[f"{signal}_past_mean_6"] = past_mean_6
        features[f"{signal}_past_std_6"] = past_std_6

    features = features.replace([np.inf, -np.inf], np.nan)
    output = pd.concat(
        [ordered[["vehicle_id", "timestamp", ANOMALY_TARGET]], features],
        axis=1,
    )
    return output.sort_values("timestamp", kind="stable").reset_index(drop=True)


def summarize_anomaly_dataset(data: pd.DataFrame) -> AnomalyDatasetSummary:
    validated = validate_anomaly_dataset(data)
    return AnomalyDatasetSummary(
        rows=len(validated),
        vehicles=validated["vehicle_id"].nunique(),
        anomaly_rate=float(validated[ANOMALY_TARGET].mean()),
        first_timestamp=validated["timestamp"].min().isoformat(),
        last_timestamp=validated["timestamp"].max().isoformat(),
    )


def population_stability_index(
    reference: np.ndarray | pd.Series,
    current: np.ndarray | pd.Series,
    *,
    bins: int = 10,
) -> float:
    reference_values = np.asarray(reference, dtype=float)
    current_values = np.asarray(current, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    current_values = current_values[np.isfinite(current_values)]

    if len(reference_values) < bins or len(current_values) < bins:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference_values, quantiles))
    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_counts, _ = np.histogram(reference_values, bins=edges)
    current_counts, _ = np.histogram(current_values, bins=edges)

    epsilon = 1e-6
    reference_share = np.clip(reference_counts / reference_counts.sum(), epsilon, None)
    current_share = np.clip(current_counts / current_counts.sum(), epsilon, None)
    ratio = current_share / reference_share
    return float(np.sum((current_share - reference_share) * np.log(ratio)))


def simulate_sensor_drift(frame: pd.DataFrame) -> pd.DataFrame:
    drifted = frame.copy()
    drifted["engine_temp_c"] = drifted["engine_temp_c"] + 9.0
    drifted["vibration_rms"] = drifted["vibration_rms"] * 1.45
    drifted["battery_voltage"] = drifted["battery_voltage"] - 0.55
    drifted["oil_pressure_kpa"] = drifted["oil_pressure_kpa"] - 28.0
    return drifted
