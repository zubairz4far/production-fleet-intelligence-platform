from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(rows: int = 1200, vehicles: int = 60, seed: int = 42) -> pd.DataFrame:
    if rows < 200:
        raise ValueError("rows must be at least 200")
    if vehicles < 10:
        raise ValueError("vehicles must be at least 10")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="6h", tz="UTC")
    vehicle_ids = np.array([f"veh-{i:03d}" for i in range(vehicles)])
    vehicle = vehicle_ids[np.arange(rows) % vehicles]

    age_factor = rng.uniform(0.7, 1.4, size=vehicles)
    vehicle_index = np.arange(rows) % vehicles
    age = age_factor[vehicle_index]
    mileage = 20_000 + np.arange(rows) * 18 + rng.normal(0, 1200, rows) + age * 18_000
    hours_since_service = np.mod(np.arange(rows) * 4 + vehicle_index * 11, 550).astype(float)
    engine_temp = 86 + 0.012 * hours_since_service + 3.5 * age + rng.normal(0, 4.0, rows)
    oil_pressure = 310 - 0.09 * hours_since_service - 10 * age + rng.normal(0, 15, rows)
    battery_voltage = 13.6 - 0.0018 * hours_since_service - 0.25 * age + rng.normal(0, 0.25, rows)
    vibration = 1.7 + 0.0055 * hours_since_service + 0.5 * age + rng.normal(0, 0.45, rows)
    fuel_rate = 7.5 + 0.000035 * mileage + 0.7 * age + rng.normal(0, 0.8, rows)

    # CI fixture only: preserve class imbalance while keeping enough positive events
    # in a chronological holdout for stable probability-metric regression tests.
    logit = (
        -4.5
        + 0.038 * (engine_temp - 90)
        - 0.010 * (oil_pressure - 260)
        - 0.80 * (battery_voltage - 12.7)
        + 0.85 * (vibration - 2.3)
        + 0.0045 * (hours_since_service - 250)
    )
    probability = 1 / (1 + np.exp(-logit))
    failure = rng.binomial(1, np.clip(probability, 0.01, 0.85))

    return pd.DataFrame(
        {
            "vehicle_id": vehicle,
            "timestamp": timestamps.astype(str),
            "mileage_km": np.maximum(mileage, 0).round(1),
            "engine_temp_c": engine_temp.round(2),
            "oil_pressure_kpa": oil_pressure.round(2),
            "battery_voltage": battery_voltage.round(3),
            "vibration_rms": np.maximum(vibration, 0.1).round(3),
            "fuel_rate_lph": np.maximum(fuel_rate, 0.1).round(3),
            "hours_since_service": hours_since_service.round(1),
            "failure_within_horizon": failure,
        }
    )


def inject_telemetry_anomalies(
    frame: pd.DataFrame,
    *,
    seed: int = 73,
    anomaly_rate: float = 0.12,
) -> pd.DataFrame:
    """Inject labeled dev/test anomalies for deterministic CI regression only."""

    if not 0.02 <= anomaly_rate <= 0.30:
        raise ValueError("anomaly_rate must be between 0.02 and 0.30")

    data = frame.copy()
    rng = np.random.default_rng(seed)
    labels = np.zeros(len(data), dtype=int)
    train_end = int(len(data) * 0.60)
    dev_end = int(len(data) * 0.80)

    windows = ((train_end, dev_end), (dev_end, len(data)))
    selected_positions: list[int] = []
    for start, end in windows:
        available = np.arange(start, end)
        count = max(8, int(round(len(available) * anomaly_rate)))
        selected = rng.choice(available, size=min(count, len(available)), replace=False)
        selected_positions.extend(int(position) for position in selected)

    selected_positions.sort()
    for offset, position in enumerate(selected_positions):
        labels[position] = 1
        anomaly_type = offset % 5
        severity = rng.uniform(0.85, 1.15)

        if anomaly_type == 0:
            data.loc[position, "engine_temp_c"] += 16.0 * severity
            data.loc[position, "vibration_rms"] += 1.6 * severity
        elif anomaly_type == 1:
            data.loc[position, "oil_pressure_kpa"] -= 82.0 * severity
            data.loc[position, "engine_temp_c"] += 8.0 * severity
        elif anomaly_type == 2:
            data.loc[position, "battery_voltage"] -= 1.65 * severity
            data.loc[position, "fuel_rate_lph"] += 2.1 * severity
        elif anomaly_type == 3:
            data.loc[position, "vibration_rms"] += 2.7 * severity
        else:
            data.loc[position, "fuel_rate_lph"] += 3.4 * severity
            data.loc[position, "oil_pressure_kpa"] -= 42.0 * severity

    data["vibration_rms"] = np.maximum(data["vibration_rms"], 0.1)
    data["oil_pressure_kpa"] = np.maximum(data["oil_pressure_kpa"], 20.0)
    data["battery_voltage"] = np.maximum(data["battery_voltage"], 8.0)
    data["fuel_rate_lph"] = np.maximum(data["fuel_rate_lph"], 0.1)
    data["telemetry_anomaly"] = labels
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_fleet.csv")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--vehicles", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-anomalies", action="store_true")
    parser.add_argument("--anomaly-rate", type=float, default=0.12)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(rows=args.rows, vehicles=args.vehicles, seed=args.seed)
    if args.with_anomalies:
        frame = inject_telemetry_anomalies(
            frame,
            seed=args.seed + 31,
            anomaly_rate=args.anomaly_rate,
        )

    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")
    print(f"positive_rate={frame['failure_within_horizon'].mean():.4f}")
    if "telemetry_anomaly" in frame.columns:
        print(f"anomaly_rate={frame['telemetry_anomaly'].mean():.4f}")


if __name__ == "__main__":
    main()
