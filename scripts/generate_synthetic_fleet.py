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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic_fleet.csv")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--vehicles", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = generate(rows=args.rows, vehicles=args.vehicles, seed=args.seed)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} rows to {output}")
    print(f"positive_rate={frame['failure_within_horizon'].mean():.4f}")


if __name__ == "__main__":
    main()
