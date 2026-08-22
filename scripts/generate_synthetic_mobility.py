from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fleet_intelligence.geospatial import haversine_km


def generate_eta_trips(rows: int = 1600, seed: int = 84) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("rows must be at least 500")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=rows, freq="2h", tz="UTC")
    hours = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()

    center_lat = 31.5204
    center_lon = 74.3587
    origin_lat = center_lat + rng.normal(0.0, 0.055, rows)
    origin_lon = center_lon + rng.normal(0.0, 0.060, rows)
    destination_lat = center_lat + rng.normal(0.0, 0.070, rows)
    destination_lon = center_lon + rng.normal(0.0, 0.075, rows)

    great_circle = haversine_km(origin_lat, origin_lon, destination_lat, destination_lon)
    planned_distance = np.maximum(great_circle * rng.uniform(1.12, 1.42, rows) + 0.8, 1.0)

    morning_rush = ((hours >= 7) & (hours <= 10)).astype(float)
    evening_rush = ((hours >= 16) & (hours <= 20)).astype(float)
    weekday = (day_of_week < 5).astype(float)
    rush = np.maximum(morning_rush, evening_rush) * weekday
    traffic_index = np.clip(
        0.82 + 0.72 * rush + 0.12 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 0.10, rows),
        0.60,
        2.20,
    )

    weather = np.clip(rng.beta(1.7, 6.0, rows) + rng.binomial(1, 0.08, rows) * 0.45, 0.0, 1.0)
    vehicle_load = rng.uniform(150.0, 1800.0, rows)
    stops_remaining = rng.integers(1, 13, rows)

    effective_speed = 46.0 / (traffic_index * (1.0 + 0.48 * weather))
    load_penalty = 0.0012 * vehicle_load
    stop_penalty = 0.62 * stops_remaining
    nonlinear_congestion = 2.8 * np.maximum(traffic_index - 1.25, 0.0) ** 2
    noise = rng.normal(0.0, 2.2 + 0.025 * planned_distance, rows)
    travel_minutes = (
        planned_distance / np.maximum(effective_speed, 8.0) * 60.0
        + load_penalty
        + stop_penalty
        + nonlinear_congestion
        + noise
    )
    travel_minutes = np.maximum(travel_minutes, 3.0)

    return pd.DataFrame(
        {
            "trip_id": [f"trip-{i:05d}" for i in range(rows)],
            "timestamp": timestamps.astype(str),
            "origin_lat": origin_lat.round(6),
            "origin_lon": origin_lon.round(6),
            "destination_lat": destination_lat.round(6),
            "destination_lon": destination_lon.round(6),
            "planned_distance_km": planned_distance.round(3),
            "traffic_index": traffic_index.round(4),
            "weather_severity": weather.round(4),
            "vehicle_load_kg": vehicle_load.round(1),
            "stops_remaining": stops_remaining,
            "actual_travel_minutes": travel_minutes.round(3),
        }
    )


def generate_routing_stops(stop_count: int = 18, seed: int = 84) -> pd.DataFrame:
    if stop_count < 6:
        raise ValueError("stop_count must be at least 6")
    rng = np.random.default_rng(seed)
    depot_lat = 31.5204
    depot_lon = 74.3587

    cluster_centers = np.array(
        [
            [31.5750, 74.3000],
            [31.4700, 74.3150],
            [31.5050, 74.4450],
        ]
    )
    rows: list[dict[str, object]] = [
        {
            "stop_id": "depot",
            "latitude": depot_lat,
            "longitude": depot_lon,
            "demand_units": 0,
            "is_depot": 1,
        }
    ]
    for index in range(stop_count):
        cluster = cluster_centers[index % len(cluster_centers)]
        rows.append(
            {
                "stop_id": f"stop-{index + 1:02d}",
                "latitude": round(float(cluster[0] + rng.normal(0.0, 0.010)), 6),
                "longitude": round(float(cluster[1] + rng.normal(0.0, 0.011)), 6),
                "demand_units": 1,
                "is_depot": 0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trips-output", default="data/synthetic_eta_trips.csv")
    parser.add_argument("--stops-output", default="data/synthetic_routing_stops.csv")
    parser.add_argument("--rows", type=int, default=1600)
    parser.add_argument("--stops", type=int, default=18)
    parser.add_argument("--seed", type=int, default=84)
    args = parser.parse_args()

    trips_output = Path(args.trips_output)
    stops_output = Path(args.stops_output)
    trips_output.parent.mkdir(parents=True, exist_ok=True)
    stops_output.parent.mkdir(parents=True, exist_ok=True)

    trips = generate_eta_trips(rows=args.rows, seed=args.seed)
    stops = generate_routing_stops(stop_count=args.stops, seed=args.seed)
    trips.to_csv(trips_output, index=False)
    stops.to_csv(stops_output, index=False)
    print(f"wrote {len(trips)} ETA trips to {trips_output}")
    print(f"wrote {len(stops) - 1} delivery stops + depot to {stops_output}")


if __name__ == "__main__":
    main()
