from __future__ import annotations

import numpy as np
import pandas as pd

from fleet_intelligence.anomaly import evaluate_anomaly_detection, temporal_anomaly_split
from fleet_intelligence.telemetry import (
    ANOMALY_TARGET,
    anomaly_feature_columns,
    build_past_only_telemetry_features,
    population_stability_index,
    simulate_sensor_drift,
)


def make_anomaly_frame(rows: int = 800, vehicles: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    positions = np.arange(rows)
    vehicle_index = positions % vehicles
    service = np.mod(positions * 5 + vehicle_index * 7, 520).astype(float)
    mileage = 28_000 + positions * 16 + vehicle_index * 500
    engine_temp = 86 + 0.014 * service + rng.normal(0, 2.5, rows)
    oil_pressure = 305 - 0.08 * service + rng.normal(0, 8.0, rows)
    battery_voltage = 13.5 - 0.0012 * service + rng.normal(0, 0.12, rows)
    vibration = 1.8 + 0.004 * service + rng.normal(0, 0.22, rows)
    fuel_rate = 8.0 + 0.00002 * mileage + rng.normal(0, 0.35, rows)
    anomaly = np.zeros(rows, dtype=int)

    for offset, position in enumerate(range(int(rows * 0.60), rows, 7)):
        anomaly[position] = 1
        if offset % 3 == 0:
            engine_temp[position] += 18
            vibration[position] += 2.0
        elif offset % 3 == 1:
            oil_pressure[position] -= 75
            battery_voltage[position] -= 1.4
        else:
            fuel_rate[position] += 3.2
            vibration[position] += 1.6

    return pd.DataFrame(
        {
            "vehicle_id": [f"veh-{index:03d}" for index in vehicle_index],
            "timestamp": pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC").astype(str),
            "mileage_km": mileage,
            "engine_temp_c": engine_temp,
            "oil_pressure_kpa": oil_pressure,
            "battery_voltage": battery_voltage,
            "vibration_rms": vibration,
            "fuel_rate_lph": fuel_rate,
            "hours_since_service": service,
            ANOMALY_TARGET: anomaly,
        }
    )


def test_past_only_features_are_not_changed_by_future_observation() -> None:
    frame = make_anomaly_frame(rows=500)
    vehicle = "veh-003"
    vehicle_rows = frame.index[frame["vehicle_id"] == vehicle].tolist()
    target_row = vehicle_rows[5]
    future_row = vehicle_rows[6]
    timestamp = frame.loc[target_row, "timestamp"]

    original = build_past_only_telemetry_features(frame)
    mutated = frame.copy()
    mutated.loc[future_row, "engine_temp_c"] += 500
    changed = build_past_only_telemetry_features(mutated)

    original_row = original.loc[
        (original["vehicle_id"] == vehicle) & (original["timestamp"].astype(str) == timestamp)
    ]
    changed_row = changed.loc[
        (changed["vehicle_id"] == vehicle) & (changed["timestamp"].astype(str) == timestamp)
    ]
    np.testing.assert_allclose(
        original_row[list(anomaly_feature_columns())].to_numpy(dtype=float),
        changed_row[list(anomaly_feature_columns())].to_numpy(dtype=float),
        equal_nan=True,
    )


def test_anomaly_split_is_chronological_and_excludes_metadata() -> None:
    table = build_past_only_telemetry_features(make_anomaly_frame())
    split = temporal_anomaly_split(table)
    assert split.train_end <= split.dev_start <= split.dev_end <= split.test_start
    assert list(split.x_train.columns) == list(anomaly_feature_columns())
    assert "vehicle_id" not in split.x_train.columns
    assert "timestamp" not in split.x_train.columns
    assert split.y_train.sum() == 0


def test_population_stability_index_reacts_to_simulated_drift() -> None:
    frame = make_anomaly_frame()
    reference = frame.iloc[:480]
    current = frame.iloc[640:]
    drifted = simulate_sensor_drift(current)

    clean = population_stability_index(reference["engine_temp_c"], current["engine_temp_c"])
    shifted = population_stability_index(reference["engine_temp_c"], drifted["engine_temp_c"])
    assert shifted > clean
    assert shifted >= 0.25


def test_v03_anomaly_report_contains_model_and_system_gates(tmp_path) -> None:
    report = evaluate_anomaly_detection(make_anomaly_frame(), artifact_dir=tmp_path)

    assert report["version"] == "0.3.0"
    assert report["feature_engineering"]["feature_count"] == len(anomaly_feature_columns())
    assert report["promotion"]["decision"] in {"promote", "reject"}
    assert report["promotion"]["test_observed_once"] is True
    assert report["threshold_policy"]["selection_partition"] == "development"
    assert report["drift"]["simulated_max_psi"] > report["drift"]["clean_max_psi"]
    assert report["drift"]["simulated_high_drift_features"]

    robustness = report["robustness"]["missing_sensor_families"]
    assert 0 <= robustness["pr_auc_drop"] <= 1
    assert 0 <= robustness["recall_drop"] <= 1
    assert (tmp_path / "anomaly_isolation_forest.joblib").exists()
    assert (tmp_path / "anomaly_metadata.json").exists()


def test_unseen_vehicle_stress_test_is_reported() -> None:
    report = evaluate_anomaly_detection(make_anomaly_frame(rows=1000, vehicles=25))
    stress = report["robustness"]["unseen_vehicle"]
    assert "available" in stress
    if stress["available"]:
        assert stress["held_out_vehicle_count"] == 5
        assert 0 <= stress["metrics"]["pr_auc"] <= 1
        assert 0 <= stress["metrics"]["roc_auc"] <= 1
