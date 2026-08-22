from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fleet_intelligence.data import temporal_train_test_split
from fleet_intelligence.evaluation import evaluate_probabilities, select_threshold_by_cost
from fleet_intelligence.schema import FEATURE_COLUMNS, validate_dataset
from fleet_intelligence.training import train_baseline


def make_frame(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    service = np.mod(np.arange(rows) * 9, 500).astype(float)
    vibration = 1.5 + service * 0.006 + rng.normal(0, 0.25, rows)
    engine_temp = 84 + service * 0.02 + rng.normal(0, 2.0, rows)
    score = -5.5 + 0.8 * (vibration - 2.2) + 0.05 * (engine_temp - 90) + 0.006 * service
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, np.clip(probability, 0.02, 0.9))

    return pd.DataFrame(
        {
            "vehicle_id": [f"veh-{i % 20:03d}" for i in range(rows)],
            "timestamp": timestamps.astype(str),
            "mileage_km": 30_000 + np.arange(rows) * 12,
            "engine_temp_c": engine_temp,
            "oil_pressure_kpa": 300 - service * 0.08 + rng.normal(0, 8, rows),
            "battery_voltage": 13.4 - service * 0.001 + rng.normal(0, 0.1, rows),
            "vibration_rms": vibration,
            "fuel_rate_lph": 8 + rng.normal(0, 0.4, rows),
            "hours_since_service": service,
            "failure_within_horizon": labels,
        }
    )


def test_schema_rejects_missing_columns() -> None:
    frame = make_frame().drop(columns=["vibration_rms"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(frame)


def test_temporal_split_excludes_identifiers_and_future_rows() -> None:
    split = temporal_train_test_split(make_frame(), test_fraction=0.25)
    assert list(split.x_train.columns) == list(FEATURE_COLUMNS)
    assert "vehicle_id" not in split.x_train
    assert "timestamp" not in split.x_train
    assert split.train_end <= split.test_start
    assert len(split.x_test) == 80


def test_false_negative_cost_can_change_threshold_policy() -> None:
    y_true = np.array([0, 0, 0, 0, 1, 1])
    probabilities = np.array([0.10, 0.20, 0.40, 0.45, 0.35, 0.70])
    low_fn_cost = select_threshold_by_cost(y_true, probabilities, false_negative_cost=1)
    high_fn_cost = select_threshold_by_cost(y_true, probabilities, false_negative_cost=10)
    assert high_fn_cost.false_negatives <= low_fn_cost.false_negatives
    assert high_fn_cost.threshold <= low_fn_cost.threshold


def test_evaluation_returns_probability_and_operating_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.30, 0.65, 0.90])
    metrics = evaluate_probabilities(y_true, probabilities)
    assert metrics.pr_auc > 0.9
    assert metrics.roc_auc == 1.0
    assert 0 <= metrics.brier_score <= 1
    assert 0.05 <= metrics.threshold <= 0.95


def test_training_report_records_leakage_and_cost_contract() -> None:
    _, report = train_baseline(make_frame())
    assert report["split"]["strategy"] == "chronological_holdout"
    assert report["threshold_policy"]["false_negative_cost"] == 5.0
    assert set(report["features"]) == set(FEATURE_COLUMNS)
    metrics = report["metrics"]
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
