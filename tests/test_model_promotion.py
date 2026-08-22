from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fleet_intelligence.data import temporal_train_dev_test_split
from fleet_intelligence.schema import FEATURE_COLUMNS
from fleet_intelligence.training import PlattCalibrator, compare_models


def make_promotion_frame(rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    timestamps = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    service = np.mod(np.arange(rows) * 7, 520).astype(float)
    engine_temp = 84 + 0.018 * service + rng.normal(0, 2.3, rows)
    vibration = 1.4 + 0.006 * service + rng.normal(0, 0.30, rows)
    oil_pressure = 305 - 0.085 * service + rng.normal(0, 10, rows)
    battery = 13.5 - 0.0014 * service + rng.normal(0, 0.15, rows)

    nonlinear_risk = (
        (service > 330).astype(float) * 1.1
        + (vibration > 3.6).astype(float) * 0.9
        + ((engine_temp > 93) & (oil_pressure < 275)).astype(float) * 1.2
    )
    score = -3.0 + nonlinear_risk + 0.004 * service + 0.35 * (vibration - 2.5)
    probability = 1 / (1 + np.exp(-score))
    labels = rng.binomial(1, np.clip(probability, 0.03, 0.85))

    return pd.DataFrame(
        {
            "vehicle_id": [f"veh-{i % 30:03d}" for i in range(rows)],
            "timestamp": timestamps.astype(str),
            "mileage_km": 25_000 + np.arange(rows) * 15,
            "engine_temp_c": engine_temp,
            "oil_pressure_kpa": oil_pressure,
            "battery_voltage": battery,
            "vibration_rms": vibration,
            "fuel_rate_lph": 7.8 + 0.00002 * np.arange(rows) + rng.normal(0, 0.5, rows),
            "hours_since_service": service,
            "failure_within_horizon": labels,
        }
    )


def test_train_dev_test_split_is_chronological_and_leakage_safe() -> None:
    split = temporal_train_dev_test_split(make_promotion_frame())
    assert split.train_end <= split.dev_start <= split.dev_end <= split.test_start
    assert list(split.x_train.columns) == list(FEATURE_COLUMNS)
    assert "vehicle_id" not in split.x_dev
    assert "timestamp" not in split.x_test
    assert len(split.x_train) == 360
    assert len(split.x_dev) == 120
    assert len(split.x_test) == 120


def test_platt_calibrator_returns_bounded_probabilities() -> None:
    raw = np.array([0.02, 0.10, 0.25, 0.55, 0.80, 0.95])
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = PlattCalibrator().fit(raw, labels)
    calibrated = calibrator.predict(raw)
    assert np.all(calibrated > 0)
    assert np.all(calibrated < 1)
    assert np.all(np.diff(calibrated) > 0)


def test_comparison_uses_dev_for_thresholds_and_test_for_promotion(tmp_path) -> None:
    report = compare_models(make_promotion_frame(), artifact_dir=tmp_path)

    assert report["version"] == "0.2.0"
    assert report["split"]["strategy"] == "chronological_train_dev_test"
    assert report["threshold_policy"]["selection_partition"] == "development"
    assert report["promotion"]["test_observed_once"] is True
    assert report["promotion"]["decision"] in {"promote", "reject"}
    assert set(report["promotion"]["criteria"]) == {
        "pr_auc_non_regression",
        "brier_non_regression",
        "business_cost_non_regression",
    }

    assert (tmp_path / "baseline_logistic.joblib").exists()
    assert (tmp_path / "candidate_calibrated.joblib").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["features"] == list(FEATURE_COLUMNS)
    assert 0 < metadata["candidate_threshold"] < 1


def test_post_test_importance_is_marked_non_tuning() -> None:
    report = compare_models(make_promotion_frame())
    analysis = report["post_test_analysis"]
    assert analysis["used_for_model_selection"] is False
    assert len(analysis["feature_importance"]) == len(FEATURE_COLUMNS)
