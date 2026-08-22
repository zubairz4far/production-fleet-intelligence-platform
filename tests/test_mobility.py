from __future__ import annotations

import numpy as np
import pytest

from fleet_intelligence.eta import evaluate_eta_models, temporal_eta_split
from fleet_intelligence.geospatial import (
    ETA_FEATURE_COLUMNS,
    ETA_TARGET,
    build_eta_features,
    haversine_km,
)
from fleet_intelligence.routing import benchmark_routing, validate_stop_dataset
from scripts.generate_synthetic_mobility import generate_eta_trips, generate_routing_stops


def test_haversine_is_zero_for_same_point_and_symmetric() -> None:
    same = haversine_km(31.5204, 74.3587, 31.5204, 74.3587)
    forward = haversine_km(31.5204, 74.3587, 31.5657, 74.3142)
    reverse = haversine_km(31.5657, 74.3142, 31.5204, 74.3587)
    assert float(same) == pytest.approx(0.0, abs=1e-9)
    assert float(forward) == pytest.approx(float(reverse), rel=1e-12)
    assert float(forward) > 0.0


def test_eta_features_exclude_target_and_preserve_chronology() -> None:
    trips = generate_eta_trips(rows=600, seed=84)
    features = build_eta_features(trips)
    split = temporal_eta_split(features)
    assert ETA_TARGET not in ETA_FEATURE_COLUMNS
    assert list(split.x_train.columns) == list(ETA_FEATURE_COLUMNS)
    assert split.train_end <= split.dev_start
    assert split.dev_end <= split.test_start
    assert np.isfinite(split.x_train.to_numpy()).all()


def test_eta_candidate_beats_predeclared_speed_baseline(tmp_path) -> None:
    trips = generate_eta_trips(rows=900, seed=84)
    report = evaluate_eta_models(trips, artifact_dir=tmp_path)
    baseline = report["baseline"]["test_metrics"]
    candidate = report["candidate"]["test_metrics"]
    assert report["promotion"]["decision"] == "promote"
    assert candidate["mae_minutes"] < baseline["mae_minutes"]
    assert candidate["rmse_minutes"] < baseline["rmse_minutes"]
    assert candidate["p90_absolute_error_minutes"] < baseline["p90_absolute_error_minutes"]
    assert (tmp_path / "eta_hist_gradient_boosting.joblib").exists()
    assert (tmp_path / "eta_metadata.json").exists()


def test_ortools_routing_satisfies_capacity_and_non_regresses_distance() -> None:
    stops = generate_routing_stops(stop_count=18, seed=84)
    report = benchmark_routing(stops, vehicle_count=3, vehicle_capacity=6)
    baseline = report["baseline"]["metrics"]
    candidate = report["candidate"]["metrics"]
    assert report["promotion"]["decision"] == "promote"
    assert candidate["served_stops"] == 18
    assert candidate["unserved_stops"] == 0
    assert candidate["capacity_violation_units"] == 0
    assert candidate["total_distance_km"] <= baseline["total_distance_km"] + 1e-6


def test_routing_rejects_infeasible_total_capacity() -> None:
    stops = generate_routing_stops(stop_count=18, seed=84)
    with pytest.raises(ValueError, match="Total demand exceeds"):
        benchmark_routing(stops, vehicle_count=2, vehicle_capacity=6)


def test_routing_requires_exactly_one_depot() -> None:
    stops = generate_routing_stops(stop_count=18, seed=84)
    stops["is_depot"] = 0
    with pytest.raises(ValueError, match="exactly one depot"):
        validate_stop_dataset(stops)
