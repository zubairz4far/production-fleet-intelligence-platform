from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fleet_intelligence.eta import evaluate_eta_models
from fleet_intelligence.geospatial import build_eta_inference_features
from fleet_intelligence.serving import EtaModelService, EtaPredictionRequest, create_app
from fleet_intelligence.streaming import decode_eta_event, process_one_message
from fleet_intelligence.synthetic_mobility import generate_eta_trips


@pytest.fixture()
def eta_release(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    trips = generate_eta_trips(rows=900, seed=84)
    artifact_dir = tmp_path / "eta"
    report = evaluate_eta_models(trips, artifact_dir=artifact_dir)
    assert report["promotion"]["decision"] == "promote"
    return artifact_dir / "eta_hist_gradient_boosting.joblib", trips


def _payload_from_trip(trip: pd.Series) -> dict[str, object]:
    return {
        "trip_id": str(trip["trip_id"]),
        "timestamp": str(trip["timestamp"]),
        "origin_lat": float(trip["origin_lat"]),
        "origin_lon": float(trip["origin_lon"]),
        "destination_lat": float(trip["destination_lat"]),
        "destination_lon": float(trip["destination_lon"]),
        "planned_distance_km": float(trip["planned_distance_km"]),
        "traffic_index": float(trip["traffic_index"]),
        "weather_severity": float(trip["weather_severity"]),
        "vehicle_load_kg": float(trip["vehicle_load_kg"]),
        "stops_remaining": int(trip["stops_remaining"]),
    }


def test_inference_feature_builder_does_not_require_target() -> None:
    trips = generate_eta_trips(rows=500, seed=84)
    payload = _payload_from_trip(trips.iloc[0])
    features = build_eta_inference_features(pd.DataFrame([payload]))
    assert features.shape == (1, 13)
    assert "actual_travel_minutes" not in features.columns


def test_fastapi_eta_prediction_and_prometheus_metrics(eta_release: tuple[Path, pd.DataFrame]) -> None:
    artifact, trips = eta_release
    client = TestClient(create_app(artifact))

    assert client.get("/health").json() == {"status": "live"}
    assert client.get("/ready").json() == {"status": "ready"}

    response = client.post("/predict", json=_payload_from_trip(trips.iloc[-1]))
    assert response.status_code == 200
    body = response.json()
    assert body["trip_id"] == trips.iloc[-1]["trip_id"]
    assert 0.0 < body["eta_minutes"] < 1440.0
    assert body["model"] == "HistGradientBoostingRegressor"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "fleet_eta_requests_total" in metrics.text
    assert 'status="success"' in metrics.text


def test_readiness_and_prediction_fail_closed_without_model() -> None:
    client = TestClient(create_app())
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
    assert client.post(
        "/predict",
        json={
            "trip_id": "trip-1",
            "timestamp": "2026-08-22T00:00:00Z",
            "origin_lat": 31.52,
            "origin_lon": 74.35,
            "destination_lat": 31.55,
            "destination_lon": 74.31,
            "planned_distance_km": 8.0,
            "traffic_index": 1.1,
            "weather_severity": 0.1,
            "vehicle_load_kg": 500.0,
            "stops_remaining": 4,
        },
    ).status_code == 503


class _Message:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def value(self) -> bytes:
        return self._payload

    def error(self) -> None:
        return None


class _Consumer:
    def __init__(self, message: _Message) -> None:
        self.message = message
        self.committed = False

    def poll(self, timeout: float) -> _Message:
        assert timeout >= 0.0
        return self.message

    def commit(self, message: _Message, asynchronous: bool = False) -> None:
        assert message is self.message
        assert asynchronous is False
        self.committed = True

    def close(self) -> None:
        return None


def test_kafka_worker_commits_only_after_successful_prediction(
    eta_release: tuple[Path, pd.DataFrame],
) -> None:
    artifact, trips = eta_release
    payload = _payload_from_trip(trips.iloc[-1])
    message = _Message(json.dumps(payload).encode("utf-8"))
    consumer = _Consumer(message)
    service = EtaModelService(artifact)

    response = process_one_message(consumer, service)
    assert response is not None
    assert response.trip_id == payload["trip_id"]
    assert consumer.committed is True


def test_invalid_kafka_event_is_not_committed() -> None:
    class _NeverCalledService:
        def predict(self, request: EtaPredictionRequest) -> None:
            raise AssertionError(f"unexpected request: {request}")

    consumer = _Consumer(_Message(b"not-json"))
    with pytest.raises(json.JSONDecodeError):
        process_one_message(consumer, _NeverCalledService())  # type: ignore[arg-type]
    assert consumer.committed is False


def test_decode_eta_event_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        decode_eta_event("[]")
