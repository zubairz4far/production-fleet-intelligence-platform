from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from .geospatial import ETA_FEATURE_COLUMNS, build_eta_inference_features


class EtaPredictionRequest(BaseModel):
    trip_id: str = Field(min_length=1, max_length=128)
    timestamp: datetime
    origin_lat: float = Field(ge=-90.0, le=90.0)
    origin_lon: float = Field(ge=-180.0, le=180.0)
    destination_lat: float = Field(ge=-90.0, le=90.0)
    destination_lon: float = Field(ge=-180.0, le=180.0)
    planned_distance_km: float = Field(gt=0.0)
    traffic_index: float = Field(gt=0.0)
    weather_severity: float = Field(ge=0.0, le=1.0)
    vehicle_load_kg: float = Field(ge=0.0)
    stops_remaining: int = Field(ge=0)


class EtaPredictionResponse(BaseModel):
    trip_id: str
    eta_minutes: float
    model: str


class EtaModelService:
    def __init__(self, artifact_path: str | Path) -> None:
        path = Path(artifact_path)
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
            raise ValueError("ETA artifact must contain model and features")
        features = tuple(bundle["features"])
        if features != ETA_FEATURE_COLUMNS:
            raise ValueError("ETA artifact feature contract does not match serving contract")
        self.artifact_path = path
        self.model = bundle["model"]
        self.features = features
        self.model_name = type(self.model).__name__

    def predict(self, request: EtaPredictionRequest) -> EtaPredictionResponse:
        row = request.model_dump(mode="json")
        frame = pd.DataFrame([row])
        features = build_eta_inference_features(frame)
        value = float(self.model.predict(features.loc[:, list(self.features)])[0])
        if not 0.0 < value < 24.0 * 60.0:
            raise ValueError("ETA prediction is outside the serving safety range")
        return EtaPredictionResponse(
            trip_id=request.trip_id,
            eta_minutes=value,
            model=self.model_name,
        )


class ServiceMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "fleet_eta_requests_total",
            "ETA prediction requests",
            ["status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "fleet_eta_request_duration_seconds",
            "ETA prediction request latency",
            registry=self.registry,
        )
        self.ready = Gauge(
            "fleet_eta_model_ready",
            "Whether the ETA model is loaded and ready",
            registry=self.registry,
        )


def create_app(model_path: str | Path | None = None) -> FastAPI:
    resolved = model_path or os.getenv("ETA_MODEL_PATH")
    service: EtaModelService | None = None
    load_error: str | None = None
    if resolved:
        try:
            service = EtaModelService(resolved)
        except (OSError, ValueError, TypeError) as exc:
            load_error = str(exc)

    metrics = ServiceMetrics()
    metrics.ready.set(1 if service is not None else 0)

    app = FastAPI(
        title="Fleet Intelligence ETA API",
        version="0.5.0",
        description="Online ETA inference over the promoted v0.4 model artifact.",
    )
    app.state.model_service = service
    app.state.model_load_error = load_error
    app.state.metrics = metrics

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/ready")
    def ready() -> dict[str, str]:
        if app.state.model_service is None:
            detail = app.state.model_load_error or "ETA_MODEL_PATH is not configured"
            raise HTTPException(status_code=503, detail=detail)
        return {"status": "ready"}

    @app.post("/predict", response_model=EtaPredictionResponse)
    def predict(request: EtaPredictionRequest) -> EtaPredictionResponse:
        active: EtaModelService | None = app.state.model_service
        if active is None:
            metrics.requests.labels(status="not_ready").inc()
            raise HTTPException(status_code=503, detail="ETA model is not ready")
        started = time.perf_counter()
        try:
            response = active.predict(request)
        except ValueError as exc:
            metrics.requests.labels(status="invalid").inc()
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            metrics.latency.observe(time.perf_counter() - started)
        metrics.requests.labels(status="success").inc()
        return response

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
