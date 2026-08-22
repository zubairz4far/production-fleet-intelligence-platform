from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class MlflowReleaseResult:
    run_id: str
    model_uri: str
    registered_model_name: str | None


def log_eta_release(
    *,
    model_artifact: str | Path,
    evaluation_report: str | Path,
    tracking_uri: str | None = None,
    experiment_name: str = "fleet-intelligence-eta",
    registered_model_name: str | None = "fleet-intelligence-eta",
) -> MlflowReleaseResult:
    """Log the promoted ETA model and its frozen evaluation evidence to MLflow."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError('Install the "mlops" extra to use MLflow integration') from exc

    artifact_path = Path(model_artifact)
    report_path = Path(evaluation_report)
    bundle = joblib.load(artifact_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("ETA artifact must contain model and features")

    report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    eta = report.get("eta", report)
    candidate = eta.get("candidate", {})
    metrics = candidate.get("test_metrics", {})
    promotion = eta.get("promotion", {})
    if promotion.get("decision") != "promote":
        raise ValueError("Only a promoted ETA evaluation may be registered")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="eta-v0.5-release") as run:
        mlflow.log_params(
            {
                "release_version": "0.5.0",
                "source_evaluation_version": report.get("version", "unknown"),
                "model_type": type(bundle["model"]).__name__,
                "feature_count": len(bundle["features"]),
                "evaluation_split": eta.get("split", {}).get("strategy", "unknown"),
                "production_evidence": bool(report.get("production_evidence", False)),
            }
        )
        numeric_metrics = {
            f"eta_test_{key}": float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float)
        }
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        model_info = mlflow.sklearn.log_model(
            sk_model=bundle["model"],
            name="eta_model",
            registered_model_name=registered_model_name,
            metadata={
                "source": "production-fleet-intelligence-platform",
                "evaluation_decision": "promote",
                "feature_contract": list(bundle["features"]),
            },
        )
        return MlflowReleaseResult(
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            registered_model_name=registered_model_name,
        )
