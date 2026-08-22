from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


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

    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
    except ImportError as exc:  # pragma: no cover - exercised when optional extra is absent
        raise RuntimeError('Install the "mlops" extra to use MLflow integration') from exc

    feature_names = list(bundle["features"])
    input_example = pd.DataFrame([{feature: 0.0 for feature in feature_names}])
    for feature in ("planned_distance_km", "great_circle_km", "detour_ratio", "traffic_index"):
        if feature in input_example.columns:
            input_example.loc[0, feature] = 1.0
    example_prediction = bundle["model"].predict(input_example)
    signature = infer_signature(input_example, example_prediction)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="eta-v0.5-release") as run:
        mlflow.log_params(
            {
                "release_version": "0.5.0",
                "source_evaluation_version": report.get("version", "unknown"),
                "model_type": type(bundle["model"]).__name__,
                "feature_count": len(feature_names),
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
            serialization_format="skops",
            signature=signature,
            input_example=input_example,
            metadata={
                "source": "production-fleet-intelligence-platform",
                "evaluation_decision": "promote",
                "feature_contract": feature_names,
            },
        )
        return MlflowReleaseResult(
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
            registered_model_name=registered_model_name,
        )
