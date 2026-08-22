from __future__ import annotations

import json
from pathlib import Path


def test_v0_5_ci_evidence_is_fail_closed_and_non_production() -> None:
    report = json.loads(Path("evals/results/mlops_v0.5_ci.json").read_text(encoding="utf-8"))

    assert report["version"] == "0.5.0"
    assert report["evidence_type"] == "ci_integration_validation"
    assert report["production_deployment_evidence"] is False

    ci = report["ci"]
    assert ci["pytest"] == "passed"
    assert ci["ruff"] == "passed"
    assert ci["docker_build"] == "passed"
    assert ci["container_runtime_smoke"]["status"] == "passed"
    assert ci["container_runtime_smoke"]["health_without_model_http_status"] == 200
    assert ci["container_runtime_smoke"]["ready_without_model_http_status"] == 503
    assert ci["terraform"]["validate"] == "passed"
    assert ci["v0_4_evaluation"] == "passed"

    mlflow = ci["mlflow"]
    assert mlflow["tracking"] == "passed"
    assert mlflow["registry"] == "passed"
    assert mlflow["serialization"] == "skops"
    assert mlflow["signature_present"] is True
    assert mlflow["registered_model_reload_prediction"] == "passed"

    assert ci["kafka"]["client_import"] == "passed"
    assert report["streaming_contract"]["commit_after_successful_scoring"] is True
    assert report["streaming_contract"]["invalid_event_not_committed"] is True
    assert report["infrastructure_scope"]["live_cluster_provisioned"] is False
    assert report["limitations"]
