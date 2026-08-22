from __future__ import annotations

from pathlib import Path

import pandas as pd

from .eta import evaluate_eta_models
from .routing import benchmark_routing


def evaluate_mobility_release(
    trips: pd.DataFrame,
    stops: pd.DataFrame,
    *,
    dev_fraction: float = 0.20,
    test_fraction: float = 0.20,
    vehicle_count: int = 3,
    vehicle_capacity: int = 6,
    artifact_dir: str | Path | None = None,
) -> dict[str, object]:
    destination = Path(artifact_dir) if artifact_dir is not None else None
    eta_artifacts = destination / "eta" if destination is not None else None

    eta_report = evaluate_eta_models(
        trips,
        dev_fraction=dev_fraction,
        test_fraction=test_fraction,
        artifact_dir=eta_artifacts,
    )
    routing_report = benchmark_routing(
        stops,
        vehicle_count=vehicle_count,
        vehicle_capacity=vehicle_capacity,
    )

    release_passed = (
        eta_report["promotion"]["decision"] == "promote"
        and routing_report["promotion"]["decision"] == "promote"
    )
    return {
        "project": "production-fleet-intelligence-platform",
        "version": "0.4.0",
        "release": "eta_prediction_and_capacitated_routing",
        "eta": eta_report,
        "routing": routing_report,
        "release_gate": {
            "decision": "promote" if release_passed else "reject",
            "criteria": {
                "eta_gate_passed": eta_report["promotion"]["decision"] == "promote",
                "routing_gate_passed": routing_report["promotion"]["decision"] == "promote",
            },
        },
        "evidence_policy": {
            "synthetic_ci_only": True,
            "production_evidence": False,
            "note": (
                "Synthetic trip and stop fixtures validate evaluation and optimization mechanics; "
                "they do not establish real fleet ETA or routing performance."
            ),
        },
    }
