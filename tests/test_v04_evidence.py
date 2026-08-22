from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_committed_v04_evidence_matches_frozen_release_contract() -> None:
    path = Path("evals/results/mobility_v0.4_synthetic.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["version"] == "0.4.0"
    assert evidence["evidence_type"] == "synthetic_ci_software_regression"
    assert evidence["production_evidence"] is False
    assert evidence["fixture"]["seed"] == 84

    eta = evidence["eta"]
    assert eta["split"]["strategy"] == "chronological_train_dev_test"
    assert eta["promotion"]["decision"] == "promote"
    assert eta["promotion"]["test_observed_once"] is True
    assert all(eta["promotion"]["criteria"].values())
    assert eta["candidate"]["test_metrics"]["mae_minutes"] == pytest.approx(
        2.8878848528504726
    )
    assert eta["baseline"]["test_metrics"]["mae_minutes"] == pytest.approx(
        7.787422497137105
    )

    routing = evidence["routing"]
    assert routing["promotion"]["decision"] == "promote"
    assert all(routing["promotion"]["criteria"].values())
    assert routing["candidate"]["metrics"]["served_stops"] == 18
    assert routing["candidate"]["metrics"]["unserved_stops"] == 0
    assert routing["candidate"]["metrics"]["capacity_violation_units"] == 0
    assert routing["promotion"]["distance_improvement_percent"] == pytest.approx(
        6.874457772280053
    )

    assert evidence["release_gate"]["decision"] == "promote"
