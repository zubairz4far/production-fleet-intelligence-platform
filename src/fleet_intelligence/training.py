from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import temporal_train_test_split
from .evaluation import evaluate_probabilities
from .schema import FEATURE_COLUMNS, summarize_dataset, validate_dataset


def build_baseline_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_baseline(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.25,
    false_negative_cost: float = 5.0,
    false_positive_cost: float = 1.0,
) -> tuple[Pipeline, dict[str, object]]:
    data = validate_dataset(frame)
    split = temporal_train_test_split(data, test_fraction=test_fraction)

    model = build_baseline_model()
    model.fit(split.x_train, split.y_train)
    probabilities = model.predict_proba(split.x_test)[:, 1]
    metrics = evaluate_probabilities(
        split.y_test.to_numpy(),
        probabilities,
        false_negative_cost=false_negative_cost,
        false_positive_cost=false_positive_cost,
    )
    summary = summarize_dataset(data)

    report: dict[str, object] = {
        "project": "production-fleet-intelligence-platform",
        "version": "0.1.0",
        "task": "failure_within_horizon",
        "model": "logistic_regression_balanced",
        "features": list(FEATURE_COLUMNS),
        "dataset": asdict(summary),
        "split": {
            "strategy": "chronological_holdout",
            "test_fraction": test_fraction,
            "train_rows": len(split.x_train),
            "test_rows": len(split.x_test),
            "train_end": split.train_end,
            "test_start": split.test_start,
        },
        "threshold_policy": {
            "false_negative_cost": false_negative_cost,
            "false_positive_cost": false_positive_cost,
            "search_range": [0.05, 0.95],
            "search_step": 0.01,
        },
        "metrics": metrics.to_dict(),
        "limitations": [
            "v0.1 uses a chronological row holdout, not a vehicle-group holdout.",
            "Synthetic benchmark data is for CI only and is not production evidence.",
            "Business-cost weights are configurable assumptions, not universal fleet economics.",
        ],
    }
    return model, report


def write_report(report: dict[str, object], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
